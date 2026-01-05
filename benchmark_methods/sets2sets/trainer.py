"""Trainer classes & training infra functionalities integrated with MLflow."""

from abc import ABC, abstractmethod

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import mlflow

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import GradScaler, autocast
from mlflow.models import ModelSignature
from tokenizers import Tokenizer
from tqdm import tqdm


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        min_delta: int = 0,
        mode: str = "min",
        save_best_weights: bool = True,
    ) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_best_weights = save_best_weights
        self.best_weights = None
        self.best_epoch = None
        self.mode = mode

        match mode:
            case "min":
                self.monitor_op, self.delta_op = lambda a, b: a < b, -1 * min_delta
            case "max":
                self.monitor_op, self.delta_op = lambda a, b: a > b, min_delta
            case _:
                raise ValueError("mode must be either `min` or `max`")

    def _is_main_process(self):
        return (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )

    def _track_best_state(self, model_to_track: nn.Module):
        if self._is_main_process():
            print("[INFO]: log state")
            self.best_model_state = model_to_track.state_dict().copy()
        return

    def step(
        self,
        metric_val: float,
        model: nn.Module | DDP,
        epoch: int,
    ):
        model_to_log = model.module if hasattr(model, "module") else model
        if self.best_score is None:
            self.best_score = metric_val
            self.counter = 0
            self.best_epoch = epoch
            self._track_best_state(model_to_log)
            return

        if self.monitor_op(metric_val, self.best_score + self.delta_op):
            self.best_score = metric_val
            self.counter = 0
            self.best_epoch = epoch
            self._track_best_state(model_to_log)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return

    def listen_to_broadcast(self, device: torch.device):
        if self._is_main_process():
            flag = self.early_stop
        else:
            flag = False
        flag_tensor = torch.tensor([flag], device=device)
        dist.broadcast(flag_tensor, src=0)
        return bool(flag_tensor.item())


class Trainer(ABC):
    """
    external state-tracking & mlflow state-logging in the end
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        model_signature: ModelSignature,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.verbose_period = verbose_period
        self.device = device
        self.model_signature = model_signature
        self.scaler = GradScaler(device=device)

    def _is_main_process(self):
        return (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )

    def log_model(self, model: nn.Module | DDP, model_name: str = "best_model"):
        if self._is_main_process():
            model_to_log = model.module if hasattr(model, "module") else model
            if self.early_stopping:
                model_to_log.load_state_dict(self.early_stopping.best_model_state)
            mlflow.pytorch.log_model(
                model_to_log,
                name=model_name,
                pip_requirements=["torch>=2.5"],
                signature=self.model_signature,
            )
            print("[INFO]: logged model")
        return

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: None | DataLoader = None,
    ):
        if self.early_stopping and not valid_loader:
            raise ValueError("EarlyStopping must be accompanied by valid data.")
        for epoch in range(epochs):
            verbose = ((epoch % self.verbose_period) == 0) and self._is_main_process()
            self._train(train_loader, verbose, epoch)
            if valid_loader:
                valid_metrics = self._valid(valid_loader, verbose, epoch)
                if self._is_main_process():
                    mlflow.log_metrics(valid_metrics["logged_metrics"], step=epoch)
                    if self.early_stopping:
                        self.early_stopping.step(
                            valid_metrics["callback_metric"], self.model, epoch
                        )
            stop = self.early_stopping.listen_to_broadcast(self.device)
            if (epoch % 10) == 0:
                self.log_model(self.model, model_name=f"epoch-{epoch}")
            if stop:
                self.log_model(self.model)
                return
        self.log_model(self.model)
        return

    @abstractmethod
    def evaluate(self, **kwarg):
        pass

    @abstractmethod
    def _train(self, **kwarg):
        pass

    @abstractmethod
    def _valid(self, **kwarg):
        pass


class Sets2SetsTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        early_stopping,
        verbose_period,
        device,
        model_signature,
    ):
        super().__init__(
            model,
            optimizer,
            early_stopping,
            verbose_period,
            device,
            model_signature,
        )
