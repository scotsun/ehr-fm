import torch
import torch.nn as nn
import mlflow
import torch.distributed as dist

from abc import ABC, abstractmethod
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from mlflow.models import ModelSignature

from src.utils.dist_utils import (
    is_main_process,
    _get_module,
    _broadcast_bool,
    _broadcast_float,
)


@torch.no_grad()
def _copy_state_dict_cpu(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        mode: str,
        min_delta: float = 0.0,
        save_best_weights: bool = True,
    ) -> None:
        if mode not in ["min", "max"]:
            raise ValueError("mode must be either `min` or `max`")
        if patience < 1:
            raise ValueError("patience must be at least 1")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score = None
        self.early_stop = False
        self.save_best_weights = save_best_weights
        self.best_epoch = None
        self.bad_epoch = 0
        self.should_stop = False
        self.best_state_dict = None

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        else:  # "max"
            return current > best + self.min_delta

    def step(
        self,
        metric_val: float,
        model: nn.Module | DDP,
        epoch: int,
        device: torch.device,
    ) -> tuple[bool, bool]:
        if is_main_process():
            improved = False
            if self.best_score is None or self._is_better(metric_val, self.best_score):
                self.best_score = float(metric_val)
                self.best_epoch = int(epoch)
                self.bad_epoch = 0
                improved = True
            else:
                self.bad_epoch += 1

            if improved and self.save_best_weights:
                module = _get_module(model)
                self.best_state_dict = _copy_state_dict_cpu(module.state_dict())

            self.should_stop = self.bad_epoch >= self.patience
        else:
            improved = False

        improved = _broadcast_bool(improved, device)
        self.should_stop = _broadcast_bool(
            self.should_stop if is_main_process() else False, device=device
        )

        if is_main_process():
            score_to_bcast = (
                float(self.best_score)
                if self.best_score is not None
                else float(metric_val)
            )
        else:
            score_to_bcast = 0.0
        self.best_score = _broadcast_float(score_to_bcast, device)
        return improved, self.should_stop

    def load_best_weights(self, model: nn.Module | DDP):
        # Ensure every rank gets the same best weights in DDP.
        if dist.is_available() and dist.is_initialized():
            obj_list = [self.best_state_dict if is_main_process() else None]
            dist.broadcast_object_list(obj_list, src=0)
            self.best_state_dict = obj_list[0]

        if self.best_state_dict is not None:
            module = _get_module(model)
            module.load_state_dict(self.best_state_dict, strict=True)


class Trainer(ABC):
    """
    external state-tracking & mlflow state-logging in the end
    """

    def __init__(
        self,
        model: nn.Module | DDP,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        model_signature: ModelSignature,
    ) -> None:
        self.model: nn.Module | DDP = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.verbose_period = verbose_period
        self.device = device
        self.model_signature = model_signature
        self.scaler = GradScaler(device=device)

    def log_model(self, model_name: str = "best_model"):
        if not is_main_process():
            return
        module = _get_module(self.model)
        mlflow.pytorch.log_model(
            pytorch_model=module,
            name=model_name,
            pip_requirements=["torch>=2.5"],
            signature=self.model_signature,
        )
        print(f"[INFO]: logged model {model_name}")

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        log_epoch_model_period: int = 10,
    ):
        if self.early_stopping and not valid_loader:
            raise ValueError("EarlyStopping must be accompanied by valid data.")

        for epoch in range(epochs):
            verbose = ((epoch % self.verbose_period) == 0) and is_main_process()
            self._train(train_loader, verbose, epoch)

            if valid_loader is not None:
                valid_metrics = self._valid(valid_loader, verbose, epoch)

                if is_main_process():
                    mlflow.log_metrics(valid_metrics["logged_metrics"], step=epoch)
                if self.early_stopping is not None:
                    _, stop = self.early_stopping.step(
                        metric_val=float(valid_metrics["callback_metric"]),
                        model=self.model,
                        epoch=epoch,
                        device=self.device,
                    )
                else:
                    stop = False
            else:
                stop = False

            if is_main_process() and (epoch % log_epoch_model_period) == 0:
                self.log_model(model_name=f"epoch-{epoch}")

            stop = _broadcast_bool(stop if is_main_process() else False, self.device)

            if stop:
                if self.early_stopping is not None:
                    # load the best weight on every rank before saving the model
                    # to ensure the best model is saved and returned in the end
                    self.early_stopping.load_best_weights(self.model)
                if is_main_process():
                    self.log_model(model_name="best_model")
                return

        if self.early_stopping is not None:
            self.early_stopping.load_best_weights(self.model)
        if is_main_process():
            self.log_model(model_name="best_model")
        return

    @abstractmethod
    def evaluate(self, **kwarg):
        raise NotImplementedError

    @abstractmethod
    def _train(self, **kwarg):
        raise NotImplementedError

    @abstractmethod
    def _valid(self, **kwarg):
        raise NotImplementedError
