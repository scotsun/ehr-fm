"""Trainer classes & training infra functionalities integrated with MLflow."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist
import mlflow

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import GradScaler, autocast
from mlflow.models import ModelSignature
from tokenizers import Tokenizer
from tqdm import tqdm

from src.utils.data_utils import random_masking
from src.models.base import FMBase
from src.metric import topk_accuracy, recall_at_k, ndcg_at_k


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


class BaseTrainer(Trainer):
    def __init__(
        self,
        model: FMBase | DDP,
        tokenizer: Tokenizer,
        optimizer: Optimizer,
        criterions: dict[str, nn.Module],
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        model_signature: ModelSignature | None,
        trainer_args: dict,
    ) -> None:
        super().__init__(
            model, optimizer, early_stopping, verbose_period, device, model_signature
        )
        self.tokenizer = tokenizer
        self.criterions = criterions
        self.trainer_args = trainer_args

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        model: FMBase | DDP = self.model
        model.train()
        scaler = self.scaler
        optimizer = self.optimizer
        criterions = self.criterions
        trainer_args = self.trainer_args
        device = self.device

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                set_attention_mask = batch["set_attention_mask"].to(device)
                t = batch["t"].to(device)

                masked_input_ids, labels = random_masking(
                    input_ids, self.tokenizer, trainer_args["mlm_probability"]
                )
                if (labels == -100).all():
                    continue
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits, _ = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        set_attention_mask=set_attention_mask,
                        t=t,
                    )

                    loss = criterions["cross_entropy"](
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                recall10 = recall_at_k(logits, input_ids, set_attention_mask, 10)
                ndcg10 = ndcg_at_k(logits, input_ids, set_attention_mask, 10)

                bar.set_postfix(
                    mlm_loss=float(loss), recall10=float(recall10), ndcg10=float(ndcg10)
                )

                cur_step = epoch_id * len(dataloader) + batch_id
                if self._is_main_process() and cur_step % 100 == 0:
                    mlflow.log_metrics({"train_mlm_loss": float(loss)}, step=cur_step)
        return

    def evaluate(self, dataloader: DataLoader, verbose: bool) -> float:
        model: FMBase = self.model
        model.eval()
        device = self.device
        criterions = self.criterions
        trainer_args = self.trainer_args

        # total_mlm
        # total_top1_acc, total_top10_acc
        # total_recall@k, total_ndcg@k
        counter = torch.zeros(5, device=device)
        with torch.no_grad():
            for batch in tqdm(
                dataloader, unit="batch", mininterval=0, disable=not verbose
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                set_attention_mask = batch["set_attention_mask"].to(device)
                t = batch["t"].to(device)

                masked_input_ids, labels = random_masking(
                    input_ids, self.tokenizer, trainer_args["mlm_probability"]
                )
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits, _ = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        set_attention_mask=set_attention_mask,
                        t=t,
                    )
                    loss = criterions["cross_entropy"](
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )
                top1_acc = topk_accuracy(logits, labels, 1)
                top10_acc = topk_accuracy(logits, labels, 10)
                recall10 = recall_at_k(logits, input_ids, set_attention_mask, 10)
                ndcg10 = ndcg_at_k(logits, input_ids, set_attention_mask, 10)
                if recall10.item() > 1:
                    raise ValueError("!!!")
                counter[0] += loss.item()
                counter[1] += top1_acc.item()
                counter[2] += top10_acc.item()
                counter[3] += recall10.item()
                counter[4] += ndcg10.item()
        dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        return (
            (counter[0] / len(dataloader)).item(),
            (counter[1] / len(dataloader)).item(),
            (counter[2] / len(dataloader)).item(),
            (counter[3] / len(dataloader)).item(),
            (counter[4] / len(dataloader)).item(),
        )

    def _valid(self, dataloader, verbose, epoch_id):
        """
        log all the metrics in mlflow;
        return the metric for save-best/early-stop.
            {
                "callback_metric": ...,
                "logged_metrics": {...}
            }
        """
        val_mlm, val_top1, val_top10, val_recall10, val_ndcg10 = self.evaluate(
            dataloader, verbose
        )
        if verbose:
            print(
                f"epoch {epoch_id}/val_mlm_loss: {round(val_mlm, 3)}/"
                f"val_top1_acc: {round(val_top1, 3)}/"
                f"val_top10_acc: {round(val_top10, 3)}/"
                f"val_recall@10: {round(val_recall10, 3)}/"
                f"val_ndcg@10: {round(val_ndcg10, 3)}"
            )

        valid_metrics = {
            "callback_metric": val_mlm,
            "logged_metrics": {
                "val_mlm_loss": val_mlm,
                "val_top1_acc": val_top1,
                "val_top10_acc": val_top10,
                "val_recall10": val_recall10,
                "val_ndcg10": val_ndcg10,
            },
        }
        return valid_metrics


class BinaryTrainer(Trainer):
    def __init__(
        self,
        fm: FMBase,
        model: nn.Module,
        optimizer,
        early_stopping,
        criterion,
        verbose_period,
        device,
        model_signature,
        outcome_name: str,
        fm_freeze: bool = True,
    ) -> None:
        super().__init__(
            model, optimizer, early_stopping, verbose_period, device, model_signature
        )
        self.fm = fm
        self.fm_freeze = fm_freeze
        if fm_freeze:
            for p in fm.parameters():
                p.requires_grad = False
        self.criterion = criterion
        self.outcome_name = outcome_name


def test_logging(
    trainer: Trainer,
    test_loader: DataLoader,
    metric_names: list[str],
    expr_name: str,
    run_name: str,
):
    mlflow.set_experiment(expr_name)
    run_data = mlflow.search_runs(filter_string=f"attributes.run_name = '{run_name}'")
    if run_data.empty:
        raise ValueError(f"run_name={run_name} does not exist")
    else:
        run_id = run_data.iloc[0].run_id
        try:
            with mlflow.start_run(run_id=run_id):
                trainer.model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model")
                test_scores = trainer.evaluate(test_loader, True, 0)
                mlflow.log_metrics(metrics=dict(zip(metric_names, test_scores)), step=0)
            print(f"Successfully added metrics to run {run_id}")
        except Exception as e:
            print(f"Error updating run: {e}")
