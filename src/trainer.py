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

from src.utils.data_utils import (
    random_masking,
    random_masking_set,
    masking_last_set,
    masking_last_set_1d,
    observed_set_distribution,
)
from src.models.base import FMBase, FMBaseWithHeads
from src.models.bert import FMBert
from src.metric import (
    topk_accuracy,
    recall_at_k,
    ndcg_at_k,
    select_last_set,
    pred_and_target_sets,
    pred_and_target_sets_1d,
)
from src.utils.train_utils import (
    _is_main_process,
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
        elif self.mode == "max":
            return current > best + self.min_delta
        else:
            raise ValueError("mode must be either `min` or `max`")

    def step(
        self,
        metric_val: float,
        model: nn.Module | DDP,
        epoch: int,
        device: torch.device,
    ):
        if _is_main_process():
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
            self.should_stop if _is_main_process() else False, device=device
        )

        if _is_main_process():
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
            obj_list = [self.best_state_dict if _is_main_process() else None]
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
        if not _is_main_process():
            return
        module = _get_module(self.model)
        mlflow.pytorch.log_model(
            model_to_log=module,
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
        log_epoch_model_period: int = 5,
    ):
        if self.early_stopping and not valid_loader:
            raise ValueError("EarlyStopping must be accompanied by valid data.")

        for epoch in range(epochs):
            verbose = ((epoch % self.verbose_period) == 0) and _is_main_process()
            self._train(train_loader, verbose, epoch)

            if valid_loader is not None:
                valid_metrics = self._valid(valid_loader, verbose, epoch)

                if _is_main_process():
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

            if _is_main_process() and (epoch % log_epoch_model_period) == 0:
                self.log_model(model_name=f"epoch-{epoch}")

            stop = _broadcast_bool(stop if _is_main_process() else False, self.device)

            if stop:
                if self.early_stopping is not None:
                    # load the best weight on every rank before saving the model
                    # to ensure the best model is saved and returned in the end
                    self.early_stopping.load_best_weights(self.model)
                if _is_main_process():
                    self.log_model(model_name="best_model")
                return

        if self.early_stopping is not None:
            self.early_stopping.load_best_weights(self.model)
        if _is_main_process():
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


class BertTrainer(Trainer):
    def __init__(
        self,
        model: FMBert | DDP,
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
        model: FMBert | DDP = self.model
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
                t = batch["t"].to(device)

                masked_input_ids, labels = random_masking(
                    input_ids.clone(), self.tokenizer, trainer_args["mlm_probability"]
                )
                if (labels == -100).all():
                    continue
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits, _ = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        t=t,
                    )

                    loss = criterions["cross_entropy"](
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bar.set_postfix(mlm_loss=float(loss))

                cur_step = epoch_id * len(dataloader) + batch_id
                if _is_main_process() and cur_step % 100 == 0:
                    mlflow.log_metrics({"train_mlm_loss": float(loss)}, step=cur_step)
        return

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool) -> torch.Tensor:
        model: FMBert | DDP = self.model
        model.eval()
        device = self.device
        criterions = self.criterions
        trainer_args = self.trainer_args

        # num_batch
        # total_mlm
        # total_top1_acc, total_top10_acc
        # optional: total_recall@k, total_ndcg@k
        counter = torch.zeros(6, device=device)
        for batch in tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            t = batch["t"].to(device)

            if trainer_args["eval_last_set"]:
                masked_last_set_input_ids = masking_last_set_1d(
                    input_ids.clone(), self.tokenizer
                )
            masked_input_ids, labels = random_masking(
                input_ids.clone(), self.tokenizer, trainer_args["mlm_probability"]
            )
            with autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    t=t,
                )
                loss = criterions["cross_entropy"](
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
                if trainer_args["eval_last_set"]:
                    masked_last_set_logts, _ = model(
                        input_ids=masked_last_set_input_ids,
                        attention_mask=attention_mask,
                        t=t,
                    )

            top1_acc = topk_accuracy(logits, labels, 1)
            top10_acc = topk_accuracy(logits, labels, 10)

            if trainer_args["eval_last_set"]:
                mask_pos = (masked_last_set_input_ids == 3).argwhere()
                p_tokens, t_tokens = pred_and_target_sets_1d(
                    masked_last_set_logts,
                    input_ids,
                    mask_pos,
                    10,
                )
                recall10 = recall_at_k(p_tokens, t_tokens)
                ndcg10 = ndcg_at_k(p_tokens, t_tokens)

            counter[0] += 1
            counter[1] += loss.item()
            counter[2] += top1_acc.item()
            counter[3] += top10_acc.item()
            if trainer_args["eval_last_set"]:
                counter[4] += recall10.item()
                counter[5] += ndcg10.item()
            else:
                recall10 = torch.zeros_like(top1_acc)
                ndcg10 = torch.zeros_like(top1_acc)

        if "LOCAL_RANK" in os.environ:
            dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        return (
            (counter[1] / counter[0]).item(),
            (counter[2] / counter[0]).item(),
            (counter[3] / counter[0]).item(),
            (counter[4] / counter[0]).item(),
            (counter[5] / counter[0]).item(),
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
            )
            if self.trainer_args["eval_last_set"]:
                print(
                    f"val_recall10: {round(val_recall10, 3)}/"
                    f"val_ndcg10: {round(val_ndcg10, 3)}/"
                )

        logged_metrics = {
            "val_mlm_loss": val_mlm,
            "val_top1_acc": val_top1,
            "val_top10_acc": val_top10,
        }
        if self.trainer_args["eval_last_set"]:
            logged_metrics["val_recall10"] = val_recall10
            logged_metrics["val_ndcg10"] = val_ndcg10
        valid_metrics = {
            "callback_metric": val_mlm,
            "logged_metrics": logged_metrics,
        }
        return valid_metrics


class LongformerTrainer(BertTrainer):
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        criterions,
        early_stopping,
        verbose_period,
        device,
        model_signature,
        trainer_args,
    ):
        super().__init__(
            model,
            tokenizer,
            optimizer,
            criterions,
            early_stopping,
            verbose_period,
            device,
            model_signature,
            trainer_args,
        )


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
                    input_ids.clone(), self.tokenizer, trainer_args["mlm_probability"]
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

                bar.set_postfix(mlm_loss=float(loss))

                cur_step = epoch_id * len(dataloader) + batch_id
                if _is_main_process() and cur_step % 100 == 0:
                    mlflow.log_metrics({"train_mlm_loss": float(loss)}, step=cur_step)
        return

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool) -> torch.Tensor:
        model: FMBase | DDP = self.model
        model.eval()
        device = self.device
        criterions = self.criterions
        trainer_args = self.trainer_args

        # num_batch
        # total_mlm
        # total_top1_acc, total_top10_acc
        # optional: total_recall@k, total_ndcg@k
        counter = torch.zeros(6, device=device)
        for batch in tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            set_attention_mask = batch["set_attention_mask"].to(device)
            t = batch["t"].to(device)

            if trainer_args["eval_last_set"]:
                masked_last_set_input_ids = masking_last_set(
                    input_ids.clone(), set_attention_mask, self.tokenizer
                )
            masked_input_ids, labels = random_masking(
                input_ids.clone(), self.tokenizer, trainer_args["mlm_probability"]
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
                if trainer_args["eval_last_set"]:
                    masked_last_set_logits, _ = model(
                        input_ids=masked_last_set_input_ids,
                        attention_mask=attention_mask,
                        set_attention_mask=set_attention_mask,
                        t=t,
                    )
            top1_acc = topk_accuracy(logits, labels, 1)
            top10_acc = topk_accuracy(logits, labels, 10)

            if trainer_args["eval_last_set"]:
                set_select_mask = select_last_set(set_attention_mask)
                p_tokens, t_tokens = pred_and_target_sets(
                    masked_last_set_logits, input_ids, set_select_mask, 10
                )
                recall10 = recall_at_k(p_tokens, t_tokens)
                ndcg10 = ndcg_at_k(p_tokens, t_tokens)

            counter[0] += 1
            counter[1] += loss.item()
            counter[2] += top1_acc.item()
            counter[3] += top10_acc.item()
            if trainer_args["eval_last_set"]:
                counter[4] += recall10.item()
                counter[5] += ndcg10.item()

        if "LOCAL_RANK" in os.environ:
            dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        return (
            (counter[1] / counter[0]).item(),
            (counter[2] / counter[0]).item(),
            (counter[3] / counter[0]).item(),
            (counter[4] / counter[0]).item(),
            (counter[5] / counter[0]).item(),
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
            )
            if self.trainer_args["eval_last_set"]:
                print(
                    f"epoch {epoch_id}/val_recall10: {round(val_recall10, 3)}/"
                    f"val_ndcg10: {round(val_ndcg10, 3)}/"
                )

        logged_metrics = {
            "val_mlm_loss": val_mlm,
            "val_top1_acc": val_top1,
            "val_top10_acc": val_top10,
        }
        if self.trainer_args["eval_last_set"]:
            logged_metrics["val_recall10"] = val_recall10
            logged_metrics["val_ndcg10"] = val_ndcg10
        valid_metrics = {
            "callback_metric": val_mlm,
            "logged_metrics": logged_metrics,
        }
        return valid_metrics


class BaseWithHeadsTrainer(Trainer):
    def __init__(
        self,
        model: FMBaseWithHeads | DDP,
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
        model: FMBaseWithHeads | DDP = self.model
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

                mlm_input_ids, mlm_labels = random_masking(
                    input_ids=input_ids.clone(),
                    tokenizer=self.tokenizer,
                    mlm_probability=trainer_args["mlm_probability"],
                )

                msm_input_ids, msm_labels, set_select_mask = random_masking_set(
                    input_ids=input_ids.clone(),
                    set_attention_mask=set_attention_mask,
                    tokenizer=self.tokenizer,
                    mask_probability=trainer_args["msm_probability"],
                )
                if (msm_labels == -100).all() or (mlm_labels == -100).all():
                    continue

                target_dist = observed_set_distribution(  # TODO: decide between masked-only or mixed
                    labels=msm_labels,
                    set_select_mask=set_select_mask,
                    # labels=input_ids,
                    # set_select_mask=set_attention_mask,
                    tokenizer=self.tokenizer,
                )

                # --- PASS 1: MLM ---
                with autocast(device_type="cuda", dtype=torch.float16):
                    mlm_logits, _, _ = model(
                        input_ids=mlm_input_ids,
                        attention_mask=attention_mask,
                        set_attention_mask=set_attention_mask,
                        t=t,
                        set_mask=set_select_mask,
                    )

                    mlm_loss = criterions["cross_entropy"](
                        mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)
                    )
                    scaled_mlm_loss = trainer_args["l_mlm"] * mlm_loss

                scaler.scale(scaled_mlm_loss).backward()

                del mlm_logits, scaled_mlm_loss

                # --- PASS 2: MSM ---
                with autocast(device_type="cuda", dtype=torch.float16):
                    _, msm_set_logits, _ = model(
                        input_ids=msm_input_ids,
                        attention_mask=attention_mask,
                        set_attention_mask=set_attention_mask,
                        t=t,
                        set_mask=set_select_mask,  # TODO: decide between masked-only or mixed
                        # set_mask=set_attention_mask,
                    )

                    msm_loss = criterions["kl_div"](
                        F.log_softmax(msm_set_logits, dim=-1), target_dist
                    )
                    scaled_msm_loss = trainer_args["l_msm"] * msm_loss

                scaler.scale(scaled_msm_loss).backward()

                # --- OPTIMIZER STEP ---
                scaler.step(optimizer)
                scaler.update()

                bar.set_postfix(mlm_loss=float(mlm_loss), msm_loss=float(msm_loss))

                cur_step = epoch_id * len(dataloader) + batch_id
                if _is_main_process() and cur_step % 100 == 0:
                    mlflow.log_metrics(
                        {
                            "train_mlm_loss": float(mlm_loss),
                            "train_msm_loss": float(msm_loss),
                        },
                        step=cur_step,
                    )
        return

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool) -> torch.Tensor:
        model: FMBaseWithHeads | DDP = self.model
        model.eval()
        device = self.device
        criterions = self.criterions
        trainer_args = self.trainer_args

        # num_batch
        # total_mlm, total_dm
        # total_top1_acc, total_top10_acc
        # total_recall@k, total_ndcg@k
        counter = torch.zeros(7, device=device)
        for batch in tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            set_attention_mask = batch["set_attention_mask"].to(device)
            t = batch["t"].to(device)

            if trainer_args["eval_last_set"]:
                masked_last_set_input_ids = masking_last_set(
                    input_ids.clone(), set_attention_mask, self.tokenizer
                )
            mlm_input_ids, mlm_labels = random_masking(
                input_ids=input_ids.clone(),
                tokenizer=self.tokenizer,
                mlm_probability=trainer_args["mlm_probability"],
            )
            msm_input_ids, msm_labels, set_select_mask = random_masking_set(
                input_ids=input_ids.clone(),
                set_attention_mask=set_attention_mask,
                tokenizer=self.tokenizer,
                mask_probability=trainer_args["msm_probability"],
            )

            target_dist = (
                observed_set_distribution(  # TODO: decide between masked-only or mixed
                    labels=msm_labels,
                    set_select_mask=set_select_mask,
                    # labels=input_ids,
                    # set_select_mask=set_attention_mask,
                    tokenizer=self.tokenizer,
                )
            )

            with autocast(device_type="cuda", dtype=torch.float16):
                mlm_logits, _, _ = model(
                    input_ids=mlm_input_ids,
                    attention_mask=attention_mask,
                    set_attention_mask=set_attention_mask,
                    t=t,
                    set_mask=set_select_mask,
                )
                msm_logits, msm_set_logits, _ = model(
                    input_ids=msm_input_ids,
                    attention_mask=attention_mask,
                    set_attention_mask=set_attention_mask,
                    t=t,
                    set_mask=set_select_mask,  # TODO: decide between masked-only or mixed
                    # set_mask=set_attention_mask,
                )
                mlm_loss = criterions["cross_entropy"](
                    mlm_logits.view(-1, mlm_logits.size(-1)),
                    mlm_labels.view(-1),
                )
                msm_loss = criterions["kl_div"](
                    F.log_softmax(msm_set_logits, dim=-1), target_dist
                )

                if trainer_args["eval_last_set"]:
                    masked_last_set_logits, _, _ = model(
                        input_ids=masked_last_set_input_ids,
                        attention_mask=attention_mask,
                        set_attention_mask=set_attention_mask,
                        t=t,
                    )
            top1_acc = topk_accuracy(mlm_logits, mlm_labels, 1)
            top10_acc = topk_accuracy(mlm_logits, mlm_labels, 10)

            if trainer_args["eval_last_set"]:
                set_select_mask = select_last_set(set_attention_mask)
                p_tokens, t_tokens = pred_and_target_sets(
                    masked_last_set_logits, input_ids, set_select_mask, 10
                )
                recall10 = recall_at_k(p_tokens, t_tokens)

                ndcg10 = ndcg_at_k(p_tokens, t_tokens)
            else:
                p_tokens, t_tokens = pred_and_target_sets(
                    msm_logits, input_ids, set_select_mask, 10
                )
                recall10 = recall_at_k(p_tokens, t_tokens)
                ndcg10 = ndcg_at_k(p_tokens, t_tokens)

            counter[0] += 1
            counter[1] += mlm_loss.item()
            counter[2] += msm_loss.item()
            counter[3] += top1_acc.item()
            counter[4] += top10_acc.item()
            counter[5] += recall10.item()
            counter[6] += ndcg10.item()

        if "LOCAL_RANK" in os.environ:
            dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        return (
            (counter[1] / counter[0]).item(),
            (counter[2] / counter[0]).item(),
            (counter[3] / counter[0]).item(),
            (counter[4] / counter[0]).item(),
            (counter[5] / counter[0]).item(),
            (counter[6] / counter[0]).item(),
        )

    def _valid(self, dataloader, verbose, epoch_id):
        val_mlm, val_msm, val_top1_acc, val_top10_acc, val_recall10, val_ndcg10 = (
            self.evaluate(dataloader, verbose)
        )
        if verbose:
            print(
                f"epoch {epoch_id}/val_mlm_loss: {round(val_mlm, 3)}/"
                f"val_msm_loss: {round(val_msm, 3)}/"
                f"val_top1_acc: {round(val_top1_acc, 3)}/"
                f"val_top10_acc: {round(val_top10_acc, 3)}/"
                f"val_recall10: {round(val_recall10, 3)}/"
                f"val_ndcg10: {round(val_ndcg10, 3)}/"
            )

        valid_metrics = {
            "callback_metric": val_ndcg10,
            "logged_metrics": {
                "val_mlm_loss": val_mlm,
                "val_msm_loss": val_msm,
                "val_top1_acc": val_top1_acc,
                "val_top10_acc": val_top10_acc,
                "val_recall10": val_recall10,
                "val_ndcg10": val_ndcg10,
            },
        }
        return valid_metrics


class BaseWithSoftCLTTrainer(Trainer):
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

        d_model = model.embeddings.embeddings.embedding_dim
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh()).to(
            self.device
        )
        self.optimizer.add_param_group(
            {"params": self.proj.parameters(), "lr": trainer_args["lr"]}
        )

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

                # mlm first
                mlm_input_ids, mlm_labels = random_masking(
                    input_ids=input_ids,
                    tokenizer=self.tokenizer,
                    mlm_probability=trainer_args["mlm_probability"],
                )
                if (mlm_labels == -100).all():
                    continue

                # duplicate
                mlm_input_ids = torch.concat([mlm_input_ids, mlm_input_ids], dim=0)
                mlm_labels = torch.concat([mlm_labels, mlm_labels], dim=0)
                attention_mask = torch.concat([attention_mask, attention_mask], dim=0)
                set_attention_mask = torch.concat(
                    [set_attention_mask, set_attention_mask], dim=0
                )
                t = torch.concat([t, t], dim=0)

                with autocast(device_type="cuda", dtype=torch.float16):
                    mlm_logits, (h, _) = model(
                        input_ids=mlm_input_ids,
                        attention_mask=attention_mask,
                        set_attention_mask=set_attention_mask,
                        t=t,
                    )
                    mlm_loss = criterions["cross_entropy"](
                        mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)
                    )

                    # h: (batch, max_seq, max_set_size, hidden_size)
                    # soft-dtw on h -> dist_mat: (batch, batch)
                    # soft inst not stable
                    # h = h.chunk(2, dim=0)[0][:, :, 0, :]
                    # mid_h: (2 * batch, max_seq, max_set_size, hidden_size)
                    h1, h2 = h[:, :, 0, :].chunk(2, dim=0)
                    h1 = self.proj(h1)
                    h2 = self.proj(h2)

                    mask = set_attention_mask.chunk(2, dim=0)[0]  # (batch, max_seq)

                    softclt_loss = criterions["softclt"](h1, h2, mask)
                    loss = (
                        trainer_args["l_mlm"] * mlm_loss
                        + trainer_args["l_softclt"] * softclt_loss
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bar.set_postfix(
                    mlm_loss=float(mlm_loss), softclt_loss=float(softclt_loss)
                )

                cur_step = epoch_id * len(dataloader) + batch_id
                if _is_main_process() and cur_step % 100 == 0:
                    mlflow.log_metrics(
                        {
                            "train_mlm_loss": float(mlm_loss),
                            "train_softclt_loss": float(softclt_loss),
                        },
                        step=cur_step,
                    )
        return

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool) -> torch.Tensor:
        model: FMBase | DDP = self.model
        model.eval()
        device = self.device
        criterions = self.criterions
        trainer_args = self.trainer_args

        # num_batch
        # total_mlm, total_dm
        # total_top1_acc, total_top10_acc
        # total_recall@k, total_ndcg@k
        counter = torch.zeros(7, device=device)
        for batch in tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            set_attention_mask = batch["set_attention_mask"].to(device)
            t = batch["t"].to(device)

            if trainer_args["eval_last_set"]:
                masked_last_set_input_ids = masking_last_set(
                    input_ids.clone(), set_attention_mask, self.tokenizer
                )
            mlm_input_ids, mlm_labels = random_masking(
                input_ids=input_ids.clone(),
                tokenizer=self.tokenizer,
                mlm_probability=trainer_args["mlm_probability"],
            )
            mlm_input_ids_dup = torch.concat([mlm_input_ids, mlm_input_ids], dim=0)
            mlm_labels_dup = torch.concat([mlm_labels, mlm_labels], dim=0)
            attention_mask_dup = torch.concat([attention_mask, attention_mask], dim=0)
            set_attention_mask_dup = torch.concat(
                [set_attention_mask, set_attention_mask], dim=0
            )
            t_dup = torch.concat([t, t], dim=0)

            with autocast(device_type="cuda", dtype=torch.float16):
                mlm_logits, (h, _) = model(
                    input_ids=mlm_input_ids_dup,
                    attention_mask=attention_mask_dup,
                    set_attention_mask=set_attention_mask_dup,
                    t=t_dup,
                )
                mlm_loss = criterions["cross_entropy"](
                    mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels_dup.view(-1)
                )
                # h: (batch, max_seq, max_set_size, hidden_size)
                # h = h.chunk(2, dim=0)[0][:, :, 0, :]
                # mid_h: (2 * batch, max_seq, max_set_size, hidden_size)
                h1, h2 = h[:, :, 0, :].chunk(2, dim=0)
                h1 = self.proj(h1)
                h2 = self.proj(h2)

                mask = set_attention_mask_dup.chunk(2, dim=0)[0]  # (batch, max_seq)

                softclt_loss = criterions["softclt"](h1, h2, mask)

                if trainer_args["eval_last_set"]:
                    masked_last_set_logits, (_, _) = model(
                        input_ids=masked_last_set_input_ids,
                        attention_mask=attention_mask,
                        set_attention_mask=set_attention_mask,
                        t=t,
                    )
            top1_acc = topk_accuracy(mlm_logits, mlm_labels_dup, 1)
            top10_acc = topk_accuracy(mlm_logits, mlm_labels_dup, 10)

            if trainer_args["eval_last_set"]:
                set_select_mask = select_last_set(set_attention_mask)
                p_tokens, t_tokens = pred_and_target_sets(
                    masked_last_set_logits, input_ids, set_select_mask, 10
                )
                recall10 = recall_at_k(p_tokens, t_tokens)
                ndcg10 = ndcg_at_k(p_tokens, t_tokens)

            counter[0] += 1
            counter[1] += mlm_loss.item()
            counter[2] += softclt_loss.item()
            counter[3] += top1_acc.item()
            counter[4] += top10_acc.item()
            counter[5] += recall10.item()
            counter[6] += ndcg10.item()

        if "LOCAL_RANK" in os.environ:
            dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        return (
            (counter[1] / counter[0]).item(),
            (counter[2] / counter[0]).item(),
            (counter[3] / counter[0]).item(),
            (counter[4] / counter[0]).item(),
            (counter[5] / counter[0]).item(),
            (counter[6] / counter[0]).item(),
        )

    def _valid(self, dataloader, verbose, epoch_id):
        (
            val_mlm,
            val_softclt_loss,
            val_top1_acc,
            val_top10_acc,
            val_recall10,
            val_ndcg10,
        ) = self.evaluate(dataloader, verbose)
        if verbose:
            print(
                f"epoch {epoch_id}/val_mlm_loss: {round(val_mlm, 3)}/"
                f"val_softclt_loss: {round(val_softclt_loss, 3)}/"
                f"val_top1_acc: {round(val_top1_acc, 3)}/"
                f"val_top10_acc: {round(val_top10_acc, 3)}/"
                f"val_recall10: {round(val_recall10, 3)}/"
                f"val_ndcg10: {round(val_ndcg10, 3)}/"
            )

        valid_metrics = {
            "callback_metric": val_ndcg10,
            "logged_metrics": {
                "val_mlm_loss": val_mlm,
                "val_softclt_loss": val_softclt_loss,
                "val_top1_acc": val_top1_acc,
                "val_top10_acc": val_top10_acc,
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
