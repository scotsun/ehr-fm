import torch
import torch.nn as nn
import torch.distributed as dist
import mlflow

from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from mlflow.models import ModelSignature

from sklearn.metrics import roc_auc_score, average_precision_score

from utils.dist_utils import is_main_process, _dist_is_initialized

from . import EarlyStopping, Trainer

# NOTE: Trainer is DDP-aware, but SetFit as a fewshot method, we do not need DDP


class SetFitSTFTTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        pooler: nn.Module,
        contrastive_criterion: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        model_signature: ModelSignature,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            early_stopping=early_stopping,
            verbose_period=verbose_period,
            device=device,
            model_signature=model_signature,
        )
        self.pooler = pooler.to(device)
        self.contrastive_criterion = contrastive_criterion

    def _autocast_kwargs(self) -> dict:
        amp_device_type = self.device.type
        enabled = amp_device_type in {"cuda", "cpu"}
        dtype = torch.float16 if amp_device_type == "cuda" else torch.bfloat16
        return {"device_type": amp_device_type, "dtype": dtype, "enabled": enabled}

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        # dataloader(setfit dataset)
        model: nn.Module = self.model
        model.train()
        self.pooler.train()
        scaler = self.scaler
        optimizer = self.optimizer
        contrastive_criterion = self.contrastive_criterion
        device = self.device

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                optimizer.zero_grad()

                inputs_a = {
                    k.split("_")[0]: v.to(device)
                    for k, v in batch.items()
                    if k.endswith("_a")
                }
                inputs_b = {
                    k.split("_")[0]: v.to(device)
                    for k, v in batch.items()
                    if k.endswith("_b")
                }
                pair_labels = batch["pair_label"].to(device)

                with autocast(**self._autocast_kwargs()):
                    h_a = model(**inputs_a)[-1]
                    z_a = self.pooler(h_a)
                    h_b = model(**inputs_b)[-1]
                    z_b = self.pooler(h_b)

                    con_loss = contrastive_criterion(z_a, z_b, pair_labels)

                scaler.scale(con_loss).backward()

                # --- optimizer step ---
                scaler.step(optimizer)
                scaler.update()

                bar.set_postfix(con_loss=float(con_loss))

                cur_step = epoch_id * len(dataloader) + batch_id
                if is_main_process() and cur_step % 100 == 0:
                    mlflow.log_metrics(
                        {"train_con_loss": float(con_loss)},
                        step=cur_step,
                    )
        return

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool):
        model: nn.Module = self.model
        model.eval()
        self.pooler.eval()
        device = self.device
        contrastive_criterion = self.contrastive_criterion

        counter = torch.zeros(2, device=device)
        for batch in tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose):
            inputs_a = {
                k.split("_")[0]: v.to(device)
                for k, v in batch.items()
                if k.endswith("_a")
            }
            inputs_b = {
                k.split("_")[0]: v.to(device)
                for k, v in batch.items()
                if k.endswith("_b")
            }
            pair_labels = batch["pair_label"].to(device)

            with autocast(**self._autocast_kwargs()):
                h_a = model(**inputs_a)[-1]
                z_a = self.pooler(h_a)
                h_b = model(**inputs_b)[-1]
                z_b = self.pooler(h_b)

                con_loss = contrastive_criterion(z_a, z_b, pair_labels)

            counter[0] += 1
            counter[1] += con_loss.item()

        if _dist_is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        return (counter[1] / counter[0]).item()

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        val_con_loss = self.evaluate(dataloader, verbose)
        if verbose:
            print(f"epoch {epoch_id}/val_con_loss: {round(val_con_loss, 3)}")

        valid_metrics = {
            "callback_metric": val_con_loss,
            "logged_metrics": {
                "val_con_loss": val_con_loss,
            },
        }
        return valid_metrics


class SetFitTaskHeadTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        task_head: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        model_signature: ModelSignature,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            early_stopping=early_stopping,
            verbose_period=verbose_period,
            device=device,
            model_signature=model_signature,
        )
        for p in model.parameters():  # freeze backbone
            p.requires_grad = False
        self.task_head = task_head.to(device)
        self.criterion = criterion

    def _autocast_kwargs(self) -> dict:
        amp_device_type = self.device.type
        enabled = amp_device_type in {"cuda", "cpu"}
        dtype = torch.float16 if amp_device_type == "cuda" else torch.bfloat16
        return {"device_type": amp_device_type, "dtype": dtype, "enabled": enabled}

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        # dataloader(fewshot dataset)
        model = self.model
        model.train()
        self.task_head.train()
        scaler = self.scaler
        optimizer = self.optimizer
        criterion = self.criterion
        device = self.device

        label_key = dataloader.dataset.label_key
        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                optimizer.zero_grad()

                inputs = {k: v.to(device) for k, v in batch.items() if k != label_key}
                y = batch[label_key].to(device)

                with autocast(**self._autocast_kwargs()):
                    h = model(**inputs)[-1]
                    yh = self.task_head(h)
                    loss = criterion(yh, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bar.set_postfix(loss=float(loss))
                cur_step = epoch_id * len(dataloader) + batch_id
                if is_main_process() and cur_step % 100 == 0:
                    mlflow.log_metrics(
                        {"train_loss": float(loss)},
                        step=cur_step,
                    )
        return

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool):
        # dataloader(fewshot dataset)
        model: nn.Module | DDP = self.model
        model.eval()
        self.task_head.eval()
        device = self.device

        label_key = dataloader.dataset.label_key
        ys, yhs = [], []
        for batch in tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose):
            inputs = {k: v.to(device) for k, v in batch.items() if k != label_key}
            y = batch[label_key]

            h = model(**inputs)[-1]
            yh = self.task_head(h)

            ys.append(y.cpu())
            yhs.append(yh.cpu())
        ys, yhs = torch.cat(ys), torch.cat(yhs)
        ys_np = ys.squeeze(-1).numpy()
        probs = torch.sigmoid(yhs).squeeze(-1).numpy()
        return (
            roc_auc_score(ys_np, probs),
            average_precision_score(ys_np, probs),
        )

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        valid_auc, valid_ap = self.evaluate(dataloader, verbose)
        if verbose:
            print(f"epoch {epoch_id}/valid_auc={round(valid_auc, 3)}")

        valid_metrics = {
            "callback_metric": valid_auc,
            "logged_metrics": {
                "val_auc": valid_auc,
                "val_ap": valid_ap,
            },
        }
        return valid_metrics
