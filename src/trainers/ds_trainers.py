import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.downstream_modules import Downstream
from src.setfit.loss import CosineSimilarityLoss


@dataclass
class SetFitConfig:
    stft_epochs: int = 1
    taskhead_epochs: int = 5
    stft_batch_size: int = 32
    task_batch_size: int = 32
    stft_lr: float = 1e-4
    task_lr: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 256
    d_hidden: int = 512
    embedding_dim: int = 128
    num_workers: int = 0
    cls_token_id: int = 2
    set_pool: str = "cls"
    pooler_model_type: str = "mlp"
    taskhead_model_type: str = "mlp"


class SetFitTwoStageRunner:
    """
    Two-stage SetFit runner.

    Stage 1: contrastive ST-FT on SetFit pair dataset.
    Stage 2: task-head fine-tuning on FewShot dataset.

    Both pooler and task-head are Downstream modules.
    """

    def __init__(self, backbone: nn.Module, device: torch.device, cfg: SetFitConfig):
        self.backbone = backbone.to(device)
        self.device = device
        self.cfg = cfg

        self.pooler = Downstream(
            d_model=cfg.d_model,
            d_hidden=cfg.d_hidden,
            d_out=cfg.embedding_dim,
            model_type=cfg.pooler_model_type,
            set_pool=cfg.set_pool,
        ).to(device)

        self.task_head = Downstream(
            d_model=cfg.d_model,
            d_hidden=cfg.d_hidden,
            d_out=1,
            model_type=cfg.taskhead_model_type,
            set_pool=cfg.set_pool,
        ).to(device)

        self.contrastive_criterion = CosineSimilarityLoss()
        self.task_criterion = nn.BCEWithLogitsLoss()

        self.stft_optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.pooler.parameters()),
            lr=cfg.stft_lr,
            weight_decay=cfg.weight_decay,
        )
        self.task_optimizer = torch.optim.AdamW(
            self.task_head.parameters(),
            lr=cfg.task_lr,
            weight_decay=cfg.weight_decay,
        )

    def _autocast_kwargs(self) -> dict:
        amp_device_type = self.device.type
        enabled = amp_device_type in {"cuda", "cpu"}
        dtype = torch.float16 if amp_device_type == "cuda" else torch.bfloat16
        return {"device_type": amp_device_type, "dtype": dtype, "enabled": enabled}

    def _split_pair_inputs(self, batch: dict, suffix: str) -> dict:
        return {
            k[: -len(suffix)]: v.to(self.device)
            for k, v in batch.items()
            if k.endswith(suffix)
        }

    def _resolve_masks(
        self,
        hidden: torch.Tensor,
        inputs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden.ndim == 4:
            token_mask = inputs["attention_mask"].bool()
            set_mask = inputs.get("set_attention_mask")
            if set_mask is not None:
                set_mask = set_mask.bool()
            else:
                set_mask = token_mask.any(dim=-1)
            return token_mask, set_mask

        if hidden.ndim == 3:
            attn_mask = inputs.get("attention_mask")
            if self.cfg.set_pool == "cls-1d":
                if "input_ids" not in inputs:
                    raise KeyError("input_ids is required when set_pool='cls-1d'.")
                token_mask = inputs["input_ids"].eq(self.cfg.cls_token_id)
                if attn_mask is not None:
                    token_mask = token_mask & attn_mask.bool()
                return token_mask, None

            # For 1D backbones, this class currently requires cls-1d pooling.
            raise ValueError(
                "For hidden shape (B, L, D), set_pool must be 'cls-1d' for Downstream."
            )

        raise ValueError(f"Unexpected hidden shape: {tuple(hidden.shape)}")

    def _encode(self, inputs: dict) -> torch.Tensor:
        return self.backbone(**inputs)[-1]

    def _downstream(self, downstream: Downstream, hidden: torch.Tensor, inputs: dict):
        token_mask, set_mask = self._resolve_masks(hidden, inputs)
        return downstream(hidden, token_mask, set_mask)

    def _stft_epoch(self, dataloader: DataLoader, scaler: GradScaler, epoch: int):
        self.backbone.train()
        self.pooler.train()

        running_loss = 0.0
        for step, batch in enumerate(dataloader):
            self.stft_optimizer.zero_grad()

            inputs_a = self._split_pair_inputs(batch, "_a")
            inputs_b = self._split_pair_inputs(batch, "_b")
            labels = batch["pair_label"].to(self.device).float()

            with autocast(**self._autocast_kwargs()):
                h_a = self._encode(inputs_a)
                h_b = self._encode(inputs_b)
                z_a = self._downstream(self.pooler, h_a, inputs_a)
                z_b = self._downstream(self.pooler, h_b, inputs_b)
                loss = self.contrastive_criterion(z_a, z_b, labels)

            scaler.scale(loss).backward()
            scaler.step(self.stft_optimizer)
            scaler.update()
            running_loss += loss.item()

            if (step + 1) % 20 == 0:
                avg = running_loss / (step + 1)
                print(f"[ST-FT] epoch={epoch} step={step + 1} loss={avg:.4f}")

        return running_loss / max(len(dataloader), 1)

    def _task_epoch(self, dataloader: DataLoader, scaler: GradScaler, epoch: int):
        self.backbone.eval()
        self.task_head.train()

        running_loss = 0.0
        label_key = dataloader.dataset.label_key

        for step, batch in enumerate(dataloader):
            self.task_optimizer.zero_grad()

            inputs = {k: v.to(self.device) for k, v in batch.items() if k != label_key}
            y = batch[label_key].to(self.device).float().view(-1, 1)

            with torch.no_grad():
                hidden = self._encode(inputs)

            with autocast(**self._autocast_kwargs()):
                logits = self._downstream(self.task_head, hidden, inputs)
                loss = self.task_criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(self.task_optimizer)
            scaler.update()
            running_loss += loss.item()

            if (step + 1) % 20 == 0:
                avg = running_loss / (step + 1)
                print(f"[TaskHead] epoch={epoch} step={step + 1} loss={avg:.4f}")

        return running_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def evaluate_task_head(self, dataloader: DataLoader) -> tuple[float, float]:
        self.backbone.eval()
        self.task_head.eval()

        label_key = dataloader.dataset.label_key
        all_y = []
        all_prob = []

        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != label_key}
            y = batch[label_key].to(self.device).float().view(-1)

            hidden = self._encode(inputs)
            logits = self._downstream(self.task_head, hidden, inputs).view(-1)
            probs = torch.sigmoid(logits)

            all_y.append(y)
            all_prob.append(probs)

        y = torch.cat(all_y)
        prob = torch.cat(all_prob)

        auroc = roc_auc_score(y.cpu().numpy(), prob.cpu().numpy())
        ap = average_precision_score(y.cpu().numpy(), prob.cpu().numpy())

        return auroc, ap

    def fit(self, setfit_dataset: Dataset, fewshot_dataset: Dataset):
        stft_loader = DataLoader(
            setfit_dataset,
            batch_size=self.cfg.stft_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )
        task_loader = DataLoader(
            fewshot_dataset,
            batch_size=self.cfg.task_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

        stft_scaler = GradScaler(device=self.device.type)
        task_scaler = GradScaler(device=self.device.type)

        print("=== Stage 1: SetFit ST-FT ===")
        for epoch in range(self.cfg.stft_epochs):
            loss = self._stft_epoch(stft_loader, stft_scaler, epoch)
            print(f"[ST-FT] epoch={epoch} avg_loss={loss:.4f}")

        print("=== Stage 2: Task-Head Fine-Tuning ===")
        for epoch in range(self.cfg.taskhead_epochs):
            loss = self._task_epoch(task_loader, task_scaler, epoch)
            auroc, ap = self.evaluate_task_head(task_loader)
            print(
                f"[TaskHead] epoch={epoch} avg_loss={loss:.4f} "
                f"train_auroc={auroc:.4f} train_ap={ap:.4f}"
            )


# Example usage:
# runner = SetFitTwoStageRunner(backbone=model, device=device, cfg=TwoStageConfig(...))
# runner.fit(setfit_dataset=setfit_dataset, fewshot_dataset=fewshot_dataset)
