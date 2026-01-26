"""Trainer classes & training infra functionalities integrated with MLflow."""

from abc import ABC, abstractmethod
import signal
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
from tokenizers import Tokenizer

from src.pretrain.data_utils import random_masking
from src.pretrain.masking import encounter_masking, observed_segment_distribution, masking_last_segment
from src.metric import topk_accuracy, recall_at_k, ndcg_at_k


class CheckpointManager:
    """Manages local checkpoint saving with support for Slurm graceful shutdown."""

    def __init__(
        self,
        checkpoint_dir: str,
        start_saving_after: int = 5,
        keep_last_n: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_saving_after = start_saving_after  # Start saving after N epochs
        self.keep_last_n = keep_last_n
        self._model = None
        self._optimizer = None
        self._scaler = None
        self._epoch = 0
        self._best_loss = float('inf')

        # Register signal handlers for Slurm
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for Slurm signals (SIGTERM, SIGUSR1)."""
        # SIGTERM: Slurm sends this before killing job
        signal.signal(signal.SIGTERM, self._signal_handler)
        # SIGUSR1: Can be used for preemption warning
        signal.signal(signal.SIGUSR1, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals by saving checkpoint."""
        sig_name = signal.Signals(signum).name
        print(f"\n⚠️  Received {sig_name} signal! Saving emergency checkpoint...")
        self.save_emergency_checkpoint()
        print(f"✅ Emergency checkpoint saved. Exiting gracefully.")
        exit(0)

    def register_model(self, model, optimizer, scaler=None):
        """Register model and optimizer for signal handler access."""
        self._model = model
        self._optimizer = optimizer
        self._scaler = scaler

    def update_epoch(self, epoch: int, best_loss: float = None):
        """Update current epoch (call at start of each epoch)."""
        self._epoch = epoch
        if best_loss is not None:
            self._best_loss = best_loss

    def should_save(self, epoch: int) -> bool:
        """Check if checkpoint should be saved this epoch (after start_saving_after epochs)."""
        return epoch >= self.start_saving_after

    def save_checkpoint(self, model, optimizer, epoch: int, loss: float, scaler=None, is_best: bool = False):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': model.config.to_dict() if hasattr(model, 'config') else None,
        }
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Save periodic checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"⭐ New best model saved: {best_path}")

        # Save latest (always overwrite)
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

    def save_emergency_checkpoint(self):
        """Save emergency checkpoint when receiving termination signal."""
        if self._model is None:
            print("⚠️  No model registered for emergency save")
            return

        checkpoint = {
            'epoch': self._epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict() if self._optimizer else None,
            'loss': self._best_loss,
            'emergency': True,
            'config': self._model.config.to_dict() if hasattr(self._model, 'config') else None,
        }
        if self._scaler is not None:
            checkpoint['scaler_state_dict'] = self._scaler.state_dict()

        emergency_path = self.checkpoint_dir / f"emergency_checkpoint_epoch_{self._epoch:04d}.pt"
        torch.save(checkpoint, emergency_path)

        # Also update latest
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints (excluding best/latest/emergency)."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                ckpt.unlink()
                print(f"🗑️  Removed old checkpoint: {ckpt.name}")

    def load_checkpoint(self, model, optimizer, scaler=None, path: str = None):
        """Load checkpoint. If path is None, loads latest."""
        if path is None:
            path = self.checkpoint_dir / "latest_checkpoint.pt"
        else:
            path = Path(path)

        if not path.exists():
            print(f"No checkpoint found at {path}")
            return None

        print(f"📂 Loading checkpoint: {path}")
        checkpoint = torch.load(path, weights_only=False)

        # Use strict=False for backward compatibility (e.g., old checkpoint without T2V)
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            print(f"⚠️  Missing keys (will be randomly initialized): {missing}")
        if unexpected:
            print(f"⚠️  Unexpected keys (ignored): {unexpected}")

        # Skip optimizer loading if model structure changed
        if missing or unexpected:
            print("⚠️  Skipping optimizer state (model structure changed)")
        elif checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"✅ Resumed from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
        return checkpoint


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        min_delta: int = 0,
        mode: str = "min",
        model_signature=None,
        local_rank: int = 0,
        use_mlflow: bool = True,
    ) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model_signature = model_signature
        self.local_rank = local_rank
        self.use_mlflow = use_mlflow

        match mode:
            case "min":
                self.monitor_op, self.delta_op = lambda a, b: a < b, -1 * min_delta
            case "max":
                self.monitor_op, self.delta_op = lambda a, b: a > b, min_delta
            case _:
                raise ValueError("mode must be either `min` or `max`")

    def _log_best_model(self, model):
        """helper function to log model."""
        if self.use_mlflow and self.local_rank == 0:
            mlflow.pytorch.log_model(
                model,
                "best_model",
                pip_requirements=None,  # Let MLflow auto-detect dependencies
                signature=self.model_signature,
            )

    def step(self, metric_val, model):
        # save the first chkpt
        if self.best_score is None:
            self.best_score = metric_val
            self._log_best_model(model)
            return
        # save the subsequent chkpt
        if self.monitor_op(metric_val, self.best_score + self.delta_op):
            self.best_score = metric_val
            self.counter = 0
            self._log_best_model(model)
            return
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
            return


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        local_rank: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.verbose_period = verbose_period
        self.device = device
        self.local_rank = local_rank
        self.best_score = -float("inf")

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: None | DataLoader = None,
    ):
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)
            if valid_loader is not None and self.local_rank == 0:
                self._valid(valid_loader, verbose, epoch)
                if self.early_stopping and self.early_stopping.early_stop:
                    break

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
        model: nn.Module,
        tokenizer: Tokenizer,
        optimizer: Optimizer,
        criterion: nn.CrossEntropyLoss,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        local_rank: int,
        use_encounter_masking: bool = False,
        encounter_mask_prob: float = 0.2,
        token_mask_prob: float = 0.15,
        use_mlflow: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        super().__init__(
            model, optimizer, early_stopping, verbose_period, device, local_rank
        )
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.use_encounter_masking = use_encounter_masking
        self.encounter_mask_prob = encounter_mask_prob
        self.token_mask_prob = token_mask_prob
        self.use_mlflow = use_mlflow
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        # Initialize AMP scaler if using mixed precision
        if self.use_amp:
            # Ultra-conservative settings to prevent overflow
            # init_scale: 2048 (32x smaller than default 65536)
            # growth_factor: 1.2 (very slow growth, vs default 2.0)
            # growth_interval: 10000 (very infrequent growth checks)
            self.scaler = torch.amp.GradScaler(
                'cuda',
                init_scale=2048.0,
                growth_factor=1.2,
                backoff_factor=0.5,
                growth_interval=10000
            )
        else:
            self.scaler = None

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        model: nn.Module = self.model
        model.train()
        optimizer = self.optimizer
        device = self.device

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                segment_attention_mask = batch["segment_attention_mask"].to(device)
                segment_time = batch.get("segment_time", None)
                if segment_time is not None:
                    segment_time = segment_time.to(device)
                token_time = batch.get("token_time", None)
                if token_time is not None:
                    token_time = token_time.to(device)

                # Use encounter masking or token masking
                if self.use_encounter_masking:
                    from src.pretrain.masking import encounter_masking
                    masked_input_ids, labels, enc_mask = encounter_masking(
                        input_ids, segment_attention_mask, self.tokenizer, self.encounter_mask_prob
                    )
                    # Skip empty batches
                    if (labels != -100).sum() == 0:
                        continue
                else:
                    masked_input_ids, labels = random_masking(input_ids, self.tokenizer, self.token_mask_prob)

                # Forward pass with AMP
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output = model(
                            input_ids=masked_input_ids,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                        )
                        # Handle both FMBase (returns logits, h) and old models (returns logits)
                        logits = output[0] if isinstance(output, tuple) else output
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / self.gradient_accumulation_steps
                else:
                    output = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        segment_attention_mask=segment_attention_mask,
                        segment_time=segment_time,
                        token_time=token_time,
                    )
                    # Handle both FMBase (returns logits, h) and old models (returns logits)
                    logits = output[0] if isinstance(output, tuple) else output
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass with AMP
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update weights with gradient accumulation
                if (batch_id + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        # Unscale gradients before clipping
                        if self.max_grad_norm > 0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # Standard gradient clipping and optimizer step
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad()

                bar.set_postfix(mlm_loss=float(loss * self.gradient_accumulation_steps))

                if self.use_mlflow:
                    cur_step = epoch_id * len(dataloader) + batch_id
                    if self.local_rank == 0 and cur_step % 100 == 0:
                        mlflow.log_metrics({"train_mlm_loss": float(loss * self.gradient_accumulation_steps)}, step=cur_step)
        return

    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int) -> dict:
        """
        Evaluate model with multiple metrics.

        Returns:
            dict with keys: mlm_loss, top1_acc, top10_acc, recall_10, ndcg_10
        """
        model: nn.Module = self.model
        model.eval()
        device = self.device

        total_mlm = 0.0
        total_top1 = 0.0
        total_top10 = 0.0
        total_recall10 = 0.0
        total_ndcg10 = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                dataloader, unit="batch", mininterval=0, disable=not verbose
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                segment_attention_mask = batch["segment_attention_mask"].to(device)
                segment_time = batch.get("segment_time", None)
                if segment_time is not None:
                    segment_time = segment_time.to(device)
                token_time = batch.get("token_time", None)
                if token_time is not None:
                    token_time = token_time.to(device)

                # Use encounter masking or token masking
                if self.use_encounter_masking:
                    masked_input_ids, labels, _ = encounter_masking(
                        input_ids, segment_attention_mask, self.tokenizer, self.encounter_mask_prob
                    )
                    # Skip empty batches
                    if (labels != -100).sum() == 0:
                        continue
                else:
                    masked_input_ids, labels = random_masking(input_ids, self.tokenizer, self.token_mask_prob)

                # Forward pass for MLM evaluation
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output = model(
                            input_ids=masked_input_ids,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                        )
                        logits = output[0] if isinstance(output, tuple) else output
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    output = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        segment_attention_mask=segment_attention_mask,
                        segment_time=segment_time,
                        token_time=token_time,
                    )
                    logits = output[0] if isinstance(output, tuple) else output
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                # MLM metrics
                total_mlm += loss.item()
                total_top1 += topk_accuracy(logits, labels, k=1).item()
                total_top10 += topk_accuracy(logits, labels, k=10).item()

                # Last segment prediction (for recall/ndcg)
                # Mask last segment and predict
                masked_last_seg_input = masking_last_segment(
                    input_ids, segment_attention_mask, self.tokenizer
                )
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        last_seg_output = model(
                            input_ids=masked_last_seg_input,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                        )
                        last_seg_logits = last_seg_output[0] if isinstance(last_seg_output, tuple) else last_seg_output
                else:
                    last_seg_output = model(
                        input_ids=masked_last_seg_input,
                        attention_mask=attention_mask,
                        segment_attention_mask=segment_attention_mask,
                        segment_time=segment_time,
                        token_time=token_time,
                    )
                    last_seg_logits = last_seg_output[0] if isinstance(last_seg_output, tuple) else last_seg_output

                total_recall10 += recall_at_k(last_seg_logits, input_ids, segment_attention_mask, k=10).item()
                total_ndcg10 += ndcg_at_k(last_seg_logits, input_ids, segment_attention_mask, k=10).item()

                num_batches += 1

        if num_batches == 0:
            return {"mlm_loss": 0.0, "top1_acc": 0.0, "top10_acc": 0.0, "recall_10": 0.0, "ndcg_10": 0.0}

        return {
            "mlm_loss": total_mlm / num_batches,
            "top1_acc": total_top1 / num_batches,
            "top10_acc": total_top10 / num_batches,
            "recall_10": total_recall10 / num_batches,
            "ndcg_10": total_ndcg10 / num_batches,
        }

    def _valid(self, dataloader, verbose, epoch_id):
        """
        log all the metrics in mlflow;
        return the metric for save-best/early-stop.
        """
        metrics = self.evaluate(dataloader, verbose, epoch_id)

        if verbose:
            print(f"  val_mlm: {metrics['mlm_loss']:.4f} | "
                  f"top1: {metrics['top1_acc']:.4f} | top10: {metrics['top10_acc']:.4f} | "
                  f"recall@10: {metrics['recall_10']:.4f} | ndcg@10: {metrics['ndcg_10']:.4f}")

        if self.use_mlflow:
            mlflow.log_metrics({
                "val_mlm_loss": metrics["mlm_loss"],
                "val_top1_acc": metrics["top1_acc"],
                "val_top10_acc": metrics["top10_acc"],
                "val_recall_10": metrics["recall_10"],
                "val_ndcg_10": metrics["ndcg_10"],
            }, step=epoch_id)

        if self.early_stopping:
            self.early_stopping.step(metrics["mlm_loss"], self.model)

        # Return MLM loss for backward compatibility
        return metrics["mlm_loss"]


class BaseWithHeadsTrainer(Trainer):
    """
    Trainer for FMBaseWithHeads model with dual-line masking and dual loss:
    - Line 1 (MLM): Token-level random masking → MLM loss (CrossEntropy)
    - Line 2 (DM):  Segment-level encounter masking → DM loss (KL Divergence)

    Two separate masking operations and two forward passes per batch.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        local_rank: int,
        token_mask_prob: float = 0.20,      # Token-level masking probability
        encounter_mask_prob: float = 0.40,  # Segment-level masking probability
        use_mlflow: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        mlm_weight: float = 1.0,
        dm_weight: float = 1.0,
        stage1_epochs: int = None,  # If set, enables staged training (MLM → DM)
    ) -> None:
        super().__init__(
            model, optimizer, early_stopping, verbose_period, device, local_rank
        )
        self.tokenizer = tokenizer
        self.token_mask_prob = token_mask_prob
        self.encounter_mask_prob = encounter_mask_prob
        self.use_mlflow = use_mlflow
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.mlm_weight = mlm_weight
        self.dm_weight = dm_weight
        self.stage1_epochs = stage1_epochs  # For staged training

        # Loss functions
        self.criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_dm = nn.KLDivLoss(reduction='batchmean')

        # Initialize AMP scaler if using mixed precision
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(
                'cuda',
                init_scale=2048.0,
                growth_factor=1.2,
                backoff_factor=0.5,
                growth_interval=10000
            )
        else:
            self.scaler = None

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        """
        Dual-line training with SEQUENTIAL BACKWARD for memory efficiency:
        - Line 1: Token-level random masking (token_mask_prob) → MLM loss → backward
        - Line 2: Segment-level encounter masking (encounter_mask_prob) → DM loss → backward

        Sequential backward: each forward pass is immediately followed by backward,
        releasing the computation graph before the next forward pass. This reduces
        peak memory from 2x to 1x compared to storing both graphs simultaneously.

        Memory optimization:
        - If mlm_weight=0 (encounter mode): skip MLM forward pass entirely
        - If dm_weight=0: skip DM forward pass entirely

        Staged training (when stage1_epochs is set):
        - Stage 1 (epoch < stage1_epochs): MLM only (mlm_weight=1.0, dm_weight=0.0)
        - Stage 2 (epoch >= stage1_epochs): DM only (mlm_weight=0.0, dm_weight=1.0)
        """
        model: nn.Module = self.model
        model.train()
        optimizer = self.optimizer
        device = self.device

        # Determine effective weights for this epoch (staged training support)
        if self.stage1_epochs is not None:
            if epoch_id < self.stage1_epochs:
                mlm_weight, dm_weight = 1.0, 0.0  # Stage 1: MLM only
                stage_name = "Stage1-MLM"
            else:
                mlm_weight, dm_weight = 0.0, 1.0  # Stage 2: DM only
                stage_name = "Stage2-DM"
        else:
            mlm_weight, dm_weight = self.mlm_weight, self.dm_weight
            stage_name = None

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            if stage_name:
                bar.set_description(f"Epoch {epoch_id} [{stage_name}]")
            else:
                bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                segment_attention_mask = batch["segment_attention_mask"].to(device)
                segment_time = batch.get("segment_time", None)
                if segment_time is not None:
                    segment_time = segment_time.to(device)
                token_time = batch.get("token_time", None)
                if token_time is not None:
                    token_time = token_time.to(device)

                # Initialize loss values for logging
                mlm_loss_val = 0.0
                dm_loss_val = 0.0

                # ============================================================
                # Line 1: MLM (Token-level masking) - Skip if mlm_weight=0
                # ============================================================
                if mlm_weight > 0:
                    mlm_input_ids, mlm_labels = random_masking(
                        input_ids.clone(), self.tokenizer, self.token_mask_prob
                    )
                    # Skip if no tokens masked
                    if (mlm_labels != -100).sum() == 0:
                        continue

                    # Need encounter_mask for model compatibility (even if not used for DM)
                    _, _, encounter_mask = encounter_masking(
                        input_ids.clone(), segment_attention_mask, self.tokenizer, self.encounter_mask_prob
                    )

                    # MLM Forward + Backward (graph released after backward)
                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            mlm_logits, _, _ = model(
                                input_ids=mlm_input_ids,
                                attention_mask=attention_mask,
                                segment_attention_mask=segment_attention_mask,
                                segment_time=segment_time,
                                token_time=token_time,
                                segment_mask=encounter_mask,
                            )
                            mlm_loss = self.criterion_mlm(
                                mlm_logits.view(-1, mlm_logits.size(-1)),
                                mlm_labels.view(-1)
                            )
                            mlm_loss_scaled = (mlm_weight * mlm_loss) / self.gradient_accumulation_steps
                        self.scaler.scale(mlm_loss_scaled).backward()
                    else:
                        mlm_logits, _, _ = model(
                            input_ids=mlm_input_ids,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                            segment_mask=encounter_mask,
                        )
                        mlm_loss = self.criterion_mlm(
                            mlm_logits.view(-1, mlm_logits.size(-1)),
                            mlm_labels.view(-1)
                        )
                        mlm_loss_scaled = (mlm_weight * mlm_loss) / self.gradient_accumulation_steps
                        mlm_loss_scaled.backward()

                    mlm_loss_val = mlm_loss.detach().item()

                # ============================================================
                # Line 2: DM (Segment-level masking) - Skip if dm_weight=0
                # ============================================================
                if dm_weight > 0:
                    dm_input_ids, dm_labels, encounter_mask = encounter_masking(
                        input_ids.clone(), segment_attention_mask, self.tokenizer, self.encounter_mask_prob
                    )
                    # Skip if no segments masked
                    if encounter_mask.sum() == 0:
                        continue

                    # Compute target distribution for DM loss
                    # Use input_ids (not dm_labels) to include PAD tokens in distribution
                    target_dist = observed_segment_distribution(
                        input_ids, encounter_mask, self.tokenizer
                    )

                    # DM Forward + Backward (graph released after backward)
                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            _, dm_logits, _ = model(
                                input_ids=dm_input_ids,
                                attention_mask=attention_mask,
                                segment_attention_mask=segment_attention_mask,
                                segment_time=segment_time,
                                token_time=token_time,
                                segment_mask=encounter_mask,
                            )
                            dm_loss = self.criterion_dm(
                                F.log_softmax(dm_logits, dim=-1),
                                target_dist
                            )
                            dm_loss_scaled = (dm_weight * dm_loss) / self.gradient_accumulation_steps
                        self.scaler.scale(dm_loss_scaled).backward()
                    else:
                        _, dm_logits, _ = model(
                            input_ids=dm_input_ids,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                            segment_mask=encounter_mask,
                        )
                        dm_loss = self.criterion_dm(
                            F.log_softmax(dm_logits, dim=-1),
                            target_dist
                        )
                        dm_loss_scaled = (dm_weight * dm_loss) / self.gradient_accumulation_steps
                        dm_loss_scaled.backward()

                    dm_loss_val = dm_loss.detach().item()

                # Update weights with gradient accumulation
                if (batch_id + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        if self.max_grad_norm > 0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad()

                # Logging
                total_loss_val = mlm_weight * mlm_loss_val + dm_weight * dm_loss_val
                bar.set_postfix(
                    mlm=mlm_loss_val,
                    dm=dm_loss_val,
                    total=total_loss_val
                )

                if self.use_mlflow:
                    cur_step = epoch_id * len(dataloader) + batch_id
                    if self.local_rank == 0 and cur_step % 100 == 0:
                        mlflow.log_metrics({
                            "train_mlm_loss": mlm_loss_val,
                            "train_dm_loss": dm_loss_val,
                            "train_total_loss": total_loss_val,
                        }, step=cur_step)
        return

    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int) -> dict:
        """
        Evaluate model with dual-line masking (same as training).

        Returns:
            dict with keys: mlm_loss, dm_loss, top1_acc, top10_acc, recall_10, ndcg_10
        """
        model: nn.Module = self.model
        model.eval()
        device = self.device

        total_mlm = 0.0
        total_dm = 0.0
        total_top1 = 0.0
        total_top10 = 0.0
        total_recall10 = 0.0
        total_ndcg10 = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                dataloader, unit="batch", mininterval=0, disable=not verbose
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                segment_attention_mask = batch["segment_attention_mask"].to(device)
                segment_time = batch.get("segment_time", None)
                if segment_time is not None:
                    segment_time = segment_time.to(device)
                token_time = batch.get("token_time", None)
                if token_time is not None:
                    token_time = token_time.to(device)

                # ============================================================
                # Line 1: Token-level random masking for MLM
                # ============================================================
                mlm_input_ids, mlm_labels = random_masking(
                    input_ids.clone(), self.tokenizer, self.token_mask_prob
                )
                if (mlm_labels != -100).sum() == 0:
                    continue

                # ============================================================
                # Line 2: Segment-level encounter masking for DM
                # ============================================================
                dm_input_ids, dm_labels, encounter_mask = encounter_masking(
                    input_ids.clone(), segment_attention_mask, self.tokenizer, self.encounter_mask_prob
                )
                if encounter_mask.sum() == 0:
                    continue

                # Compute target distribution
                # Use input_ids (not dm_labels) to include PAD tokens in distribution
                target_dist = observed_segment_distribution(
                    input_ids, encounter_mask, self.tokenizer
                )

                # Forward passes
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        # Forward pass 1: MLM
                        mlm_logits, _, _ = model(
                            input_ids=mlm_input_ids,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                            segment_mask=encounter_mask,
                        )

                        # Forward pass 2: DM
                        _, dm_logits, _ = model(
                            input_ids=dm_input_ids,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                            segment_mask=encounter_mask,
                        )

                        mlm_loss = self.criterion_mlm(
                            mlm_logits.view(-1, mlm_logits.size(-1)),
                            mlm_labels.view(-1)
                        )
                        dm_loss = self.criterion_dm(
                            F.log_softmax(dm_logits, dim=-1),
                            target_dist
                        )
                else:
                    # Forward pass 1: MLM
                    mlm_logits, _, _ = model(
                        input_ids=mlm_input_ids,
                        attention_mask=attention_mask,
                        segment_attention_mask=segment_attention_mask,
                        segment_time=segment_time,
                        token_time=token_time,
                        segment_mask=encounter_mask,
                    )

                    # Forward pass 2: DM
                    _, dm_logits, _ = model(
                        input_ids=dm_input_ids,
                        attention_mask=attention_mask,
                        segment_attention_mask=segment_attention_mask,
                        segment_time=segment_time,
                        token_time=token_time,
                        segment_mask=encounter_mask,
                    )

                    mlm_loss = self.criterion_mlm(
                        mlm_logits.view(-1, mlm_logits.size(-1)),
                        mlm_labels.view(-1)
                    )
                    dm_loss = self.criterion_dm(
                        F.log_softmax(dm_logits, dim=-1),
                        target_dist
                    )

                # Loss metrics
                total_mlm += mlm_loss.item()
                total_dm += dm_loss.item()

                # MLM accuracy metrics (using token-level masked logits/labels)
                total_top1 += topk_accuracy(mlm_logits, mlm_labels, k=1).item()
                total_top10 += topk_accuracy(mlm_logits, mlm_labels, k=10).item()

                # Last segment prediction (for recall/ndcg)
                masked_last_seg_input = masking_last_segment(
                    input_ids, segment_attention_mask, self.tokenizer
                )
                seg_mask_bool = segment_attention_mask.bool()
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        last_seg_logits, _, _ = model(
                            input_ids=masked_last_seg_input,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                            segment_mask=seg_mask_bool,
                        )
                else:
                    last_seg_logits, _, _ = model(
                        input_ids=masked_last_seg_input,
                        attention_mask=attention_mask,
                        segment_attention_mask=segment_attention_mask,
                        segment_time=segment_time,
                        token_time=token_time,
                        segment_mask=seg_mask_bool,
                    )

                total_recall10 += recall_at_k(last_seg_logits, input_ids, segment_attention_mask, k=10).item()
                total_ndcg10 += ndcg_at_k(last_seg_logits, input_ids, segment_attention_mask, k=10).item()

                num_batches += 1

        if num_batches == 0:
            return {
                "mlm_loss": 0.0, "dm_loss": 0.0,
                "top1_acc": 0.0, "top10_acc": 0.0,
                "recall_10": 0.0, "ndcg_10": 0.0
            }

        return {
            "mlm_loss": total_mlm / num_batches,
            "dm_loss": total_dm / num_batches,
            "top1_acc": total_top1 / num_batches,
            "top10_acc": total_top10 / num_batches,
            "recall_10": total_recall10 / num_batches,
            "ndcg_10": total_ndcg10 / num_batches,
        }

    def _valid(self, dataloader, verbose, epoch_id):
        """
        Log all the metrics in mlflow;
        Return the metric for save-best/early-stop.
        """
        # Determine effective weights for this epoch (staged training support)
        if self.stage1_epochs is not None:
            if epoch_id < self.stage1_epochs:
                mlm_weight, dm_weight = 1.0, 0.0  # Stage 1: MLM only
            else:
                mlm_weight, dm_weight = 0.0, 1.0  # Stage 2: DM only
        else:
            mlm_weight, dm_weight = self.mlm_weight, self.dm_weight

        metrics = self.evaluate(dataloader, verbose, epoch_id)
        val_total = mlm_weight * metrics["mlm_loss"] + dm_weight * metrics["dm_loss"]

        if verbose:
            print(f"  val_mlm: {metrics['mlm_loss']:.4f} | val_dm: {metrics['dm_loss']:.4f} | "
                  f"top1: {metrics['top1_acc']:.4f} | top10: {metrics['top10_acc']:.4f} | "
                  f"recall@10: {metrics['recall_10']:.4f} | ndcg@10: {metrics['ndcg_10']:.4f}")

        if self.use_mlflow:
            mlflow.log_metrics({
                "val_mlm_loss": metrics["mlm_loss"],
                "val_dm_loss": metrics["dm_loss"],
                "val_total_loss": val_total,
                "val_top1_acc": metrics["top1_acc"],
                "val_top10_acc": metrics["top10_acc"],
                "val_recall_10": metrics["recall_10"],
                "val_ndcg_10": metrics["ndcg_10"],
            }, step=epoch_id)

        if self.early_stopping:
            self.early_stopping.step(val_total, self.model)

        return val_total


def test_logging(
    trainer: Trainer,
    test_loader: DataLoader,
    expr_name: str,
    run_name: str,
):
    """
    Log test metrics to an existing MLflow run.

    Args:
        trainer: Trainer instance with model and evaluate method
        test_loader: DataLoader for test set
        expr_name: MLflow experiment name
        run_name: MLflow run name to add metrics to
    """
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
                # Add "test_" prefix to distinguish from validation metrics
                test_metrics = {f"test_{k}": v for k, v in test_scores.items()}
                mlflow.log_metrics(metrics=test_metrics, step=0)
            print(f"Successfully added metrics to run {run_id}")
        except Exception as e:
            print(f"Error updating run: {e}")
