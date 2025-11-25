"""Trainer classes & training infra functionalities integrated with MLflow."""

from abc import ABC, abstractmethod
import signal
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
from tokenizers import Tokenizer

from src.utils.data_utils import random_masking


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

        model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] is not None:
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
        encounter_mask_prob: float = 0.3,
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
                    from src.utils.encounter_masking import encounter_masking
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
                        logits = model(
                            input_ids=masked_input_ids,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                        )
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / self.gradient_accumulation_steps
                else:
                    logits = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        segment_attention_mask=segment_attention_mask,
                        segment_time=segment_time,
                        token_time=token_time,
                    )
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

    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int) -> float:
        model: nn.Module = self.model
        model.eval()
        device = self.device

        total_mlm = 0.0
        total_num_batch = len(dataloader)
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
                    from src.utils.encounter_masking import encounter_masking
                    masked_input_ids, labels, _ = encounter_masking(
                        input_ids, segment_attention_mask, self.tokenizer, self.encounter_mask_prob
                    )
                    # Skip empty batches
                    if (labels != -100).sum() == 0:
                        continue
                else:
                    masked_input_ids, labels = random_masking(input_ids, self.tokenizer, self.token_mask_prob)

                # Forward pass with AMP (validation)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(
                            input_ids=masked_input_ids,
                            attention_mask=attention_mask,
                            segment_attention_mask=segment_attention_mask,
                            segment_time=segment_time,
                            token_time=token_time,
                        )
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    logits = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        segment_attention_mask=segment_attention_mask,
                        segment_time=segment_time,
                        token_time=token_time,
                    )
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                total_mlm += loss.item()

        return total_mlm / total_num_batch

    def _valid(self, dataloader, verbose, epoch_id):
        """
        log all the metrics in mlflow;
        return the metric for save-best/early-stop.
        """
        if verbose:
            valid_mlm = self.evaluate(dataloader, verbose, epoch_id)
            if self.use_mlflow:
                mlflow.log_metrics({"val_mlm_loss": valid_mlm}, step=epoch_id)
        if self.early_stopping:
            self.early_stopping.step(valid_mlm, self.model)
        return valid_mlm


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
