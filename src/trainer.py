"""Trainer classes & training infra functionalities integrated with MLflow."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
from tokenizers import Tokenizer

from src.utils.data_utils import random_masking


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
        self.use_mlflow = use_mlflow
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        # Initialize AMP scaler if using mixed precision
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
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
                    masked_input_ids, labels = random_masking(input_ids, self.tokenizer)

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
                    masked_input_ids, labels = random_masking(input_ids, self.tokenizer)

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
