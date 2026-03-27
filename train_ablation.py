#!/usr/bin/env python3
"""
Pre-training script for NEST ablation variants (Group B).

B1 (swe_only):    SWE-only encoder, no CSE. MLM-only pretraining.
B2 (no_swe):      Mean pool → CSE, no SWE attention. MLM-only pretraining.
B3 (swe_with_pe): Standard NEST + learnable positional encoding. MLM + MSM pretraining.

Usage:
    python train_ablation.py --variant swe_only \\
        --data_path dataset/mimic4/data/mimic4_tokens.parquet \\
        --output_dir checkpoints/ablation_pretrain/swe_only \\
        --use_amp

    python train_ablation.py --variant swe_with_pe \\
        --data_path dataset/mimic4/data/mimic4_tokens.parquet \\
        --output_dir checkpoints/ablation_pretrain/swe_with_pe \\
        --masking_strategy both --use_amp
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from src.tokenizer import get_tokenizer
from src.pretrain.data_utils import EHRDataset
from src.fm import FMConfig
from src.pretrain.trainer import BaseTrainer, BaseWithHeadsTrainer, EarlyStopping, CheckpointManager
from src.ablation.models import create_ablation_pretrain_model


def parse_args():
    parser = argparse.ArgumentParser(description="NEST Ablation Pre-training")

    # Ablation variant
    parser.add_argument("--variant", type=str, required=True,
                        choices=["swe_only", "no_swe", "swe_with_pe"],
                        help="Ablation variant to pretrain")

    # Data
    parser.add_argument("--data_path", type=str,
                        default="dataset/mimic4/data/mimic4_tokens.parquet")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/ablation_pretrain")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true")

    # Masking (B3 can use "both" for MLM+MSM; B1/B2 forced to "token")
    parser.add_argument("--masking_strategy", type=str, default="token",
                        choices=["token", "both"],
                        help="'token' (MLM only) or 'both' (MLM+MSM, B3 only)")
    parser.add_argument("--token_mask_prob", type=float, default=0.20)
    parser.add_argument("--encounter_mask_prob", type=float, default=0.40)
    parser.add_argument("--mlm_weight", type=float, default=1.0)
    parser.add_argument("--dm_weight", type=float, default=1.0)

    # Model
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--max_seg", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--swe_rope", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--use_t2v", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--t2v_scale", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Other
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mlflow", action="store_true")

    # Checkpoint
    parser.add_argument("--start_saving_after", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate: B1/B2 must use MLM-only
    if args.variant in ("swe_only", "no_swe") and args.masking_strategy != "token":
        print(f"Warning: {args.variant} requires MLM-only pretraining. "
              f"Forcing masking_strategy='token'.")
        args.masking_strategy = "token"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output directory
    output_dir = Path(args.output_dir) / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ablation pre-training: {args.variant}")
    print(f"Output: {output_dir}")

    # ================================================================
    # Data loading (same as train.py)
    # ================================================================
    data_path = Path(args.data_path)

    if data_path.is_dir():
        if list(data_path.glob("subject_id=*")):
            partition_col = "subject_id"
            enc_col_raw = "hadm_id"
        elif list(data_path.glob("user_id=*")):
            partition_col = "user_id"
            enc_col_raw = "order_id"
        else:
            raise ValueError(f"Cannot find partitions in {data_path}")

        print(f"Detected Hive partitions: {partition_col}=*")
        print("Using LAZY LOADING mode")

        # Tokenizer: read all patient codes
        print("Loading patient data for tokenizer training...")
        all_tokenizer_dirs = sorted(data_path.glob(f"{partition_col}=*"))
        if args.max_patients:
            all_tokenizer_dirs = all_tokenizer_dirs[:args.max_patients]

        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def read_patient_codes(subject_dir):
            try:
                table = pq.read_table(subject_dir, columns=[partition_col, 'code'])
                return table.to_pandas()
            except Exception:
                return None

        dfs = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(read_patient_codes, d): d for d in all_tokenizer_dirs}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Loading for tokenizer"):
                result = future.result()
                if result is not None:
                    dfs.append(result)

        df_sample = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df_sample):,} events from {len(dfs)} patients\n")

        tokenizer = get_tokenizer([df_sample], {
            "tokenizer_path": str(output_dir / "tokenizer.json"),
            "patient_id_col": partition_col,
            "token_col": "code",
            "min_frequency": 5,
        })
        vocab_size = tokenizer.get_vocab_size()
        print(f"Vocab size: {vocab_size}\n")

        # Split patients
        all_subject_dirs = sorted(data_path.glob(f"{partition_col}=*"))
        if args.max_patients:
            all_subject_dirs = all_subject_dirs[:args.max_patients]
        all_patient_ids = [d.name.split('=')[1] for d in all_subject_dirs]
        np.random.shuffle(all_patient_ids)

        n_train = int(len(all_patient_ids) * args.train_ratio)
        n_val = int(len(all_patient_ids) * args.val_ratio)
        train_ids = set(all_patient_ids[:n_train])
        val_ids = set(all_patient_ids[n_train:n_train + n_val])

        print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, "
              f"Test={len(all_patient_ids) - n_train - n_val}\n")

        train_cohort = pd.DataFrame({partition_col: list(train_ids)})
        val_cohort = pd.DataFrame({partition_col: list(val_ids)})

        common_cfg = {
            'tokenizer': tokenizer,
            'max_seg': args.max_seg,
            'max_seq_len': args.max_seq_len,
            'patient_id_col': partition_col,
            'enc_id_col': enc_col_raw,
            'token_col': 'code',
            'time_col': 'days_since_prior_admission' if partition_col == 'subject_id' else 'days_since_prior_order',
            'sort_col': 'visit_seq' if partition_col == 'subject_id' else 'order_number',
            'token_time_col': 'time_offset_hours' if partition_col == 'subject_id' else None,
        }

        use_gpu = (args.device == "cuda" and torch.cuda.is_available())
        train_loader = DataLoader(
            EHRDataset(data=str(data_path), supervised_task_cohort=train_cohort, **common_cfg),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=use_gpu,
        )
        val_loader = DataLoader(
            EHRDataset(data=str(data_path), supervised_task_cohort=val_cohort, **common_cfg),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=use_gpu,
        )
    else:
        raise ValueError("Single parquet file not supported. Use Hive-partitioned directory.")

    # ================================================================
    # Model
    # ================================================================
    config = FMConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout,
        swe_rope=args.swe_rope, use_t2v=args.use_t2v, t2v_scale=args.t2v_scale,
    )

    model, needs_heads = create_ablation_pretrain_model(
        config, args.variant, max_seq_len=args.max_seq_len
    )

    device = torch.device(args.device)
    model = model.to(device)

    print(f"Model: {args.variant} | Needs heads: {needs_heads}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}\n")

    # ================================================================
    # Trainer
    # ================================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    if needs_heads and args.masking_strategy == "both":
        # B3: MLM + MSM
        trainer = BaseWithHeadsTrainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            early_stopping=EarlyStopping(patience=args.patience, min_delta=0.001,
                                         mode="min", use_mlflow=args.use_mlflow),
            verbose_period=1,
            device=device,
            local_rank=0,
            token_mask_prob=args.token_mask_prob,
            encounter_mask_prob=args.encounter_mask_prob,
            use_mlflow=args.use_mlflow,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp,
            mlm_weight=args.mlm_weight,
            dm_weight=args.dm_weight,
        )
        print(f"Trainer: BaseWithHeadsTrainer (MLM+MSM)")
        print(f"Loss weights: MLM={args.mlm_weight}, DM={args.dm_weight}")
    else:
        # B1/B2 or any MLM-only
        trainer = BaseTrainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(ignore_index=-100),
            early_stopping=EarlyStopping(patience=args.patience, min_delta=0.001,
                                         mode="min", use_mlflow=args.use_mlflow),
            verbose_period=1,
            device=device,
            local_rank=0,
            use_encounter_masking=False,
            token_mask_prob=args.token_mask_prob,
            encounter_mask_prob=args.encounter_mask_prob,
            use_mlflow=args.use_mlflow,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp,
        )
        print(f"Trainer: BaseTrainer (MLM only)")

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir=str(output_dir),
        start_saving_after=args.start_saving_after,
        keep_last_n=3,
    )
    ckpt_manager.register_model(model, optimizer, trainer.scaler)

    # Resume
    start_epoch = 0
    if args.resume_from:
        ckpt = ckpt_manager.load_checkpoint(model, optimizer, trainer.scaler, args.resume_from)
        if ckpt:
            start_epoch = ckpt['epoch'] + 1
    else:
        latest_ckpt = output_dir / "latest_checkpoint.pt"
        if latest_ckpt.exists():
            ckpt = ckpt_manager.load_checkpoint(model, optimizer, trainer.scaler)
            if ckpt:
                start_epoch = ckpt['epoch'] + 1

    # Save config
    save_config = {**config.to_dict(), "variant": args.variant, "max_seq_len": args.max_seq_len}
    with open(output_dir / "config.json", "w") as f:
        json.dump(save_config, f, indent=2)

    # ================================================================
    # Training loop
    # ================================================================
    print(f"\n{'='*80}")
    print(f"Training ablation variant: {args.variant}")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
    print(f"{'='*80}\n")

    best_val_loss = float('inf')
    use_dual_loss = needs_heads and args.masking_strategy == "both"

    for epoch in range(start_epoch, args.num_epochs):
        ckpt_manager.update_epoch(epoch, best_val_loss)

        trainer._train(train_loader, verbose=True, epoch_id=epoch)
        metrics = trainer.evaluate(val_loader, verbose=True, epoch_id=epoch)

        if use_dual_loss:
            val_mlm = metrics["mlm_loss"]
            val_dm = metrics["dm_loss"]
            val_loss = args.mlm_weight * val_mlm + args.dm_weight * val_dm
            print(f"Epoch {epoch} | val_mlm: {val_mlm:.4f} | val_dm: {val_dm:.4f} | "
                  f"top1: {metrics['top1_acc']:.4f} | recall@10: {metrics['recall_10']:.4f}")
        else:
            val_loss = metrics["mlm_loss"]
            print(f"Epoch {epoch} | val_mlm: {val_loss:.4f} | "
                  f"top1: {metrics['top1_acc']:.4f} | recall@10: {metrics['recall_10']:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if ckpt_manager.should_save(epoch) or is_best:
            # Save with config for easy finetune loading
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.to_dict(),
                'variant': args.variant,
            }
            if trainer.scaler is not None:
                ckpt_data['scaler_state_dict'] = trainer.scaler.state_dict()

            if is_best:
                torch.save(ckpt_data, output_dir / "best_model.pt")
                print(f"  Saved best model (val_loss={val_loss:.4f})")

            ckpt_manager.save_checkpoint(
                model, optimizer, epoch, val_loss,
                scaler=trainer.scaler, is_best=is_best,
            )

        if trainer.early_stopping:
            trainer.early_stopping.step(val_loss, model)
            if trainer.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nPre-training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
