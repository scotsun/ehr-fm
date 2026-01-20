#!/usr/bin/env python3
"""
HAT Model Training Script
Self-supervised pre-training with encounter-level masking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from src.fm import FMConfig, FMBase, FMBaseWithHeads
from src.pretrain.trainer import BaseTrainer, BaseWithHeadsTrainer, EarlyStopping, CheckpointManager

def parse_args():
    parser = argparse.ArgumentParser(description="HAT Model Training")
    
    # ========================================================================
    # CRITICAL PARAMETERS - Must configure for your setup
    # ========================================================================
    parser.add_argument("--data_path", type=str, 
                       default="dataset/mimic4/data/mimic4_tokens.parquet",
                       help="Path to MIMIC-IV parquet data")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory for models and logs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (adjust for GPU memory)")
    parser.add_argument("--num_workers", type=int, default=6,
                       help="Number of DataLoader workers (default: 6)")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--masking_strategy", type=str, default="both",
                       choices=["token", "encounter", "both", "staged"],
                       help="'token' (MLM only), 'encounter' (DM only), 'both' (MLM+DM), 'staged' (MLM→DM)")
    parser.add_argument("--stage1_epochs", type=int, default=20,
                       help="Number of epochs for Stage 1 (MLM only) in 'staged' strategy (default: 20)")

    # ========================================================================
    # MODEL PARAMETERS - Optimized defaults, rarely need changes
    # ========================================================================
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--max_seg", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--swe_rope", type=lambda x: x.lower() == 'true',
                       default=True,
                       help="Whether to use RoPE in SWE")
    parser.add_argument("--use_t2v", type=lambda x: x.lower() == 'true',
                       default=True,
                       help="Whether to use Time2Vec encoding")
    parser.add_argument("--t2v_scale", type=float, default=1.0,
                       help="Time2Vec scale factor")

    # Dual-task loss weights (used when masking_strategy='encounter' or 'both')
    parser.add_argument("--mlm_weight", type=float, default=1.0,
                       help="Weight for MLM loss (default: 1.0)")
    parser.add_argument("--dm_weight", type=float, default=1.0,
                       help="Weight for Distribution Matching loss (default: 1.0)")
    
    # ========================================================================
    # TRAINING PARAMETERS - Good defaults (aligned with teammate's config)
    # ========================================================================
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--token_mask_prob", type=float, default=0.20,
                       help="Token-level masking probability for MLM (default: 0.20)")
    parser.add_argument("--encounter_mask_prob", type=float, default=0.40,
                       help="Segment-level masking probability for DM (default: 0.40)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps (effective batch_size=32)")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                       help="Max gradient norm for clipping (0 = disabled)")

    # ========================================================================
    # OPTIONAL - For debugging/testing
    # ========================================================================
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mlflow", action="store_true",
                       help="Enable MLflow tracking (default: False)")
    parser.add_argument("--use_amp", action="store_true",
                       help="Enable Automatic Mixed Precision (AMP) for faster training (default: False)")

    # ========================================================================
    # CHECKPOINT PARAMETERS - For Slurm job recovery
    # ========================================================================
    parser.add_argument("--start_saving_after", type=int, default=5,
                       help="Start saving checkpoints after N epochs (default: 5)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from (default: auto-detect latest)")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow if enabled
    if args.use_mlflow:
        import mlflow
        mlflow.set_tracking_uri("file:./mlruns")
        experiment_name = "HAT_MIMIC_Pretraining"
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass  # Experiment already exists
        mlflow.set_experiment(experiment_name)
        print(f"✅ MLflow initialized: {experiment_name}")
        print(f"   Tracking URI: {mlflow.get_tracking_uri()}\n")
    
    # Load data
    print("Loading data...")
    data_path = Path(args.data_path)
    
    # Determine partition column name
    if data_path.is_dir():
        # Check partition structure
        if list(data_path.glob("subject_id=*")):
            partition_col = "subject_id"
            enc_col_raw = "hadm_id"
        elif list(data_path.glob("user_id=*")):
            partition_col = "user_id"
            enc_col_raw = "order_id"
        else:
            raise ValueError(f"Cannot find subject_id=* or user_id=* partitions in {data_path}")
        
        print(f"Detected Hive partitions: {partition_col}=*")
        
        # Use lazy loading for large datasets
        print(f"Using LAZY LOADING mode (memory-efficient)")
        
        # For tokenizer: use ALL patients for complete vocabulary coverage
        print("Loading ALL patient data for tokenizer training...")
        all_tokenizer_dirs = sorted(data_path.glob(f"{partition_col}=*"))
        if args.max_patients:
            all_tokenizer_dirs = all_tokenizer_dirs[:args.max_patients]
        
        print(f"Total patients for tokenizer: {len(all_tokenizer_dirs)}")
        print("Reading patient data with PyArrow + multi-threading (fast)...")
        
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def read_patient_codes(subject_dir):
            """Read minimal columns needed for tokenizer training"""
            try:
                # Need partition_col (subject_id/user_id) for groupby + code column
                table = pq.read_table(subject_dir, columns=[partition_col, 'code'])
                return table.to_pandas()
            except Exception:
                return None
        
        # Multi-threaded reading (much faster!)
        dfs = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(read_patient_codes, d): d for d in all_tokenizer_dirs}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading for tokenizer"):
                result = future.result()
                if result is not None:
                    dfs.append(result)
        
        df_sample = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df_sample):,} events from {len(dfs)} patients for complete vocab\n")
        
        # Tokenizer
        print("Training tokenizer...")
        tokenizer = get_tokenizer([df_sample], {
            "tokenizer_path": str(Path(args.output_dir) / "tokenizer.json"),
            "patient_id_col": partition_col,
            "token_col": "code",
            "min_frequency": 5,
        })
        vocab_size = tokenizer.get_vocab_size()
        print(f"Vocab size: {vocab_size}\n")
        
        # Get all patient IDs for splitting
        all_subject_dirs = sorted(data_path.glob(f"{partition_col}=*"))
        if args.max_patients:
            all_subject_dirs = all_subject_dirs[:args.max_patients]
        
        all_patient_ids = [d.name.split('=')[1] for d in all_subject_dirs]
        print(f"Total patients: {len(all_patient_ids)}")
        
        # Split patients
        np.random.shuffle(all_patient_ids)
        n_train = int(len(all_patient_ids) * args.train_ratio)
        n_val = int(len(all_patient_ids) * args.val_ratio)
        
        train_ids = set(all_patient_ids[:n_train])
        val_ids = set(all_patient_ids[n_train:n_train+n_val])
        test_ids = set(all_patient_ids[n_train+n_val:])
        
        print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}\n")
        
        # Create cohort DataFrames for filtering
        train_cohort = pd.DataFrame({partition_col: list(train_ids)})
        val_cohort = pd.DataFrame({partition_col: list(val_ids)})
        test_cohort = pd.DataFrame({partition_col: list(test_ids)})
        
    else:
        # Single parquet file (legacy for small datasets)
        print(f"Loading from single file: {data_path}")
        df = pd.read_parquet(data_path)
        
        # Rename columns if needed
        if 'subject_id' in df.columns:
            df = df.rename(columns={'subject_id': 'patient_id', 'hadm_id': 'visit_id'})
            partition_col = 'patient_id'
            enc_col_raw = 'visit_id'
        elif 'user_id' in df.columns:
            df = df.rename(columns={'user_id': 'patient_id', 'order_id': 'visit_id'})
            partition_col = 'patient_id'
            enc_col_raw = 'visit_id'
        else:
            partition_col = 'patient_id'
            enc_col_raw = 'visit_id'
        
        print(f"{len(df):,} events, {df['patient_id'].nunique():,} patients\n")
        
        # Tokenizer
        print("Preparing tokenizer...")
        tokenizer = get_tokenizer([df], {
            "tokenizer_path": str(Path(args.output_dir) / "tokenizer.json"),
            "patient_id_col": partition_col,
            "token_col": "code",
            "min_frequency": 5,
        })
        vocab_size = tokenizer.get_vocab_size()
        print(f"Vocab size: {vocab_size}\n")
        
        # Split datasets
        print("Splitting datasets...")
        patients = df[partition_col].unique()
        np.random.shuffle(patients)
        
        n_train = int(len(patients) * args.train_ratio)
        n_val = int(len(patients) * args.val_ratio)
        
        train_df = df[df[partition_col].isin(patients[:n_train])]
        val_df = df[df[partition_col].isin(patients[n_train:n_train+n_val])]
        test_df = df[df[partition_col].isin(patients[n_train+n_val:])]
        
        print(f"Train: {n_train}, Val: {n_val}, Test: {len(patients)-n_train-n_val}\n")
        
        train_cohort, val_cohort, test_cohort = None, None, None
    
    # Create datasets with appropriate mode
    common_cfg = {
        'tokenizer': tokenizer, 
        'max_seg': args.max_seg, 
        'max_seq_len': args.max_seq_len,
        'patient_id_col': partition_col, 
        'enc_id_col': enc_col_raw, 
        'token_col': 'code',
        'time_col': 'days_since_prior_admission' if partition_col == 'subject_id' else 'days_since_prior_order',
        'sort_col': 'visit_seq' if partition_col == 'subject_id' else 'order_number',
        'token_time_col': 'time_offset_hours' if partition_col == 'subject_id' else None
    }
    
    use_gpu = (args.device == "cuda" and torch.cuda.is_available())
    
    if data_path.is_dir():
        # Lazy mode
        print("Creating datasets with LAZY LOADING...")
        train_loader = DataLoader(
            EHRDataset(data=str(data_path), supervised_task_cohort=train_cohort, **common_cfg),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=use_gpu
        )
        val_loader = DataLoader(
            EHRDataset(data=str(data_path), supervised_task_cohort=val_cohort, **common_cfg),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_gpu
        )
        test_loader = DataLoader(
            EHRDataset(data=str(data_path), supervised_task_cohort=test_cohort, **common_cfg),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_gpu
        )
    else:
        # Eager mode (legacy)
        print("Creating datasets with EAGER LOADING...")
        train_loader = DataLoader(EHRDataset(data=train_df, **common_cfg), 
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=use_gpu)
        val_loader = DataLoader(EHRDataset(data=val_df, **common_cfg),
                                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=use_gpu)
        test_loader = DataLoader(EHRDataset(data=test_df, **common_cfg),
                                 batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=use_gpu)
    
    # Model
    print("Creating model...")
    config = FMConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout,
        swe_rope=args.swe_rope, use_t2v=args.use_t2v, t2v_scale=args.t2v_scale
    )

    # Choose model based on masking strategy
    if args.masking_strategy == "token":
        model = FMBase(config)
        print(f"Model: FMBase | Strategy: token | Loss: MLM only")
    else:
        # "encounter", "both", or "staged" need FMBaseWithHeads
        model = FMBaseWithHeads(config)
        if args.masking_strategy == "encounter":
            print(f"Model: FMBaseWithHeads | Strategy: encounter | Loss: DM only")
        elif args.masking_strategy == "staged":
            print(f"Model: FMBaseWithHeads | Strategy: staged | Loss: MLM (epoch 0-{args.stage1_epochs-1}) → DM (epoch {args.stage1_epochs}+)")
        else:
            print(f"Model: FMBaseWithHeads | Strategy: both | Loss: MLM + DM")

    # Move model to device (GPU if available)
    device = torch.device(args.device)
    model = model.to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"SWE RoPE: {'Enabled' if args.swe_rope else 'Disabled'} (CSE RoPE: Always Enabled)")
    print(f"T2V: {'Enabled' if args.use_t2v else 'Disabled'}" + (f" (scale={args.t2v_scale})" if args.use_t2v else ""))
    print(f"Device: {device}\n")

    # Trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    if args.masking_strategy == "token":
        # Token-level masking with MLM only (FMBase)
        trainer = BaseTrainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(ignore_index=-100),
            early_stopping=EarlyStopping(patience=args.patience, min_delta=0.001, mode="min", use_mlflow=args.use_mlflow),
            verbose_period=1,
            device=device,
            local_rank=0,
            use_encounter_masking=False,
            encounter_mask_prob=args.encounter_mask_prob,  # Not used when use_encounter_masking=False
            token_mask_prob=args.token_mask_prob,
            use_mlflow=args.use_mlflow,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp
        )
    elif args.masking_strategy == "encounter":
        # DM only mode (FMBaseWithHeads, mlm_weight=0)
        # Still uses dual-line masking but ignores MLM loss
        trainer = BaseWithHeadsTrainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            early_stopping=EarlyStopping(patience=args.patience, min_delta=0.001, mode="min", use_mlflow=args.use_mlflow),
            verbose_period=1,
            device=device,
            local_rank=0,
            token_mask_prob=args.token_mask_prob,
            encounter_mask_prob=args.encounter_mask_prob,
            use_mlflow=args.use_mlflow,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp,
            mlm_weight=0.0,  # DM only
            dm_weight=args.dm_weight,
        )
        print(f"Loss weights: MLM=0.0 (disabled), DM={args.dm_weight}")
    elif args.masking_strategy == "staged":
        # Staged training: MLM first, then DM (FMBaseWithHeads)
        # Dynamically switches loss weights based on epoch
        trainer = BaseWithHeadsTrainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            early_stopping=EarlyStopping(patience=args.patience, min_delta=0.001, mode="min", use_mlflow=args.use_mlflow),
            verbose_period=1,
            device=device,
            local_rank=0,
            token_mask_prob=args.token_mask_prob,
            encounter_mask_prob=args.encounter_mask_prob,
            use_mlflow=args.use_mlflow,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp,
            mlm_weight=1.0,  # Will be dynamically adjusted
            dm_weight=1.0,   # Will be dynamically adjusted
            stage1_epochs=args.stage1_epochs,  # Enable staged training
        )
        print(f"Staged training: Stage 1 (MLM) for epochs 0-{args.stage1_epochs-1}, Stage 2 (DM) for epochs {args.stage1_epochs}+")
    else:
        # "both": Dual-line masking with MLM + DM (FMBaseWithHeads)
        trainer = BaseWithHeadsTrainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            early_stopping=EarlyStopping(patience=args.patience, min_delta=0.001, mode="min", use_mlflow=args.use_mlflow),
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
        print(f"Loss weights: MLM={args.mlm_weight}, DM={args.dm_weight}")

    # Setup checkpoint manager for Slurm graceful shutdown
    ckpt_manager = CheckpointManager(
        checkpoint_dir=args.output_dir,
        start_saving_after=args.start_saving_after,
        keep_last_n=3
    )
    ckpt_manager.register_model(model, optimizer, trainer.scaler)

    # Resume from checkpoint if exists
    start_epoch = 0
    if args.resume_from:
        ckpt = ckpt_manager.load_checkpoint(model, optimizer, trainer.scaler, args.resume_from)
        if ckpt:
            start_epoch = ckpt['epoch'] + 1
    else:
        # Auto-detect latest checkpoint
        latest_ckpt = Path(args.output_dir) / "latest_checkpoint.pt"
        if latest_ckpt.exists():
            ckpt = ckpt_manager.load_checkpoint(model, optimizer, trainer.scaler)
            if ckpt:
                start_epoch = ckpt['epoch'] + 1

    print(f"Masking strategy: {args.masking_strategy}")
    if args.masking_strategy == "token":
        print(f"  - Token mask probability: {args.token_mask_prob}")
    elif args.masking_strategy == "staged":
        print(f"  - Stage 1 (MLM only): epochs 0-{args.stage1_epochs-1}")
        print(f"  - Stage 2 (DM only): epochs {args.stage1_epochs}+")
        print(f"  - Token mask probability: {args.token_mask_prob}")
        print(f"  - Encounter mask probability: {args.encounter_mask_prob}")
    else:
        print(f"  - Token mask probability: {args.token_mask_prob} (token-level)")
        print(f"  - Encounter mask probability: {args.encounter_mask_prob} (segment-level)")
    print(f"Mixed Precision (AMP): {'Enabled' if args.use_amp else 'Disabled'}")
    print()
    
    # Get patient count for logging
    if data_path.is_dir():
        n_patients_total = len(all_patient_ids)
    else:
        n_patients_total = len(patients)
    
    # Start MLflow run if enabled
    if args.use_mlflow:
        import mlflow
        run_name = f"{args.masking_strategy}_bs{args.batch_size}_lr{args.learning_rate}"
        mlflow_run = mlflow.start_run(run_name=run_name)

        # Log all parameters
        params = {
            "masking_strategy": args.masking_strategy,
            "data_path": args.data_path,
            "n_patients": n_patients_total,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "token_mask_prob": args.token_mask_prob,
            "encounter_mask_prob": args.encounter_mask_prob,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_blocks": args.n_blocks,
            "max_seg": args.max_seg,
            "max_seq_len": args.max_seq_len,
            "use_t2v": args.use_t2v,
            "t2v_scale": args.t2v_scale,
        }
        if args.masking_strategy in ["encounter", "both"]:
            params["mlm_weight"] = args.mlm_weight if args.masking_strategy == "both" else 0.0
            params["dm_weight"] = args.dm_weight
        mlflow.log_params(params)
        print(f"✅ MLflow run started: {run_name}")
        print(f"   Run ID: {mlflow_run.info.run_id}\n")

    # Training
    print("="*80)
    print("Training...")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
    print("="*80)

    # Custom training loop with checkpoint saving
    best_val_loss = float('inf')
    use_dual_loss = (args.masking_strategy in ["encounter", "both", "staged"])

    for epoch in range(start_epoch, args.num_epochs):
        ckpt_manager.update_epoch(epoch, best_val_loss)

        # Train one epoch
        trainer._train(train_loader, verbose=True, epoch_id=epoch)

        # Validate (evaluate returns dict with metrics)
        metrics = trainer.evaluate(val_loader, verbose=True, epoch_id=epoch)

        # Handle different return types
        if use_dual_loss:
            val_mlm = metrics["mlm_loss"]
            val_dm = metrics["dm_loss"]
            # For staged training, use epoch-dependent weights
            if args.masking_strategy == "staged":
                if epoch < args.stage1_epochs:
                    eff_mlm_weight, eff_dm_weight = 1.0, 0.0
                    stage_str = f"[Stage1-MLM] "
                else:
                    eff_mlm_weight, eff_dm_weight = 0.0, 1.0
                    stage_str = f"[Stage2-DM] "
            else:
                eff_mlm_weight, eff_dm_weight = args.mlm_weight, args.dm_weight
                stage_str = ""
            val_loss = eff_mlm_weight * val_mlm + eff_dm_weight * val_dm
            print(f"Epoch {epoch} {stage_str}| val_mlm: {val_mlm:.4f} | val_dm: {val_dm:.4f} | "
                  f"top1: {metrics['top1_acc']:.4f} | top10: {metrics['top10_acc']:.4f} | "
                  f"recall@10: {metrics['recall_10']:.4f} | ndcg@10: {metrics['ndcg_10']:.4f}")
            if args.use_mlflow:
                import mlflow
                mlflow.log_metrics({
                    "val_mlm_loss": val_mlm,
                    "val_dm_loss": val_dm,
                    "val_total_loss": val_loss,
                    "val_top1_acc": metrics["top1_acc"],
                    "val_top10_acc": metrics["top10_acc"],
                    "val_recall_10": metrics["recall_10"],
                    "val_ndcg_10": metrics["ndcg_10"],
                }, step=epoch)
        else:
            val_loss = metrics["mlm_loss"]
            print(f"Epoch {epoch} | val_mlm: {val_loss:.4f} | "
                  f"top1: {metrics['top1_acc']:.4f} | top10: {metrics['top10_acc']:.4f} | "
                  f"recall@10: {metrics['recall_10']:.4f} | ndcg@10: {metrics['ndcg_10']:.4f}")
            if args.use_mlflow:
                import mlflow
                mlflow.log_metrics({
                    "val_mlm_loss": val_loss,
                    "val_top1_acc": metrics["top1_acc"],
                    "val_top10_acc": metrics["top10_acc"],
                    "val_recall_10": metrics["recall_10"],
                    "val_ndcg_10": metrics["ndcg_10"],
                }, step=epoch)

        # Check if best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save checkpoint periodically or if best
        if ckpt_manager.should_save(epoch) or is_best:
            ckpt_manager.save_checkpoint(
                model, optimizer, epoch, val_loss,
                scaler=trainer.scaler, is_best=is_best
            )

        # Early stopping
        if trainer.early_stopping:
            trainer.early_stopping.step(val_loss, model)
            if trainer.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Test
    print("\n" + "="*80)
    print("Testing...")
    print("="*80)
    test_metrics = trainer.evaluate(test_loader, verbose=True, epoch_id=0)

    if use_dual_loss:
        test_mlm = test_metrics["mlm_loss"]
        test_dm = test_metrics["dm_loss"]
        # For staged training, final test uses DM loss (Stage 2 objective)
        if args.masking_strategy == "staged":
            test_loss = test_dm  # After training, model is optimized for DM
            print(f"Test MLM Loss: {test_mlm:.4f}")
            print(f"Test DM Loss: {test_dm:.4f} (final objective)")
        else:
            test_loss = args.mlm_weight * test_mlm + args.dm_weight * test_dm
            print(f"Test MLM Loss: {test_mlm:.4f}")
            print(f"Test DM Loss: {test_dm:.4f}")
            print(f"Test Total Loss: {test_loss:.4f}")
    else:
        test_loss = test_metrics["mlm_loss"]
        print(f"Test Loss: {test_loss:.4f}")

    print(f"Test Top-1 Acc: {test_metrics['top1_acc']:.4f}")
    print(f"Test Top-10 Acc: {test_metrics['top10_acc']:.4f}")
    print(f"Test Recall@10: {test_metrics['recall_10']:.4f}")
    print(f"Test NDCG@10: {test_metrics['ndcg_10']:.4f}")

    # Log test metrics to MLflow
    if args.use_mlflow:
        test_log = {
            "test_mlm_loss": test_metrics["mlm_loss"],
            "test_top1_acc": test_metrics["top1_acc"],
            "test_top10_acc": test_metrics["top10_acc"],
            "test_recall_10": test_metrics["recall_10"],
            "test_ndcg_10": test_metrics["ndcg_10"],
        }
        if use_dual_loss:
            test_log["test_dm_loss"] = test_dm
            test_log["test_total_loss"] = test_loss
        mlflow.log_metrics(test_log)
        mlflow.end_run()
        print(f"\n✅ MLflow run completed. View at: http://localhost:5000")

if __name__ == "__main__":
    main()
