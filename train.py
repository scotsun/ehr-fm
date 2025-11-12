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
from src.utils.data_utils import EHRDataset
from src.fm import FMConfig, FMBase
from src.trainer import BaseTrainer, EarlyStopping

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
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--masking_strategy", type=str, default="encounter", 
                       choices=["token", "encounter"],
                       help="'encounter' (default) or 'token'")
    
    # ========================================================================
    # MODEL PARAMETERS - Optimized defaults, rarely need changes
    # ========================================================================
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--max_seg", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--swe_rope", type=lambda x: x.lower() == 'true', 
                       default=True,
                       help="Whether to use RoPE in SWE")
    
    # ========================================================================
    # TRAINING PARAMETERS - Good defaults
    # ========================================================================
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--encounter_mask_prob", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps (1=disabled, >1=enabled)")
    parser.add_argument("--max_grad_norm", type=float, default=0.0,
                       help="Max gradient norm for clipping (0=disabled, >0=enabled)")
    
    # ========================================================================
    # OPTIONAL - For debugging/testing
    # ========================================================================
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--token_mask_prob", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mlflow", action="store_true",
                       help="Enable MLflow tracking (default: False)")
    
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
            "min_frequency": 1,
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
            "min_frequency": 1,
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
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu
        )
        val_loader = DataLoader(
            EHRDataset(data=str(data_path), supervised_task_cohort=val_cohort, **common_cfg),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu
        )
        test_loader = DataLoader(
            EHRDataset(data=str(data_path), supervised_task_cohort=test_cohort, **common_cfg),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu
        )
    else:
        # Eager mode (legacy)
        print("Creating datasets with EAGER LOADING...")
        train_loader = DataLoader(EHRDataset(data=train_df, **common_cfg), 
                                  batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=use_gpu)
        val_loader = DataLoader(EHRDataset(data=val_df, **common_cfg),
                                batch_size=args.batch_size, shuffle=False, num_workers=4,
                                pin_memory=use_gpu)
        test_loader = DataLoader(EHRDataset(data=test_df, **common_cfg),
                                 batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=use_gpu)
    
    # Model
    print("Creating model...")
    model = FMBase(FMConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_blocks=args.n_blocks,
        n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout,
        swe_rope=args.swe_rope
    ))
    
    # Move model to device (GPU if available)
    device = torch.device(args.device)
    model = model.to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"SWE RoPE: {'Enabled' if args.swe_rope else 'Disabled'} (CSE RoPE: Always Enabled)")
    print(f"Device: {device}\n")
    
    # Trainer
    use_encounter = (args.masking_strategy == "encounter")
    
    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01),
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
        early_stopping=EarlyStopping(patience=args.patience, min_delta=0.001, mode="min", use_mlflow=args.use_mlflow),
        verbose_period=1,
        device=device,
        local_rank=0,
        use_encounter_masking=use_encounter,
        encounter_mask_prob=args.encounter_mask_prob,
        use_mlflow=args.use_mlflow,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm
    )
    
    print(f"Masking strategy: {args.masking_strategy}")
    if use_encounter:
        print(f"  - Encounter mask probability: {args.encounter_mask_prob}")
    else:
        print(f"  - Token mask probability: {args.token_mask_prob}")
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
        mlflow.log_params({
            "data_path": args.data_path,
            "n_patients": n_patients_total,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "masking_strategy": args.masking_strategy,
            "encounter_mask_prob": args.encounter_mask_prob,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_blocks": args.n_blocks,
            "max_seg": args.max_seg,
            "max_seq_len": args.max_seq_len,
        })
        print(f"✅ MLflow run started: {run_name}")
        print(f"   Run ID: {mlflow_run.info.run_id}\n")
    
    # Training
    print("="*80)
    print("Training...")
    print("="*80)
    
    trainer.fit(epochs=args.num_epochs, train_loader=train_loader, valid_loader=val_loader)
    
    # Test
    print("\n" + "="*80)
    print("Testing...")
    print("="*80)
    test_loss = trainer.evaluate(test_loader, verbose=True, epoch_id=0)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Log test loss to MLflow
    if args.use_mlflow:
        mlflow.log_metric("test_loss", test_loss)
        mlflow.end_run()
        print(f"\n✅ MLflow run completed. View at: http://localhost:5000")

if __name__ == "__main__":
    main()
