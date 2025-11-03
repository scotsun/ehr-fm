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
    
    # Check if it's a single parquet file or directory
    if data_path.is_file():
        # Single parquet file (e.g., sample_mimic4_30patients.parquet)
        df = pd.read_parquet(data_path)
        print(f"Loaded from single file: {data_path}")
    else:
        # Directory with subject_id partitions
        subject_dirs = sorted(data_path.glob("subject_id=*"))
        if args.max_patients:
            subject_dirs = subject_dirs[:args.max_patients]
        
        dfs = []
        for subject_dir in tqdm(subject_dirs):
            for pf in subject_dir.glob("*.parquet"):
                dfs.append(pd.read_parquet(pf))
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded from {len(subject_dirs)} patient directories")
    
    # Rename columns if needed
    if 'subject_id' in df.columns:
        df = df.rename(columns={'subject_id': 'patient_id', 'hadm_id': 'visit_id'})
    
    print(f"{len(df):,} events, {df['patient_id'].nunique():,} patients\n")
    
    # Tokenizer
    print("Preparing tokenizer...")
    tokenizer = get_tokenizer([df], {
        "tokenizer_path": str(Path(args.output_dir) / "tokenizer.json"),
        "patient_id_col": "patient_id",
        "token_col": "code",
        "min_frequency": 1,
    })
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}\n")
    
    # Split datasets
    print("Splitting datasets...")
    patients = df['patient_id'].unique()
    np.random.shuffle(patients)
    
    n_train = int(len(patients) * args.train_ratio)
    n_val = int(len(patients) * args.val_ratio)
    
    train_df = df[df['patient_id'].isin(patients[:n_train])]
    val_df = df[df['patient_id'].isin(patients[n_train:n_train+n_val])]
    test_df = df[df['patient_id'].isin(patients[n_train+n_val:])]
    
    print(f"Train: {n_train}, Val: {n_val}, Test: {len(patients)-n_train-n_val}\n")
    
    # Create datasets
    common_cfg = {
        'tokenizer': tokenizer, 'max_seg': args.max_seg, 'max_seq_len': args.max_seq_len,
        'patient_id_col': 'patient_id', 'enc_id_col': 'visit_id', 'token_col': 'code',
        'time_col': 'days_since_prior_admission', 'sort_col': 'visit_seq',
        'token_time_col': 'time_offset_hours'
    }
    
    use_gpu = (args.device == "cuda" and torch.cuda.is_available())
    
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
        n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout
    ))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Trainer
    use_encounter = (args.masking_strategy == "encounter")
    
    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01),
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
        early_stopping=EarlyStopping(patience=args.patience, min_delta=0.001, mode="min", use_mlflow=args.use_mlflow),
        verbose_period=1,
        device=torch.device(args.device),
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
    
    # Start MLflow run if enabled
    if args.use_mlflow:
        import mlflow
        run_name = f"{args.masking_strategy}_bs{args.batch_size}_lr{args.learning_rate}"
        mlflow_run = mlflow.start_run(run_name=run_name)
        
        # Log all parameters
        mlflow.log_params({
            "data_path": args.data_path,
            "n_patients": df['patient_id'].nunique(),
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
