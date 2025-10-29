#!/bin/bash

# HAT Model Training Launch Script

echo "=========================================="
echo "HAT Model Training - MIMIC-IV"
echo "=========================================="

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hat

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Training parameters
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/run_${TIMESTAMP}"

echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Starting training..."

python train.py \
    --data_path dataset/mimic4/data/mimic4_tokens.parquet \
    --d_model 768 \
    --n_heads 12 \
    --n_blocks 6 \
    --d_ff 3072 \
    --dropout 0.1 \
    --max_seg 32 \
    --max_seq_len 512 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --patience 10 \
    --masking_strategy encounter \
    --encounter_mask_prob 0.3 \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/train_${TIMESTAMP}.log"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Log file: logs/train_${TIMESTAMP}.log"
echo "Model files: ${OUTPUT_DIR}"
echo "==========================================" 
