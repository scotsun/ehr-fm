#!/bin/bash
#SBATCH --job-name=hat_mimic4
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu

# HAT Model Training - MIMIC-IV on DCC

echo "=========================================="
echo "HAT Model Training - MIMIC-IV"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load conda
source /hpc/group/rekerlab/apps/miniforge3/etc/profile.d/conda.sh

# Activate the existing hat environment
conda activate hat
echo "Activated hat environment"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid warnings with multiprocessing DataLoader
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation
cd /hpc/home/hg176/work/ehr-fm/ehr-fm

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Training parameters
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/run_${TIMESTAMP}"

echo "Output directory: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "Starting training..."
echo "Note: Using LAZY LOADING mode for memory-efficient data loading"
echo "      Data will be loaded on-demand from Hive-partitioned parquet files"
echo ""

python train.py \
    --data_path dataset/mimic4/data/mimic4_tokens.parquet \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 1 \
    --num_epochs 100 \
    --masking_strategy encounter \
    --encounter_mask_prob 0.3 \
    --d_model 768 \
    --n_heads 12 \
    --n_blocks 6 \
    --d_ff 3072 \
    --dropout 0.1 \
    --max_seg 32 \
    --max_seq_len 512 \
    --swe_rope True \
    --learning_rate 1e-4 \
    --patience 10 \
    --gradient_accumulation_steps 32 \
    --use_mlflow

echo ""
echo "=========================================="
echo "Training completed!"
echo "Log file: logs/train_${SLURM_JOB_ID}.log"
echo "Model files: ${OUTPUT_DIR}"
echo "=========================================="
