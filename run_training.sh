#!/bin/bash
#SBATCH --job-name=hat_mimic4
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=h200alloc
#SBATCH --account=engelhardlab
#SBATCH --qos=h200alloc-u
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu

# ============================================================================
# HAT Model Training - MIMIC-IV on H200 GPU (Optimized)
# Context: max_seg=16, max_seq_len=256 (4,096 tokens/patient)
# Effective batch size: 32 × 2 = 64 patients
# Masking: Token-level (basic BERT-style)
# Mixed Precision: AMP enabled (FP16 training for 2-3x speedup)
# ============================================================================

echo "=========================================="
echo "HAT Model Training - MIMIC-IV"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Load conda environment
source /hpc/group/rekerlab/apps/miniforge3/etc/profile.d/conda.sh
conda activate hat
echo "✓ Activated hat environment"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /hpc/group/engelhardlab/hg176/ehr-fm

# Create output directories
mkdir -p checkpoints logs

# Training configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/run_${TIMESTAMP}"

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "Training Configuration:"
echo "  Context window: 16 segments × 256 tokens = 4,096 tokens/patient"
echo "  Batch size: 32 (real) × 2 (accumulation) = 64 (effective)"
echo "  Masking strategy: token-level (25% mask probability)"
echo "  Mixed Precision: AMP enabled (FP16)"
echo "  Model: 8 layers, 768 dim, 12 heads"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Start training
python train.py \
    --data_path /hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 32 \
    --num_epochs 100 \
    --masking_strategy token \
    --token_mask_prob 0.25 \
    --d_model 768 \
    --n_heads 12 \
    --n_blocks 8 \
    --d_ff 3072 \
    --dropout 0.1 \
    --max_seg 16 \
    --max_seq_len 256 \
    --swe_rope True \
    --learning_rate 5e-5 \
    --patience 10 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 0.0 \
    --use_mlflow \
    --use_amp

echo ""
echo "=========================================="
echo "Training completed!"
echo "End: $(date)"
echo "Log: logs/train_${SLURM_JOB_ID}.log"
echo "Model: ${OUTPUT_DIR}"
echo "=========================================="
