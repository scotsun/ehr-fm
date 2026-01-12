#!/bin/bash
#SBATCH --job-name=hat_mimic4
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=h200ea
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu
#SBATCH --signal=B:TERM@120  # Send SIGTERM 120 seconds before timeout for graceful checkpoint save

# ============================================================================
# HAT Model Training - MIMIC-IV on H200 GPU (Optimized)
# Context: max_seg=8, max_seq_len=512 (4,096 tokens/patient)
# Effective batch size: 24 × 2 = 48 patients
# Mixed Precision: AMP enabled (FP16 training for 2-3x speedup)
#
# Model Types:
#   - base: FMBase with MLM only (token/encounter masking)
#   - with_heads: FMBaseWithHeads with MLM + Distribution Matching (dual-task)
#
# Usage:
#   sbatch run_training.sh              # Default: base model with token masking
#   sbatch run_training.sh with_heads   # Dual-task model with encounter masking
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

# ==================== Configuration ====================
# Model type: "base" (MLM only) or "with_heads" (MLM + DM dual-task)
MODEL_TYPE=${1:-base}

# Set masking strategy based on model type
if [ "$MODEL_TYPE" == "with_heads" ]; then
    MASKING_STRATEGY="encounter"
    MASK_PROB_ARG="--encounter_mask_prob 0.2"
    EXTRA_ARGS="--mlm_weight 1.0 --dm_weight 1.0"
else
    MASKING_STRATEGY="token"
    MASK_PROB_ARG="--token_mask_prob 0.15"
    EXTRA_ARGS=""
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/run_${MODEL_TYPE}_${TIMESTAMP}"

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "Training Configuration:"
echo "  Model type:      ${MODEL_TYPE}"
echo "  Context window:  8 segments × 512 tokens = 4,096 tokens/patient"
echo "  Batch size:      24 (real) × 2 (accumulation) = 48 (effective)"
echo "  Masking:         ${MASKING_STRATEGY}"
echo "  Mixed Precision: AMP enabled (FP16)"
echo "  Architecture:    8 layers, 768 dim, 12 heads"
echo "  Time encoding:   T2V (scale=1.0) + RoPE"
echo "  Checkpoint:      Save every epoch after epoch 5 (keep last 3)"
echo "  Output:          ${OUTPUT_DIR}"
echo ""

# Start training
python train.py \
    --data_path /hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet \
    --output_dir "${OUTPUT_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --batch_size 24 \
    --num_epochs 100 \
    --masking_strategy "${MASKING_STRATEGY}" \
    ${MASK_PROB_ARG} \
    --d_model 768 \
    --n_heads 12 \
    --n_blocks 8 \
    --d_ff 3072 \
    --dropout 0.1 \
    --max_seg 8 \
    --max_seq_len 512 \
    --swe_rope True \
    --use_t2v True \
    --t2v_scale 1.0 \
    --learning_rate 5e-5 \
    --patience 10 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 5.0 \
    --start_saving_after 5 \
    --use_mlflow \
    --use_amp \
    ${EXTRA_ARGS}

echo ""
echo "=========================================="
echo "Training completed!"
echo "End: $(date)"
echo "Log: logs/train_${SLURM_JOB_ID}.log"
echo "Model: ${OUTPUT_DIR}"
echo "=========================================="
