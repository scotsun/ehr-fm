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
# Masking Strategies:
#   - token:     Token-level masking (15%), MLM loss only
#   - encounter: Segment-level masking (20%), DM loss only
#   - both:      Segment-level masking (20%), MLM + DM loss (default)
#
# Usage:
#   sbatch run_training.sh              # Default: both (MLM + DM)
#   sbatch run_training.sh token        # MLM only
#   sbatch run_training.sh encounter    # DM only (ablation)
#   sbatch run_training.sh both         # MLM + DM
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
# Masking strategy: "token" (MLM only), "encounter" (DM only), "both" (MLM+DM)
MASKING_STRATEGY=${1:-both}

# Set parameters based on masking strategy
case "$MASKING_STRATEGY" in
    "token")
        MASK_PROB_ARG="--token_mask_prob 0.15"
        EXTRA_ARGS=""
        ;;
    "encounter")
        MASK_PROB_ARG="--encounter_mask_prob 0.2"
        EXTRA_ARGS="--mlm_weight 0.0 --dm_weight 1.0"
        ;;
    "both")
        MASK_PROB_ARG="--encounter_mask_prob 0.2"
        EXTRA_ARGS="--mlm_weight 1.0 --dm_weight 1.0"
        ;;
    *)
        echo "Error: Invalid masking_strategy '$MASKING_STRATEGY'. Use 'token', 'encounter', or 'both'."
        exit 1
        ;;
esac

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/run_${MASKING_STRATEGY}_${TIMESTAMP}"

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "Training Configuration:"
echo "  Masking strategy: ${MASKING_STRATEGY}"
echo "  Context window:   8 segments × 512 tokens = 4,096 tokens/patient"
echo "  Batch size:       24 (real) × 2 (accumulation) = 48 (effective)"
echo "  Mixed Precision: AMP enabled (FP16)"
echo "  Architecture:    6 layers, 768 dim, 12 heads"
echo "  Time encoding:   T2V (scale=1.0) + RoPE"
echo "  Checkpoint:      Save every epoch after epoch 5 (keep last 3)"
echo "  Output:          ${OUTPUT_DIR}"
echo ""

# Start training
python train.py \
    --data_path /hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet \
    --output_dir "${OUTPUT_DIR}" \
    --masking_strategy "${MASKING_STRATEGY}" \
    --batch_size 24 \
    --num_epochs 100 \
    ${MASK_PROB_ARG} \
    --d_model 768 \
    --n_heads 12 \
    --n_blocks 6 \
    --d_ff 2048 \
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
