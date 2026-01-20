#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=logs/baseline_%j.log
#SBATCH --error=logs/baseline_%j.err
#SBATCH --partition=h200ea
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=36G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu
#SBATCH --signal=B:TERM@120  # Send SIGTERM 120 seconds before timeout for graceful checkpoint save

# ============================================================================
# Baseline Models MLM Pre-training - MIMIC-IV on H200 GPU
#
# Models (MLM pre-training):
#   - core-behrt: Flat Transformer with T2V + RoPE (BEHRT-style)
#   - heart:      Heterogeneous Relation-Aware Transformer
#
# Note: Hi-BEHRT does NOT support MLM pre-training.
#       Use run_hi_behrt.sh for Hi-BEHRT end-to-end training on downstream tasks.
#
# Usage:
#   sbatch run_baselines.sh core-behrt    # Train CORE-BEHRT
#   sbatch run_baselines.sh heart         # Train HEART
#
# Or run both:
#   for m in core-behrt heart; do sbatch run_baselines.sh $m; done
# ============================================================================

# Check model argument
MODEL=${1:-core-behrt}

if [[ ! "$MODEL" =~ ^(core-behrt|heart)$ ]]; then
    echo "Error: Invalid model '$MODEL'. Use 'core-behrt' or 'heart'."
    echo "Note: Hi-BEHRT does not support MLM pre-training. Use run_hi_behrt.sh instead."
    exit 1
fi

echo "=========================================="
echo "Baseline Model Training: ${MODEL^^}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Load conda environment
source /hpc/group/rekerlab/apps/miniforge3/etc/profile.d/conda.sh
conda activate hat
echo "Activated hat environment"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /hpc/group/engelhardlab/hg176/ehr-fm

# Create output directories
mkdir -p checkpoints logs

# ==================== Configuration ====================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/${MODEL}_${TIMESTAMP}"

# Model-specific configurations (from original papers/code)
# CORE-BEHRT: arXiv:2404.15201 - "6 layers, 6 heads, hidden_size=192, intermediate_size=64"
# HEART: pretrain.py defaults - "5 layers, 6 heads, hidden_size=288, intermediate_size=288"
case "$MODEL" in
    "core-behrt")
        D_MODEL=192
        N_BLOCKS=6
        N_HEADS=6
        D_FF=64
        DROPOUT=0.1
        LEARNING_RATE=1e-3  # Original paper uses 1e-3 for pretraining
        MASK_PROB=0.15
        BATCH_SIZE=32
        ;;
    "heart")
        D_MODEL=288
        N_BLOCKS=5
        N_HEADS=6
        D_FF=288
        DROPOUT=0.2
        LEARNING_RATE=2e-5
        MASK_PROB=0.7  # HEART uses 70% mask rate for MEP
        BATCH_SIZE=24  # Smaller due to edge module memory
        ;;
esac

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "Training Configuration:"
echo "  Model:          ${MODEL}"
echo "  d_model:        ${D_MODEL}"
echo "  n_blocks:       ${N_BLOCKS}"
echo "  n_heads:        ${N_HEADS}"
echo "  d_ff:           ${D_FF}"
echo "  dropout:        ${DROPOUT}"
echo "  batch_size:     ${BATCH_SIZE}"
echo "  max_seq_len:    512"
echo "  mask_prob:      ${MASK_PROB}"
echo "  learning_rate:  ${LEARNING_RATE}"
echo "  Mixed Precision: AMP enabled"
echo "  Output:         ${OUTPUT_DIR}"
echo ""

# Start training
python train_baselines.py \
    --model "${MODEL}" \
    --data_path /hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet \
    --tokenizer_path /hpc/group/engelhardlab/hg176/ehr-fm/tokenizer.json \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs 50 \
    --d_model ${D_MODEL} \
    --n_blocks ${N_BLOCKS} \
    --n_heads ${N_HEADS} \
    --d_ff ${D_FF} \
    --max_seq_len 512 \
    --mask_prob ${MASK_PROB} \
    --learning_rate ${LEARNING_RATE} \
    --patience 10 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 5.0 \
    --dropout ${DROPOUT} \
    --save_every 5 \
    --use_amp

echo ""
echo "=========================================="
echo "Training completed!"
echo "End: $(date)"
echo "Log: logs/baseline_${SLURM_JOB_ID}.log"
echo "Model: ${OUTPUT_DIR}"
echo "=========================================="
