#!/bin/bash
#SBATCH --job-name=baseline_finetune
#SBATCH --output=finetune_logs/baseline_finetune_%j.log
#SBATCH --error=finetune_logs/baseline_finetune_%j.err
#SBATCH --partition=h200ea
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=36G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu

# ============================================================================
# Baseline Models Fine-tuning on Downstream Tasks
#
# Models: core-behrt, heart
# Tasks: mortality, readmission_30d, prolonged_los, icd_chapter
#
# Usage:
#   sbatch run_finetune_baselines.sh core-behrt mortality
#   sbatch run_finetune_baselines.sh heart readmission_30d
#
# Or run all tasks for a model:
#   for task in mortality readmission_30d prolonged_los icd_chapter; do
#     sbatch run_finetune_baselines.sh core-behrt $task
#   done
# ============================================================================

# Parse arguments
MODEL=${1:-core-behrt}
TASK=${2:-mortality}

if [[ ! "$MODEL" =~ ^(core-behrt|heart)$ ]]; then
    echo "Error: Invalid model '$MODEL'. Use 'core-behrt' or 'heart'."
    exit 1
fi

if [[ ! "$TASK" =~ ^(mortality|readmission_30d|prolonged_los|icd_chapter)$ ]]; then
    echo "Error: Invalid task '$TASK'."
    echo "Available tasks: mortality, readmission_30d, prolonged_los, icd_chapter"
    exit 1
fi

echo "=========================================="
echo "Baseline Fine-tuning: ${MODEL^^} on ${TASK}"
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
mkdir -p checkpoints/finetune finetune_logs

# ==================== Configuration ====================
# Pre-trained model checkpoints
# Update these paths to your actual checkpoint locations
CORE_BEHRT_PRETRAINED="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/core-behrt/best_model.pt"
HEART_PRETRAINED="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/heart/best_model.pt"

# Data paths
DATA_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet"
LABELS_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/downstream_labels.csv"
TOKENIZER_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/tokenizer.json"
OUTPUT_DIR="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/finetune"

# Unified model configuration (must match pre-trained model)
D_MODEL=768
N_BLOCKS=6
N_HEADS=12
D_FF=2048

# Training hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=32
EPOCHS=30
WARMUP_RATIO=0.1
PATIENCE=5
DROPOUT=0.1  # For classifier head

# Select pretrained checkpoint based on model
case "$MODEL" in
    "core-behrt")
        PRETRAINED=$CORE_BEHRT_PRETRAINED
        ;;
    "heart")
        PRETRAINED=$HEART_PRETRAINED
        BATCH_SIZE=16  # Smaller due to memory
        ;;
esac

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "Configuration:"
echo "  Model:         $MODEL"
echo "  Task:          $TASK"
echo "  Pretrained:    $PRETRAINED"
echo "  d_model:       $D_MODEL"
echo "  n_blocks:      $N_BLOCKS"
echo "  n_heads:       $N_HEADS"
echo "  d_ff:          $D_FF"
echo "  batch_size:    $BATCH_SIZE"
echo "  learning_rate: $LEARNING_RATE"
echo "  epochs:        $EPOCHS"
echo "  patience:      $PATIENCE"
echo "  dropout:       $DROPOUT"
echo "=========================================="
echo ""

# Run fine-tuning
python finetune_baselines.py \
    --model "$MODEL" \
    --task "$TASK" \
    --pretrained "$PRETRAINED" \
    --data_path "$DATA_PATH" \
    --labels_path "$LABELS_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --d_model $D_MODEL \
    --n_blocks $N_BLOCKS \
    --n_heads $N_HEADS \
    --d_ff $D_FF \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --patience $PATIENCE \
    --dropout $DROPOUT \
    --use_amp

echo ""
echo "=========================================="
echo "Fine-tuning completed!"
echo "End: $(date)"
echo "Log: finetune_logs/baseline_finetune_${SLURM_JOB_ID}.log"
echo "=========================================="
