#!/bin/bash
#SBATCH --job-name=hat_finetune
#SBATCH --output=finetune_logs/finetune_%j.log
#SBATCH --error=finetune_logs/finetune_%j.err
#SBATCH --partition=h200ea
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu

# ============================================================================
# HAT Fine-tuning for Downstream Tasks on H200 GPU
# Tasks: mortality, readmission_30d, prolonged_los, icd_chapter, icd_category_multilabel
# ============================================================================

echo "=========================================="
echo "HAT Fine-tuning for Downstream Tasks"
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

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ==================== Configuration ====================
# Use the best pretrain from 20251210_081245
PRETRAINED="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/run_20251210_081245/best_model.pt"
DATA_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet"
LABELS_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/downstream_labels.csv"
TOKENIZER_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/run_20251210_081245/tokenizer.json"
OUTPUT_DIR="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/finetune"

# Training hyperparameters
LR=1e-5
BATCH_SIZE=32
EPOCHS=30
WARMUP_RATIO=0.1
MAX_SEG=8      
MAX_SEQ_LEN=512  

# ==================== Parse Arguments ====================
TASK=${1:-mortality}  # Default to mortality if not specified

# Check if --wandb flag is present
WANDB_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--wandb" ]; then
        WANDB_FLAG="--wandb"
    fi
done

echo "Configuration:"
echo "  Task:        $TASK"
echo "  Pretrained:  $PRETRAINED"
echo "  LR:          $LR"
echo "  Batch size:  $BATCH_SIZE"
echo "  Epochs:      $EPOCHS"
echo "  Max seg:     $MAX_SEG"
echo "  Max seq len: $MAX_SEQ_LEN"
echo "=========================================="
echo ""

# ==================== Run Fine-tuning ====================
if [ "$TASK" == "all" ]; then
    python run_finetune.py \
        --all \
        --pretrained "$PRETRAINED" \
        --data-path "$DATA_PATH" \
        --labels-path "$LABELS_PATH" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --lr $LR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --warmup-ratio $WARMUP_RATIO \
        --max-seg $MAX_SEG \
        --max-seq-len $MAX_SEQ_LEN \
        $WANDB_FLAG
else
    python run_finetune.py \
        --task "$TASK" \
        --pretrained "$PRETRAINED" \
        --data-path "$DATA_PATH" \
        --labels-path "$LABELS_PATH" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --lr $LR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --warmup-ratio $WARMUP_RATIO \
        --max-seg $MAX_SEG \
        --max-seq-len $MAX_SEQ_LEN \
        $WANDB_FLAG
fi

echo ""
echo "=========================================="
echo "Fine-tuning completed!"
echo "End: $(date)"
echo "Log: finetune_logs/finetune_${SLURM_JOB_ID}.log"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
