#!/bin/bash
#SBATCH --job-name=hat_finetune
#SBATCH --output=finetune_logs/finetune_%j.log
#SBATCH --error=finetune_logs/finetune_%j.err
#SBATCH --partition=h200ea
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=36G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu

# ============================================================================
# HAT Fine-tuning for Downstream Tasks on H200 GPU
# Tasks: mortality, readmission_30d, prolonged_los, icd_chapter, icd_category_multilabel, next_visit
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
# Use token mode pretrained model
PRETRAINED="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/run_token_20260122_210214/best_model.pt"
DATA_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet"
LABELS_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/downstream_labels.csv"
TOKENIZER_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/run_token_20260122_210214/tokenizer.json"
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

# Check for optional flags
WANDB_FLAG=""
K_VALUES="10,20,30"  # Default k values for next_visit task
METRIC=""            # Auto-determined by task type
RESUME_PATH=""       # Resume from checkpoint directory

for arg in "$@"; do
    if [ "$arg" == "--wandb" ]; then
        WANDB_FLAG="--wandb"
    fi
done

# Parse --k-values and --resume flags
for i in $(seq 1 $#); do
    arg="${!i}"
    if [[ "$arg" == "--k-values="* ]]; then
        K_VALUES="${arg#*=}"
    elif [[ "$arg" == "--resume" ]]; then
        next=$((i + 1))
        RESUME_PATH="${!next}"
    fi
done

# Build resume flag
RESUME_FLAG=""
if [ -n "$RESUME_PATH" ]; then
    RESUME_FLAG="--resume $RESUME_PATH"
    echo "  Resuming from: $RESUME_PATH"
fi

echo "Configuration:"
echo "  Task:        $TASK"
echo "  Pretrained:  $PRETRAINED"
echo "  LR:          $LR"
echo "  Batch size:  $BATCH_SIZE"
echo "  Epochs:      $EPOCHS"
echo "  Max seg:     $MAX_SEG"
echo "  Max seq len: $MAX_SEQ_LEN"
if [ "$TASK" == "next_visit" ] || [ "$TASK" == "all" ]; then
    echo "  K values:    $K_VALUES (for next_visit)"
fi
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
        --k-values "$K_VALUES" \
        $WANDB_FLAG $RESUME_FLAG
elif [ "$TASK" == "next_visit" ]; then
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
        --k-values "$K_VALUES" \
        --metric "recall@20" \
        $WANDB_FLAG $RESUME_FLAG
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
        $WANDB_FLAG $RESUME_FLAG
fi

echo ""
echo "=========================================="
echo "Fine-tuning completed!"
echo "End: $(date)"
echo "Log: finetune_logs/finetune_${SLURM_JOB_ID}.log"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
