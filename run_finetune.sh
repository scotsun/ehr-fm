#!/bin/bash
# Fine-tuning script for HAT downstream tasks
#
# Usage:
#   ./run_finetune.sh mortality           # Single task
#   ./run_finetune.sh all                 # All tasks
#   ./run_finetune.sh mortality --wandb   # With wandb logging

set -e

# ==================== Configuration ====================
PRETRAINED="checkpoints/best.pt"           # Path to pre-trained checkpoint
DATA_PATH="dataset/mimic4/data/mimic4_tokens.parquet"
LABELS_PATH="dataset/mimic4/data/downstream_labels.csv"
TOKENIZER_PATH="dataset/mimic4/data/mapping/mimic4_tokenizer.json"
OUTPUT_DIR="checkpoints/finetune"

# Training hyperparameters
LR=2e-5
BATCH_SIZE=32
EPOCHS=10
WARMUP_RATIO=0.1
MAX_SEG=32
MAX_SEQ_LEN=512

# ==================== Parse Arguments ====================
TASK=$1
shift || true  # Remove first argument

# Check if --wandb flag is present
WANDB_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--wandb" ]; then
        WANDB_FLAG="--wandb"
    fi
done

# ==================== Run Fine-tuning ====================
echo "=============================================="
echo "HAT Fine-tuning for Downstream Tasks"
echo "=============================================="
echo "Task:        $TASK"
echo "Pretrained:  $PRETRAINED"
echo "LR:          $LR"
echo "Batch size:  $BATCH_SIZE"
echo "Epochs:      $EPOCHS"
echo "=============================================="

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
echo "Fine-tuning complete!"
echo "Results saved to: $OUTPUT_DIR"
