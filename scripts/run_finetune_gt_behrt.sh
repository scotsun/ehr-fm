#!/bin/bash
#SBATCH --job-name=gtbehrt_finetune
#SBATCH --output=finetune_logs/gtbehrt_finetune_%j.log
#SBATCH --error=finetune_logs/gtbehrt_finetune_%j.err
#SBATCH --partition=h200ea
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=36G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu

# ============================================================================
# GT-BEHRT Fine-tuning on Downstream Tasks
#
# Tasks: mortality, readmission_30d, prolonged_los, icd_chapter, icd_category_multilabel, next_visit
#
# Usage:
#   sbatch run_finetune_gt_behrt.sh mortality
#   sbatch run_finetune_gt_behrt.sh readmission_30d
#
# Or run all tasks:
#   for task in mortality readmission_30d prolonged_los icd_chapter icd_category_multilabel next_visit; do
#     sbatch run_finetune_gt_behrt.sh $task
#   done
# ============================================================================

# Parse arguments
TASK=${1:-mortality}

if [[ ! "$TASK" =~ ^(mortality|readmission_30d|prolonged_los|icd_chapter|icd_category_multilabel|next_visit)$ ]]; then
    echo "Error: Invalid task '$TASK'."
    echo "Available tasks: mortality, readmission_30d, prolonged_los, icd_chapter, icd_category_multilabel, next_visit"
    exit 1
fi

echo "=========================================="
echo "GT-BEHRT Fine-tuning on ${TASK}"
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
# Pre-trained model checkpoint (update to your actual path)
PRETRAINED="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/gt-behrt/best_model.pt"

# Data paths
DATA_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet"
LABELS_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/downstream_labels.csv"
TOKENIZER_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/gt-behrt/tokenizer.json"
OUTPUT_DIR="/hpc/group/engelhardlab/hg176/ehr-fm/checkpoints/finetune"

# Model configuration (must match pre-trained model)
HIDDEN_SIZE=540
N_GRAPH_LAYERS=3
N_BERT_LAYERS=6
N_HEADS=12
MAX_VISITS=50
MAX_CODES_PER_VISIT=100

# Training hyperparameters
LEARNING_RATE=1e-5
EPOCHS=30
WARMUP_RATIO=0.1
PATIENCE=5
BATCH_SIZE=8
GRAD_ACCUM=4  # Effective batch size = 8 × 4 = 32

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "Configuration:"
echo "  Task:             $TASK"
echo "  Pretrained:       $PRETRAINED"
echo "  hidden_size:      $HIDDEN_SIZE"
echo "  n_graph_layers:   $N_GRAPH_LAYERS"
echo "  n_bert_layers:    $N_BERT_LAYERS"
echo "  n_heads:          $N_HEADS"
echo "  max_visits:       $MAX_VISITS"
echo "  batch_size:       $BATCH_SIZE"
echo "  grad_accum:       $GRAD_ACCUM (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  learning_rate:    $LEARNING_RATE"
echo "  epochs:           $EPOCHS"
echo "  patience:         $PATIENCE"
echo "=========================================="
echo ""

# Run fine-tuning
python finetune_gt_behrt.py \
    --task "$TASK" \
    --pretrained "$PRETRAINED" \
    --data_path "$DATA_PATH" \
    --labels_path "$LABELS_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --hidden_size $HIDDEN_SIZE \
    --n_graph_layers $N_GRAPH_LAYERS \
    --n_bert_layers $N_BERT_LAYERS \
    --n_heads $N_HEADS \
    --max_visits $MAX_VISITS \
    --max_codes_per_visit $MAX_CODES_PER_VISIT \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --patience $PATIENCE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --use_amp

echo ""
echo "=========================================="
echo "Fine-tuning completed!"
echo "End: $(date)"
echo "Log: finetune_logs/gtbehrt_finetune_${SLURM_JOB_ID}.log"
echo "=========================================="
