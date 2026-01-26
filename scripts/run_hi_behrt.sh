#!/bin/bash
#SBATCH --job-name=hi_behrt
#SBATCH --output=logs/hi_behrt_%j.log
#SBATCH --error=logs/hi_behrt_%j.err
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
# Hi-BEHRT Training with BYOL Pre-training
#
# Hi-BEHRT uses BYOL (Bootstrap Your Own Latent) for self-supervised pre-training.
# This is the CORE of Hi-BEHRT - without BYOL pretraining, it's not true Hi-BEHRT!
#
# Reference: Hi-BEHRT: Hierarchical Transformer-based model for accurate
#            prediction of clinical events (Li et al., 2021)
#
# Workflow:
#   1. BYOL Pre-training: Self-supervised learning on all patient data
#   2. Fine-tuning: Task-specific training using pretrained weights
#
# Usage:
#   sbatch run_hi_behrt.sh pretrain              # Step 1: BYOL pretraining
#   sbatch run_hi_behrt.sh finetune mortality    # Step 2: Finetune on mortality
#   sbatch run_hi_behrt.sh finetune readmission  # Step 2: Finetune on readmission
#   sbatch run_hi_behrt.sh finetune los          # Step 2: Finetune on LOS
#   sbatch run_hi_behrt.sh finetune icd_chapter  # Step 2: Finetune on ICD chapter
#   sbatch run_hi_behrt.sh finetune icd_multilabel  # Step 2: Finetune on ICD multilabel
#   sbatch run_hi_behrt.sh finetune next_visit   # Step 2: Finetune on next visit
# ============================================================================

# Check arguments
MODE=${1:-pretrain}
TASK=${2:-mortality}

echo "=========================================="
echo "Hi-BEHRT Training with BYOL"
echo "Mode: ${MODE}"
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

# Unified model configuration for fair comparison with HAT
D_MODEL=768
N_EXTRACTOR_LAYERS=4  # Paper uses 4+4
N_AGGREGATOR_LAYERS=4
N_HEADS=12
D_FF=2048
WINDOW_SIZE=50
STRIDE=30
T2V_DIM=64  # Time2Vec dimension

# Data sequence length configuration (unified to 2048 for consistency)
MAX_TOTAL_LEN=2048      # For pretrain (flat sequence)
MAX_SEG=8               # For finetune (hierarchical: max_seg * max_seq_len = 2048)
MAX_SEQ_LEN=256         # Tokens per segment

# Data paths
DATA_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet"
LABELS_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/downstream_labels.csv"
TOKENIZER_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/tokenizer.json"

# BYOL pretrained checkpoint path (set after pretraining)
BYOL_CHECKPOINT_DIR="checkpoints/hi-behrt-byol"

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ==================== BYOL Pretraining ====================
if [ "$MODE" = "pretrain" ]; then
    echo "=========================================="
    echo "Stage 1: BYOL Self-Supervised Pre-training"
    echo "=========================================="
    echo ""
    echo "BYOL is the CORE of Hi-BEHRT!"
    echo "  - Online network + Target network (EMA updated)"
    echo "  - Segment-level masking augmentation"
    echo "  - Cosine similarity loss"
    echo ""

    OUTPUT_DIR="${BYOL_CHECKPOINT_DIR}/${TIMESTAMP}"

    echo "Training Configuration:"
    echo "  Model:          Hi-BEHRT BYOL"
    echo "  d_model:        ${D_MODEL}"
    echo "  extractor:      ${N_EXTRACTOR_LAYERS} layers"
    echo "  aggregator:     ${N_AGGREGATOR_LAYERS} layers"
    echo "  window_size:    ${WINDOW_SIZE}, stride: ${STRIDE}"
    echo "  max_total_len:  ${MAX_TOTAL_LEN}"
    echo "  t2v_dim:        ${T2V_DIM}"
    echo "  Output:         ${OUTPUT_DIR}"
    echo ""

    python train_hi_behrt_byol.py \
        --data_path "${DATA_PATH}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size 32 \
        --num_epochs 100 \
        --d_model ${D_MODEL} \
        --n_extractor_layers ${N_EXTRACTOR_LAYERS} \
        --n_aggregator_layers ${N_AGGREGATOR_LAYERS} \
        --n_heads ${N_HEADS} \
        --d_ff ${D_FF} \
        --window_size ${WINDOW_SIZE} \
        --stride ${STRIDE} \
        --max_total_len ${MAX_TOTAL_LEN} \
        --t2v_dim ${T2V_DIM} \
        --byol_momentum 0.996 \
        --mask_probability 0.5 \
        --learning_rate 1e-4 \
        --patience 5 \
        --use_amp

    echo ""
    echo "BYOL pretraining complete!"
    echo "Checkpoint saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Next step: Run finetuning with:"
    echo "  sbatch run_hi_behrt.sh finetune mortality --pretrained_path ${OUTPUT_DIR}/best_model.pt"

# ==================== Fine-tuning ====================
elif [ "$MODE" = "finetune" ]; then
    # Validate task argument
    if [[ ! "$TASK" =~ ^(mortality|readmission|los|icd_chapter|icd_multilabel|next_visit)$ ]]; then
        echo "Error: Invalid task '$TASK'. Use 'mortality', 'readmission', 'los', 'icd_chapter', 'icd_multilabel', or 'next_visit'."
        exit 1
    fi

    echo "=========================================="
    echo "Stage 2: Fine-tuning on ${TASK^^}"
    echo "=========================================="

    # Map task names
    case "$TASK" in
        "mortality")
            TASK_ARG="mortality"
            ;;
        "readmission")
            TASK_ARG="readmission_30d"
            ;;
        "los")
            TASK_ARG="prolonged_los"
            ;;
        "icd_chapter")
            TASK_ARG="icd_chapter"
            ;;
        "icd_multilabel")
            TASK_ARG="icd_category_multilabel"
            ;;
        "next_visit")
            TASK_ARG="next_visit"
            ;;
    esac

    OUTPUT_DIR="checkpoints/hi-behrt_${TASK}_${TIMESTAMP}"

    # Find latest BYOL checkpoint if not specified
    PRETRAINED_PATH=${3:-""}
    if [ -z "$PRETRAINED_PATH" ]; then
        # Find most recent BYOL checkpoint
        LATEST_BYOL=$(ls -td ${BYOL_CHECKPOINT_DIR}/*/best_model.pt 2>/dev/null | head -1)
        if [ -n "$LATEST_BYOL" ]; then
            PRETRAINED_PATH="$LATEST_BYOL"
            echo "Using latest BYOL checkpoint: ${PRETRAINED_PATH}"
        else
            echo "WARNING: No BYOL pretrained checkpoint found!"
            echo "Training from scratch (not recommended for Hi-BEHRT)"
            echo ""
            echo "To run BYOL pretraining first:"
            echo "  sbatch run_hi_behrt.sh pretrain"
            echo ""
        fi
    fi

    echo ""
    echo "Training Configuration:"
    echo "  Model:          Hi-BEHRT"
    echo "  Task:           ${TASK_ARG}"
    echo "  d_model:        ${D_MODEL}"
    echo "  extractor:      ${N_EXTRACTOR_LAYERS} layers"
    echo "  aggregator:     ${N_AGGREGATOR_LAYERS} layers"
    echo "  window_size:    ${WINDOW_SIZE}, stride: ${STRIDE}"
    echo "  max_seg:        ${MAX_SEG}, max_seq_len: ${MAX_SEQ_LEN} (total: $((MAX_SEG * MAX_SEQ_LEN)))"
    echo "  t2v_dim:        ${T2V_DIM}"
    echo "  Pretrained:     ${PRETRAINED_PATH:-None (training from scratch)}"
    echo "  Output:         ${OUTPUT_DIR}"
    echo ""

    # Build command
    CMD="python train_hi_behrt.py \
        --task ${TASK_ARG} \
        --data_path ${DATA_PATH} \
        --labels_path ${LABELS_PATH} \
        --tokenizer_path ${TOKENIZER_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --num_epochs 30 \
        --d_model ${D_MODEL} \
        --n_extractor_layers ${N_EXTRACTOR_LAYERS} \
        --n_aggregator_layers ${N_AGGREGATOR_LAYERS} \
        --n_heads ${N_HEADS} \
        --d_ff ${D_FF} \
        --window_size ${WINDOW_SIZE} \
        --stride ${STRIDE} \
        --max_seg ${MAX_SEG} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --t2v_dim ${T2V_DIM} \
        --learning_rate 5e-5 \
        --patience 2 \
        --use_amp"

    # Add pretrained path if available
    if [ -n "$PRETRAINED_PATH" ]; then
        CMD="$CMD --pretrained_path ${PRETRAINED_PATH}"
    fi

    eval $CMD

else
    echo "Error: Invalid mode '$MODE'. Use 'pretrain' or 'finetune'."
    echo ""
    echo "Usage:"
    echo "  sbatch run_hi_behrt.sh pretrain              # BYOL pretraining"
    echo "  sbatch run_hi_behrt.sh finetune <task>       # Finetuning"
    echo ""
    echo "Tasks: mortality, readmission, los, icd_chapter, icd_multilabel, next_visit"
    exit 1
fi

echo ""
echo "=========================================="
echo "Script completed!"
echo "End: $(date)"
echo "=========================================="
