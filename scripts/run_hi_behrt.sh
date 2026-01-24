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
# Hi-BEHRT End-to-End Training - MIMIC-IV on H200 GPU
#
# Hi-BEHRT does NOT support MLM pre-training (by design).
# It is trained end-to-end directly on downstream tasks.
#
# Reference: Hi-BEHRT: Hierarchical Transformer-based model for accurate
#            prediction of clinical events (Li et al., 2021)
#
# Downstream Tasks:
#   - mortality:   In-hospital mortality prediction
#   - readmission: 30-day readmission prediction
#   - los:         Length of stay > 7 days prediction
#
# Usage:
#   sbatch run_hi_behrt.sh mortality      # Train on mortality prediction
#   sbatch run_hi_behrt.sh readmission    # Train on readmission prediction
#   sbatch run_hi_behrt.sh los            # Train on length-of-stay prediction
# ============================================================================

# Check task argument
TASK=${1:-mortality}

if [[ ! "$TASK" =~ ^(mortality|readmission|los|icd_chapter)$ ]]; then
    echo "Error: Invalid task '$TASK'. Use 'mortality', 'readmission', 'los', or 'icd_chapter'."
    exit 1
fi

echo "=========================================="
echo "Hi-BEHRT End-to-End Training: ${TASK^^}"
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
OUTPUT_DIR="checkpoints/hi-behrt_${TASK}_${TIMESTAMP}"

# Unified model configuration for fair comparison with HAT
# All baselines use the same architecture hyperparameters
D_MODEL=768
N_EXTRACTOR_LAYERS=6
N_AGGREGATOR_LAYERS=6
N_HEADS=12
D_FF=2048
WINDOW_SIZE=50
STRIDE=30
BATCH_SIZE=16  # Smaller due to hierarchical model memory
LEARNING_RATE=5e-5

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "Training Configuration:"
echo "  Model:          Hi-BEHRT"
echo "  Task:           ${TASK}"
echo "  d_model:        ${D_MODEL}"
echo "  extractor:      ${N_EXTRACTOR_LAYERS} layers"
echo "  aggregator:     ${N_AGGREGATOR_LAYERS} layers"
echo "  n_heads:        ${N_HEADS}"
echo "  d_ff:           ${D_FF}"
echo "  window_size:    ${WINDOW_SIZE}"
echo "  stride:         ${STRIDE}"
echo "  batch_size:     ${BATCH_SIZE}"
echo "  learning_rate:  ${LEARNING_RATE}"
echo "  Mixed Precision: AMP enabled"
echo "  Output:         ${OUTPUT_DIR}"
echo ""

# Map task names to match train_hi_behrt.py expectations
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
esac

# Data paths
DATA_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet"
LABELS_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/downstream_labels.csv"
TOKENIZER_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/tokenizer.json"

# Run end-to-end training
python train_hi_behrt.py \
    --task "${TASK_ARG}" \
    --data_path "${DATA_PATH}" \
    --labels_path "${LABELS_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs 30 \
    --d_model ${D_MODEL} \
    --n_extractor_layers ${N_EXTRACTOR_LAYERS} \
    --n_aggregator_layers ${N_AGGREGATOR_LAYERS} \
    --n_heads ${N_HEADS} \
    --d_ff ${D_FF} \
    --window_size ${WINDOW_SIZE} \
    --stride ${STRIDE} \
    --learning_rate ${LEARNING_RATE} \
    --patience 10 \
    --use_amp

echo ""
echo "=========================================="
echo "Script completed!"
echo "End: $(date)"
echo "=========================================="
