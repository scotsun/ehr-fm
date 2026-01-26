#!/bin/bash
#SBATCH --job-name=gt_behrt
#SBATCH --output=logs/gt_behrt_%j.log
#SBATCH --error=logs/gt_behrt_%j.err
#SBATCH --partition=h200ea
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=36G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hg176@duke.edu
#SBATCH --signal=B:TERM@120  # Send SIGTERM 120 seconds before timeout

# ============================================================================
# GT-BEHRT Training Script
#
# Graph Transformer BERT for Electronic Health Records
# Reference: "Graph Transformers on EHRs: Better Representation Improves
#            Downstream Performance" (Poulain & Beheshti, ICLR 2024)
#
# Pre-training:
#   Step 1: Node Attribute Masking (NAM) - trains Graph Transformer only
#   Step 2: MNP + VTP - trains full model
#
# Usage:
#   sbatch run_gt_behrt.sh
# ============================================================================

echo "=========================================="
echo "GT-BEHRT Pre-training"
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

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ==================== Configuration ====================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/gt-behrt_${TIMESTAMP}"

# Data paths
DATA_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/dataset/mimic4/data/mimic4_tokens.parquet"
TOKENIZER_PATH="/hpc/group/engelhardlab/hg176/ehr-fm/tokenizer.json"

# Model configuration (GT-BEHRT architectural constraints)
# Note: hidden_size must be 5 * d_stream (paper: 540 = 5 * 108)
# This is an architectural constraint unique to GT-BEHRT
HIDDEN_SIZE=540      # Must be 5 * d_stream (paper default)
N_GRAPH_LAYERS=3     # Graph Transformer layers
N_BERT_LAYERS=6      # BERT Transformer layers
N_HEADS=12           # Attention heads
DROPOUT=0.0          # Aligned with other baselines

# Sequence configuration
MAX_VISITS=50
MAX_CODES_PER_VISIT=100

# Training configuration (aligned with other baselines)
BATCH_SIZE=24        # Larger batch for faster training (GT-BEHRT uses less memory)
NUM_EPOCHS=30
LEARNING_RATE=5e-5   # Aligned with other baselines
WARMUP_RATIO=0.1
PATIENCE=5

# Pre-training configuration
NAM_MASK_PROB=0.20   # Aligned with MASK_PROB in other baselines
VTP_MASK_PROB=0.5    # Visit type masking probability
NAM_EPOCHS=10        # Epochs for NAM-only pre-training

echo "Training Configuration:"
echo "  Output:           ${OUTPUT_DIR}"
echo "  hidden_size:      ${HIDDEN_SIZE}"
echo "  n_graph_layers:   ${N_GRAPH_LAYERS}"
echo "  n_bert_layers:    ${N_BERT_LAYERS}"
echo "  n_heads:          ${N_HEADS}"
echo "  max_visits:       ${MAX_VISITS}"
echo "  max_codes:        ${MAX_CODES_PER_VISIT}"
echo "  batch_size:       ${BATCH_SIZE}"
echo "  learning_rate:    ${LEARNING_RATE}"
echo "  NAM epochs:       ${NAM_EPOCHS}"
echo "  patience:         ${PATIENCE}"
echo "=========================================="
echo ""

# ==================== Training ====================
python train_gt_behrt.py \
    --data_path "${DATA_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --hidden_size ${HIDDEN_SIZE} \
    --n_graph_layers ${N_GRAPH_LAYERS} \
    --n_bert_layers ${N_BERT_LAYERS} \
    --n_heads ${N_HEADS} \
    --dropout ${DROPOUT} \
    --max_visits ${MAX_VISITS} \
    --max_codes_per_visit ${MAX_CODES_PER_VISIT} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --patience ${PATIENCE} \
    --nam_mask_prob ${NAM_MASK_PROB} \
    --vtp_mask_prob ${VTP_MASK_PROB} \
    --nam_epochs ${NAM_EPOCHS} \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 5.0 \
    --save_every 5 \
    --use_amp

echo ""
echo "=========================================="
echo "Training completed!"
echo "End: $(date)"
echo "Log: logs/gt_behrt_${SLURM_JOB_ID}.log"
echo "Model: ${OUTPUT_DIR}"
echo "=========================================="
