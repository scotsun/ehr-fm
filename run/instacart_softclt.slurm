#!/bin/bash

#SBATCH --job-name=instacart_softclt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00


#SBATCH --partition=h200ea
#SBATCH --account=h200ea

#SBATCH --mail-user=ms1008@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL


# --- SETUP ---
mkdir -p logs

# --- CONTAINER CONFIGURATION ---
CONTAINER=/opt/apps/containers/community/fm101/fm-image.sif

# Bind paths: Make host directories accessible inside container
# Typically you need: your code location, data directories, and scratch space
BIND_PATHS="/hpc,/work,/datacommons"

# --- DEBUGGING INFO ---
echo "Job started on $(hostname) at $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Container: $CONTAINER"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# --- EXECUTION WITH CONTAINER ---
echo "Starting torchrun inside Singularity container..."

# Set OMP threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Execute torchrun inside the container
# --nv: Enable NVIDIA GPU support
# --bind: Mount host directories into container
# --cleanenv: Start with clean environment (optional, but recommended for reproducibility)
apptainer exec --nv \
    --bind $BIND_PATHS \
    $CONTAINER \
    torchrun --standalone --nproc_per_node=1 run/instacart_softclt.py \
        --experiment-name instacart \
        --run-name softclt

echo "Job finished at $(date)"
