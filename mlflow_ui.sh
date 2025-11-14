#!/bin/bash

# Start MLflow UI to view experiment results
# Adapted for DCC cluster environment

echo "=========================================="
echo "Starting MLflow UI on DCC"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load conda (DCC path)
source /hpc/group/rekerlab/apps/miniforge3/etc/profile.d/conda.sh

# Activate environment
conda activate hat
echo "✅ Activated hat environment"

# Check if mlflow is installed
if ! command -v mlflow &> /dev/null; then
    echo "❌ MLflow not installed!"
    echo "Installing MLflow..."
    pip install mlflow
fi

# Set MLflow tracking URI to match training script
export MLFLOW_TRACKING_URI="file:./mlruns"

# Get hostname for display
HOSTNAME=$(hostname)
PORT=${1:-5000}  # Allow port to be specified as argument, default 5000

echo ""
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
echo "MLflow runs directory: $(pwd)/mlruns"
echo ""
echo "Starting MLflow UI..."
echo "  - Host: 0.0.0.0 (accessible from all interfaces)"
echo "  - Port: $PORT"
echo "  - Server: $HOSTNAME"
echo ""
echo "=========================================="
echo "IMPORTANT: To access MLflow UI from your local machine:"
echo "=========================================="
echo ""
echo "1. Open a new terminal on your LOCAL machine"
echo "2. Run this SSH port forwarding command:"
echo ""
echo "   ssh -L $PORT:localhost:$PORT hg176@dcc-login.oit.duke.edu"
echo ""
echo "   Or if you're already SSH'd into DCC, use:"
echo "   ssh -L $PORT:localhost:$PORT $HOSTNAME"
echo ""
echo "3. Then open in your browser:"
echo "   http://localhost:$PORT"
echo ""
echo "=========================================="
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port "$PORT" --backend-store-uri "$MLFLOW_TRACKING_URI"

