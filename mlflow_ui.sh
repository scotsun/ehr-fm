#!/bin/bash

# Start MLflow UI to view experiment results

echo "=========================================="
echo "Starting MLflow UI"
echo "=========================================="
echo ""

# Activate environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hat

# Check if mlflow is installed
if ! command -v mlflow &> /dev/null; then
    echo "❌ MLflow not installed!"
    echo "Installing MLflow..."
    pip install mlflow
fi

echo "Starting MLflow UI on http://localhost:5000"
echo ""
echo "Keep this terminal open while using MLflow UI"
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

