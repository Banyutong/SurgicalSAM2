#!/bin/bash

# Baseline evaluation runner script for SAM2 video training
# Executes baseline evaluation on all combo configurations

set -e  # Exit on any error

echo "🚀 Starting baseline evaluation for all combo configurations..."
echo "=================================================="

# Change to script directory
cd "$(dirname "$0")"

# Check if python environment is activated
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected. Make sure you have the correct environment activated."
fi

# Run the baseline evaluation
echo "Running baseline_eval.py..."
python baseline_eval.py

echo "=================================================="
echo "✅ Baseline evaluation completed!"
echo "Results saved to: baseline_results/"
echo "Summary available at: baseline_results/summary_results.csv"