#!/bin/bash

# Train & Evaluate Pipeline (Shell Script)
# ==========================================
# Trains for 200 episodes by default and automatically evaluates
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh                    # 200 episodes
#   ./run_full_pipeline.sh 50                 # 50 episodes
#   ./run_full_pipeline.sh 100 50             # 100 train, 50 eval

set -e  # Exit on error

# Default values
TRAIN_EPISODES=${1:-200}
EVAL_EPISODES=${2:-30}

echo "========================================"
echo "🚀 TRAIN & EVALUATE PIPELINE"
echo "========================================"
echo "Training Episodes:  $TRAIN_EPISODES"
echo "Evaluation Episodes: $EVAL_EPISODES"
echo ""

# Activate venv
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Step 1: Train
echo ""
echo "========================================"
echo "STEP 1: TRAINING ($TRAIN_EPISODES episodes)"
echo "========================================"
python3 train_chronological.py --episodes $TRAIN_EPISODES --log-episodes 1

# Step 2: Get latest run
echo ""
echo "Finding latest run..."
LATEST_RUN=$(ls -td output_data/*ppo_run_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ Error: No training run found"
    exit 1
fi

LATEST_RUN_NAME=$(basename "$LATEST_RUN")
echo "✅ Latest run: $LATEST_RUN_NAME"

# Step 3: Evaluate
echo ""
echo "========================================"
echo "STEP 2: EVALUATION ($EVAL_EPISODES episodes)"
echo "========================================"
python3 evaluate_model.py --run-dir "$LATEST_RUN" --eval-episodes $EVAL_EPISODES

# Step 4: Print summary paths
echo ""
echo "========================================"
echo "📁 RESULTS SAVED TO:"
echo "========================================"
echo "Training Results:   $LATEST_RUN/"
echo "Evaluation Results: $LATEST_RUN/evaluation_results/"
echo ""
echo "Key Files:"
echo "  - episode_metrics.csv"
echo "  - training_summary.txt"
echo "  - evaluation_results/evaluation_summary.txt"
echo "  - evaluation_results/evaluation_results.csv"
echo ""
echo "✅ PIPELINE COMPLETE!"
echo "========================================"
