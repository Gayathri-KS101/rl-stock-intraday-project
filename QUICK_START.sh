#!/bin/bash
# 
# QUICK START GUIDE
# =================
# Copy-paste commands below to get started
#

# 1. Navigate to project
cd /home/gayathri/min\ proj2/rl-stock-intraday-project

# 2. Activate virtual environment
source .venv/bin/activate

# ============================================
# OPTION A: Python Script (Most Flexible)
# ============================================

# Default: 200 episodes training, 30 evaluation
python3 train_and_evaluate.py

# Quick test: 20 episodes
python3 train_and_evaluate.py --episodes 20

# Thorough: 500 episodes
python3 train_and_evaluate.py --episodes 500 --eval-episodes 50

# ============================================
# OPTION B: Shell Script (Simplest)
# ============================================

# Default: 200 episodes training, 30 evaluation
./run_full_pipeline.sh

# Custom: 100 training, 25 evaluation
./run_full_pipeline.sh 100 25

# ============================================
# OPTIONAL: Monitor on Dashboard
# ============================================
# In another terminal (while pipeline runs):

cd /home/gayathri/min\ proj2/rl-stock-intraday-project
source .venv/bin/activate
streamlit run dashboard.py
# Then open http://localhost:8501

# ============================================
# AFTER PIPELINE COMPLETES
# ============================================
# Results are automatically saved to:
# output_data/fake_ppo_run_YYYY-MM-DD_HH-MM-SS/

# View evaluation results:
cat output_data/fake_ppo_run_*/evaluation_results/evaluation_summary.txt

# ============================================
# COMMAND REFERENCE
# ============================================

# View all available runs
ls -la output_data/ | grep fake_ppo

# Run only evaluation (don't train)
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-20_10-42-15 --eval-episodes 30

# Run only training (don't evaluate)
python3 train_chronological.py --episodes 200

# ============================================
# ARGS FOR train_and_evaluate.py
# ============================================
# --episodes N              : Training episodes (default: 200)
# --eval-episodes N         : Evaluation episodes (default: 30)
# --log-episodes N          : Log every N episodes (default: 1)
# --initial-balance AMOUNT  : Starting balance (default: 10000)

# Example:
python3 train_and_evaluate.py --episodes 100 --eval-episodes 50 --log-episodes 5

# ============================================
# ARGS FOR run_full_pipeline.sh
# ============================================
# Arg 1: Training episodes (default: 200)
# Arg 2: Evaluation episodes (default: 30)

# Example:
./run_full_pipeline.sh 150 40
