#!/usr/bin/env python3
"""
Train & Evaluate Pipeline
==========================
Trains the model for 200 episodes and automatically evaluates on test set.

Usage:
    python3 train_and_evaluate.py
    
    Or with custom settings:
    python3 train_and_evaluate.py --episodes 100 --eval-episodes 30
"""

import os
import sys
import subprocess
import glob
from datetime import datetime

def get_latest_run(root_dir="output_data"):
    """Get the latest training run directory"""
    pattern = os.path.join(root_dir, "*ppo_run_*")
    runs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return runs[0] if runs else None

def run_training(episodes=200, log_episodes=1):
    """Run training script"""
    print("\n" + "="*100)
    print("STEP 1: TRAINING")
    print("="*100)
    print(f"Training for {episodes} episodes...\n")
    
    cmd = [
        "python3", "train_chronological.py",
        "--episodes", str(episodes),
        "--log-episodes", str(log_episodes)
    ]
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ Training completed successfully!")
    return True

def run_evaluation(run_dir, eval_episodes=30):
    """Run evaluation script"""
    print("\n" + "="*100)
    print("STEP 2: EVALUATION")
    print("="*100)
    print(f"Evaluating on test set ({eval_episodes} episodes)...\n")
    
    if not os.path.exists(run_dir):
        print(f"❌ Error: Run directory not found: {run_dir}")
        return False
    
    cmd = [
        "python3", "evaluate_model.py",
        "--run-dir", run_dir,
        "--eval-episodes", str(eval_episodes)
    ]
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"\n❌ Evaluation failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ Evaluation completed successfully!")
    return True

def print_results_summary(run_dir):
    """Print summary of results"""
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    
    eval_summary_path = os.path.join(run_dir, "evaluation_results", "evaluation_summary.txt")
    training_summary_path = os.path.join(run_dir, "training_summary.txt")
    
    if os.path.exists(eval_summary_path):
        print(f"\n📊 Evaluation Results:")
        print("-" * 100)
        with open(eval_summary_path, 'r') as f:
            content = f.read()
            # Print key sections
            lines = content.split('\n')
            in_test_section = False
            in_comparison = False
            for line in lines:
                if 'TEST SET RESULTS' in line:
                    in_test_section = True
                elif 'TRAIN vs TEST' in line:
                    in_comparison = True
                    in_test_section = False
                elif 'COMPARISON' in line and in_comparison:
                    in_comparison = True
                
                if in_test_section or in_comparison:
                    if line.strip() and not '=' in line[:10]:
                        print(line)
    
    if os.path.exists(training_summary_path):
        print(f"\n📈 Training Results:")
        print("-" * 100)
        with open(training_summary_path, 'r') as f:
            content = f.read()
            # Print key sections
            lines = content.split('\n')
            in_metrics = False
            for i, line in enumerate(lines):
                if 'EPISODE METRICS STATISTICS' in line:
                    in_metrics = True
                elif 'OUTPUT FILES' in line:
                    break
                
                if in_metrics and line.strip() and not '=' in line[:10]:
                    print(line)
    
    print("\n" + "="*100)
    print("📁 Output Files:")
    print("="*100)
    print(f"Training Outputs:  {run_dir}/")
    print(f"Evaluation Outputs: {run_dir}/evaluation_results/")
    print(f"\nKey Files:")
    print(f"  - episode_metrics.csv (training metrics)")
    print(f"  - training_summary.txt (training stats)")
    print(f"  - evaluation_results/evaluation_summary.txt (test stats)")
    print(f"  - evaluation_results/evaluation_results.csv (detailed test metrics)")
    print(f"  - policy_net.pth (trained model)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate DQN agent in one go.')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes (default: 200)')
    parser.add_argument('--log-episodes', type=int, default=1,
                       help='Log every N episodes (default: 1, i.e., log all)')
    parser.add_argument('--eval-episodes', type=int, default=30,
                       help='Number of evaluation episodes (default: 30)')
    parser.add_argument('--initial-balance', type=float, default=10000,
                       help='Starting balance (default: 10000)')
    
    args = parser.parse_args()
    
    print("\n" + "="*100)
    print("🚀 TRAIN & EVALUATE PIPELINE")
    print("="*100)
    print(f"Episodes:           {args.episodes}")
    print(f"Log Frequency:      Every {args.log_episodes} episode(s)")
    print(f"Eval Episodes:      {args.eval_episodes}")
    print(f"Initial Balance:    ${args.initial_balance:,.2f}")
    
    # Step 1: Train
    if not run_training(args.episodes, args.log_episodes):
        sys.exit(1)
    
    # Step 2: Get latest run
    latest_run = get_latest_run()
    if not latest_run:
        print(f"❌ Error: No training run found")
        sys.exit(1)
    
    print(f"\n✅ Latest run: {os.path.basename(latest_run)}")
    
    # Step 3: Evaluate
    if not run_evaluation(latest_run, args.eval_episodes):
        sys.exit(1)
    
    # Step 4: Print summary
    print_results_summary(latest_run)
    
    print("\n" + "="*100)
    print("✅ PIPELINE COMPLETE!")
    print("="*100)
