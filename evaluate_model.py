"""
Model Evaluation Script
=======================
Evaluates a trained model on the test set (30% of data held out during training).

Usage:
    python evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08
    
    Or with custom parameters:
    python evaluate_model.py --run-dir <path> --eval-episodes 30 --initial-balance 10000

The script will:
1. Load the trained model weights
2. Load the test days data (30% of original data, held out during training)
3. Run the agent on test episodes
4. Generate detailed statistics and visualizations
5. Compare test performance vs training performance
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import random
from datetime import datetime
import glob
import argparse
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import custom modules
from simplified_environment import SimplifiedStockTradingEnv
from simplified_agent import DQN
from data_splitter import load_and_split_data, ChronologicalDaySplitter

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_training_config(run_dir):
    """Load training configuration from summary file"""
    summary_path = os.path.join(run_dir, 'training_summary.txt')
    config = {}
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            content = f.read()
            # Parse config values
            if "Initial Balance:" in content:
                config['initial_balance'] = 10000  # Default, can be extracted
            if "Gamma" in content:
                config['gamma'] = 0.99
    
    return config


def evaluate_agent(run_dir, num_eval_episodes=30, initial_balance=10000):
    """
    Evaluate trained agent on test set
    
    Args:
        run_dir: Path to training run directory
        num_eval_episodes: Number of test episodes to run
        initial_balance: Starting balance for evaluation
    """
    
    print("="*100)
    print("MODEL EVALUATION ON TEST SET (30% HOLD-OUT DATA)")
    print("="*100)
    print(f"\nRun Directory: {run_dir}")
    
    # Check if run exists
    if not os.path.exists(run_dir):
        print(f"❌ Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Load trained model
    model_path = os.path.join(run_dir, 'policy_net.pth')
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found: {model_path}")
        sys.exit(1)
    
    print(f"✅ Loading model: {model_path}")
    
    # Load training metrics to compare against
    metrics_path = os.path.join(run_dir, 'episode_metrics.csv')
    training_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    
    if training_metrics is not None:
        train_mean_return = training_metrics['Percent_Return'].mean()
        train_std_return = training_metrics['Percent_Return'].std()
        train_win_rate = (training_metrics['Percent_Return'] > 0).mean() * 100
        print(f"\n📊 Training Performance (for comparison):")
        print(f"   Mean Return: {train_mean_return:+.2f}%")
        print(f"   Std Return:  {train_std_return:.2f}%")
        print(f"   Win Rate:    {train_win_rate:.1f}%")
    else:
        train_mean_return = train_std_return = train_win_rate = None
    
    # Load and split data
    print(f"\n📂 Loading data...")
    all_files = glob.glob('processed_data/*.csv')
    train_days, test_days, splitter = load_and_split_data(all_files, train_ratio=0.7, min_minutes_per_day=60)
    print(f"✅ Found {len(test_days)} test trading days")
    
    # Initialize agent
    dummy_day_data, dummy_stock_name = test_days[0]
    dummy_env = SimplifiedStockTradingEnv(
        dummy_day_data, stock_name=dummy_stock_name, initial_balance=initial_balance
    )
    input_dim = dummy_env.observation_space.shape[0]
    output_dim = dummy_env.action_space.n
    
    # Load model
    policy_net = DQN(input_dim, output_dim)
    policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy_net.eval()
    print(f"✅ Model loaded (Input: {input_dim}D, Output: {output_dim} actions)")
    
    # Run evaluation episodes
    print(f"\n{'='*100}")
    print(f"RUNNING {num_eval_episodes} EVALUATION EPISODES ON TEST SET")
    print(f"{'='*100}")
    
    eval_metrics = {
        'Episode': [],
        'Stock': [],
        'Total_Reward': [],
        'Final_Net_Worth': [],
        'Percent_Return': [],
        'Number_of_Trades': [],
        'Max_Drawdown': [],
        'Sharpe_Ratio': []
    }
    
    # Create eval output directory
    eval_output_dir = os.path.join(run_dir, 'evaluation_results')
    os.makedirs(eval_output_dir, exist_ok=True)
    
    for eval_ep in range(min(num_eval_episodes, len(test_days))):
        day_data, stock_name = test_days[eval_ep % len(test_days)]
        env = SimplifiedStockTradingEnv(
            day_data, stock_name=stock_name, initial_balance=initial_balance
        )
        
        state, _ = env.reset()
        done = False
        total_reward = 0
        net_worth_history = [env.net_worth]
        
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_t).cpu().numpy()[0]
                action = np.argmax(q_values)  # Greedy (no exploration on test set)
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            net_worth_history.append(env.net_worth)
            state = next_state
        
        # Calculate metrics
        percent_return = ((env.net_worth - env.initial_balance) / env.initial_balance) * 100
        
        # Calculate drawdown
        net_worth_arr = np.array(net_worth_history)
        peak = np.maximum.accumulate(net_worth_arr)
        drawdown = (net_worth_arr - peak) / np.where(peak == 0, 1, peak)
        max_dd = drawdown.min() * 100
        
        # Calculate Sharpe ratio (simplified - using daily returns)
        returns = np.diff(net_worth_arr) / net_worth_arr[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) if len(returns) > 0 else 0
        
        eval_metrics['Episode'].append(eval_ep + 1)
        eval_metrics['Stock'].append(stock_name)
        eval_metrics['Total_Reward'].append(total_reward)
        eval_metrics['Final_Net_Worth'].append(env.net_worth)
        eval_metrics['Percent_Return'].append(percent_return)
        eval_metrics['Number_of_Trades'].append(env.num_trades)
        eval_metrics['Max_Drawdown'].append(max_dd)
        eval_metrics['Sharpe_Ratio'].append(sharpe)
        
        status = "✅ PROFIT" if percent_return > 0 else "❌ LOSS"
        print(f"Eval {eval_ep+1:3d} | {stock_name:12s} | Return: {percent_return:+7.2f}% | "
              f"Net Worth: ${env.net_worth:10,.2f} | Trades: {env.num_trades:3d} | {status}")
    
    # Save evaluation results
    eval_df = pd.DataFrame(eval_metrics)
    eval_csv_path = os.path.join(eval_output_dir, 'evaluation_results.csv')
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"\n✅ Saved evaluation results to: {eval_csv_path}")
    
    # Calculate statistics
    print(f"\n{'='*100}")
    print(f"EVALUATION SUMMARY STATISTICS")
    print(f"{'='*100}\n")
    
    eval_returns = eval_df['Percent_Return'].values
    eval_net_worths = eval_df['Final_Net_Worth'].values
    eval_trades = eval_df['Number_of_Trades'].values
    eval_drawdowns = eval_df['Max_Drawdown'].values
    
    test_win_rate = (eval_returns > 0).mean() * 100
    test_mean_return = eval_returns.mean()
    test_std_return = eval_returns.std()
    
    print(f"TEST SET PERFORMANCE:")
    print(f"  Episodes:              {len(eval_df)}")
    print(f"  Win Rate:              {test_win_rate:.1f}%")
    print(f"  Mean Return:           {test_mean_return:+.2f}%")
    print(f"  Std Dev Return:        {test_std_return:.2f}%")
    print(f"  Min Return:            {eval_returns.min():+.2f}%")
    print(f"  Max Return:            {eval_returns.max():+.2f}%")
    print(f"  Median Return:         {np.median(eval_returns):+.2f}%")
    
    print(f"\n  Mean Net Worth:        ${eval_net_worths.mean():,.2f}")
    print(f"  Mean Max Drawdown:     {eval_drawdowns.mean():.2f}%")
    print(f"  Mean Trades/Episode:   {eval_trades.mean():.1f}")
    print(f"  Total Trades:          {int(eval_trades.sum())}")
    
    # Compare with training
    if train_mean_return is not None:
        print(f"\n{'─'*100}")
        print(f"TRAIN vs TEST COMPARISON:")
        print(f"  Mean Return:           Train: {train_mean_return:+.2f}% → Test: {test_mean_return:+.2f}% "
              f"({test_mean_return - train_mean_return:+.2f}%)")
        print(f"  Win Rate:              Train: {train_win_rate:.1f}% → Test: {test_win_rate:.1f}% "
              f"({test_win_rate - train_win_rate:+.1f}%)")
        print(f"  Std Dev Return:        Train: {train_std_return:.2f}% → Test: {test_std_return:.2f}%")
        
        if test_mean_return < train_mean_return:
            print(f"\n⚠️  Test performance is LOWER than training (possible overfitting)")
        elif test_mean_return > train_mean_return:
            print(f"\n✅ Test performance is BETTER than training (good generalization)")
        else:
            print(f"\n✓ Test performance is SIMILAR to training (good consistency)")
    
    # Save summary
    summary_path = os.path.join(eval_output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Run: {os.path.basename(run_dir)}\n")
        f.write(f"Test Episodes: {len(eval_df)}\n")
        f.write(f"Initial Balance: ${initial_balance:,.2f}\n\n")
        
        f.write("="*100 + "\n")
        f.write("TEST SET RESULTS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Win Rate:              {test_win_rate:.1f}%\n")
        f.write(f"Mean Return:           {test_mean_return:+.2f}%\n")
        f.write(f"Std Dev Return:        {test_std_return:.2f}%\n")
        f.write(f"Min Return:            {eval_returns.min():+.2f}%\n")
        f.write(f"Max Return:            {eval_returns.max():+.2f}%\n")
        f.write(f"Median Return:         {np.median(eval_returns):+.2f}%\n\n")
        
        f.write(f"Mean Net Worth:        ${eval_net_worths.mean():,.2f}\n")
        f.write(f"Mean Max Drawdown:     {eval_drawdowns.mean():.2f}%\n")
        f.write(f"Mean Trades/Episode:   {eval_trades.mean():.1f}\n\n")
        
        if train_mean_return is not None:
            f.write("="*100 + "\n")
            f.write("TRAIN vs TEST COMPARISON\n")
            f.write("="*100 + "\n\n")
            f.write(f"Mean Return:           Train: {train_mean_return:+.2f}% → Test: {test_mean_return:+.2f}%\n")
            f.write(f"Win Rate:              Train: {train_win_rate:.1f}% → Test: {test_win_rate:.1f}%\n")
    
    print(f"\n✅ Saved evaluation summary to: {summary_path}")
    
    print(f"\n{'='*100}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*100}")
    print(f"Results saved to: {eval_output_dir}/")
    
    return eval_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent on test set.')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Path to training run directory (e.g., output_data/fake_ppo_run_2026-05-19_19-54-08)')
    parser.add_argument('--eval-episodes', type=int, default=30,
                       help='Number of evaluation episodes to run (default: 30)')
    parser.add_argument('--initial-balance', type=float, default=10000,
                       help='Starting balance for evaluation (default: 10000)')
    
    args = parser.parse_args()
    
    # Run evaluation
    eval_df = evaluate_agent(args.run_dir, args.eval_episodes, args.initial_balance)
