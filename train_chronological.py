"""
Updated Training Script with Chronological Train/Test Split

This script fixes temporal leakage by:
- Splitting data chronologically at the trading-day level
- Using only training days during training
- Reserving test days for evaluation
- Preventing any overlap between train and test sets
"""
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd

# Import custom modules
from simplified_environment import load_data, SimplifiedStockTradingEnv
from simplified_agent import (DQN, ReplayBuffer, create_episode_logger, log_step,
                               save_episode_log, print_log_preview, 
                               plot_episode_performance, analyze_episode_decisions)
from data_splitter import load_and_split_data, ChronologicalDaySplitter

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def evaluate_agent(policy_net, test_days, initial_balance, num_eval_episodes=None):
    """
    Evaluate trained agent on test set.
    
    Args:
        policy_net: Trained DQN network
        test_days: List of DataFrames for test days
        initial_balance: Initial trading balance
        num_eval_episodes: Number of episodes to evaluate (None = all test days)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if num_eval_episodes is None:
        eval_days = test_days
    else:
        eval_days = random.sample(test_days, min(num_eval_episodes, len(test_days)))
    
    policy_net.eval()  # Set to evaluation mode
    
    eval_rewards = []
    eval_returns = []
    eval_trades = []
    eval_net_worths = []
    
    print("\n" + "="*100)
    print(f"EVALUATING ON TEST SET ({len(eval_days)} episodes)")
    print("="*100)
    
    for i, (day_data, stock_name) in enumerate(eval_days):
        env = SimplifiedStockTradingEnv(
            day_data,
            stock_name=stock_name,
            initial_balance=initial_balance
        )

        state, _ = env.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            # Use greedy policy (no exploration)
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_tensor = policy_net(state_t)
                action = q_tensor.argmax().item()
            
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
        
        # Calculate metrics
        percent_return = ((env.net_worth - env.initial_balance) / env.initial_balance) * 100
        
        eval_rewards.append(total_reward)
        eval_returns.append(percent_return)
        eval_trades.append(env.num_trades)
        eval_net_worths.append(env.net_worth)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(eval_days):
            print(f"Evaluated {i+1}/{len(eval_days)} episodes...")
    
    # Calculate summary statistics
    eval_metrics = {
        'num_episodes': len(eval_days),
        'avg_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'avg_return': np.mean(eval_returns),
        'std_return': np.std(eval_returns),
        'avg_trades': np.mean(eval_trades),
        'win_rate': sum(1 for r in eval_returns if r > 0) / len(eval_returns) * 100,
        'best_return': np.max(eval_returns),
        'worst_return': np.min(eval_returns),
        'final_avg_net_worth': np.mean(eval_net_worths)
    }
    
    return eval_metrics


def print_evaluation_results(eval_metrics):
    """Print formatted evaluation results."""
    print("\n" + "="*100)
    print("TEST SET EVALUATION RESULTS")
    print("="*100)
    print(f"\nEpisodes Evaluated:     {eval_metrics['num_episodes']}")
    print(f"\nREWARD METRICS:")
    print(f"  Average Reward:       {eval_metrics['avg_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"\nRETURN METRICS:")
    print(f"  Average Return:       {eval_metrics['avg_return']:.2f}% ± {eval_metrics['std_return']:.2f}%")
    print(f"  Best Return:          {eval_metrics['best_return']:.2f}%")
    print(f"  Worst Return:         {eval_metrics['worst_return']:.2f}%")
    print(f"  Win Rate:             {eval_metrics['win_rate']:.2f}%")
    print(f"\nTRADING METRICS:")
    print(f"  Avg Trades/Episode:   {eval_metrics['avg_trades']:.1f}")
    print(f"  Avg Final Net Worth:  ${eval_metrics['final_avg_net_worth']:.2f}")
    print("="*100 + "\n")


def train_agent():
    """Main training function with chronological train/test split."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train simplified stock trading agent with proper data split.')
    parser.add_argument('--episodes', type=int, default=200, 
                        help='Number of training episodes')
    parser.add_argument('--log-episodes', type=int, default=10, 
                        help='Save detailed logs every N episodes (0 to disable)')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                        help='Directory containing CSV data files')
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial trading balance')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Proportion of days to use for training')
    parser.add_argument('--min-minutes', type=int, default=60,
                        help='Minimum minutes required for valid trading day')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate')
    parser.add_argument('--epsilon-min', type=float, default=0.05,
                        help='Minimum epsilon value')
    parser.add_argument('--eval-frequency', type=int, default=50,
                        help='Evaluate on test set every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=20,
                        help='Number of test episodes to run during evaluation')
    args = parser.parse_args()

    # ========================================================================
    # STEP 1: LOAD AND SPLIT DATA CHRONOLOGICALLY
    # ========================================================================
    print("="*100)
    print("LOADING AND SPLITTING DATA")
    print("="*100)
    
    all_files = glob.glob(f'{args.data_dir}/*.csv')
    print(f"Found {len(all_files)} files in {args.data_dir}/")
    
    if not all_files:
        print(f"ERROR: No CSV files found in {args.data_dir}/")
        return
    
    # Load data with chronological split
    train_days, test_days, splitter = load_and_split_data(
        all_files,
        train_ratio=args.train_ratio,
        min_minutes_per_day=args.min_minutes
    )
    
    print(f"✓ Training set: {len(train_days)} days")
    print(f"✓ Test set: {len(test_days)} days")
    print(f"✓ Zero temporal leakage guaranteed\n")
    
    # ========================================================================
    # STEP 2: INITIALIZE AGENT
    # ========================================================================
    print("="*100)
    print("INITIALIZING SIMPLIFIED AGENT")
    print("="*100)
    
    dummy_day_data, dummy_stock_name = train_days[0]
    dummy_env = SimplifiedStockTradingEnv(
        dummy_day_data,
        stock_name=dummy_stock_name,
        initial_balance=args.initial_balance
    )
    input_dim = dummy_env.observation_space.shape[0]
    output_dim = dummy_env.action_space.n
    
    print(f"Input Dimension:  {input_dim} (price window + shares + balance)")
    print(f"Output Dimension: {output_dim} (Hold, Buy 1, Sell 1)")
    print(f"Initial Balance:  ${args.initial_balance:,.2f}")
    print(f"Commission Rate:  {dummy_env.commission*100:.2f}%\n")
    
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(50000)
    
    epsilon = 1.0
    rewards_history = []
    
    episode_metrics = {
        'Episode': [],
        'Total_Reward': [],
        'Final_Net_Worth': [],
        'Percent_Return': [],
        'Number_of_Trades': []
    }
    
    eval_history = []
    
    print(f"Hyperparameters:")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Learning Rate:  {args.learning_rate}")
    print(f"  Gamma:          {args.gamma}")
    print(f"  Epsilon Decay:  {args.epsilon_decay}")
    print(f"  Epsilon Min:    {args.epsilon_min}")
    print(f"  Buffer Size:    50000\n")

    # ========================================================================
    # CREATE TRAINING DATA FOLDER STRUCTURE
    # ========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("output_data", f"run_{timestamp}")
    rewards_dir = os.path.join(base_dir, "training_rewards")
    episode_csv_dir = os.path.join(base_dir, "episode_logs_csv")
    episode_png_dir = os.path.join(base_dir, "episode_logs_png")
    eval_dir = os.path.join(base_dir, "evaluation_results")

    os.makedirs(rewards_dir, exist_ok=True)
    os.makedirs(episode_csv_dir, exist_ok=True)
    os.makedirs(episode_png_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save split information
    split_info = splitter.get_split_info()
    split_df = pd.DataFrame([split_info])
    split_df.to_csv(os.path.join(base_dir, 'data_split_info.csv'), index=False)
    
    # ========================================================================
    # STEP 3: TRAINING LOOP (USING ONLY TRAINING DAYS)
    # ========================================================================
    print("="*100)
    print("STARTING TRAINING (TRAIN SET ONLY)")
    print("="*100)
    
    for episode in range(args.episodes):

        # IMPORTANT: Sample only from training days
        day_data, stock_name = random.choice(train_days)
        env = SimplifiedStockTradingEnv(
            day_data,
            stock_name=stock_name,
            initial_balance=args.initial_balance
        )

        print(f"Episode {episode+1} | Trading Stock: {stock_name}")
        
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        should_log = (args.log_episodes > 0) and ((episode + 1) % args.log_episodes == 0)
        if should_log:
            episode_logger = create_episode_logger()
        
        step_count = 0
        
        while not done:

            was_random = False
            if random.random() < epsilon:
                action = env.action_space.sample()
                was_random = True
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_t).cpu().numpy()[0]
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_tensor = policy_net(state_t)
                    action = q_tensor.argmax().item()
                    q_values = q_tensor.cpu().numpy()[0]
            
            next_state, reward, done, _, _ = env.step(action)
            
            if should_log:
                log_step(episode_logger, episode + 1, step_count, env, state, 
                        action, q_values, epsilon, reward, was_random)
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if len(replay_buffer) > 500:
                transitions = replay_buffer.sample(args.batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.FloatTensor(np.array(batch_state))
                batch_action = torch.LongTensor(batch_action).unsqueeze(1)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                batch_next_state = torch.FloatTensor(np.array(batch_next_state))
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1)

                curr_q = policy_net(batch_state).gather(1, batch_action)
                next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                expected_q = batch_reward + args.gamma * next_q * (1 - batch_done)

                loss = nn.MSELoss()(curr_q, expected_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        if episode % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        percent_return = ((env.net_worth - env.initial_balance) / env.initial_balance) * 100

        episode_metrics['Episode'].append(episode + 1)
        episode_metrics['Total_Reward'].append(total_reward)
        episode_metrics['Final_Net_Worth'].append(env.net_worth)
        episode_metrics['Percent_Return'].append(percent_return)
        episode_metrics['Number_of_Trades'].append(env.num_trades)

        if should_log:
            log_df, log_path = save_episode_log(
                episode_logger, episode + 1, output_dir=episode_csv_dir)

            print(f"\n{'='*100}")
            print(f"Episode {episode+1} Complete - Detailed Analysis")
            print(f"{'='*100}")
            print(f"Saved detailed log to: {log_path}")

            print_log_preview(log_df, num_rows=10)

            plot_path = plot_episode_performance(
                log_df, episode + 1, output_dir=episode_png_dir)
            print(f"Saved performance plot to: {plot_path}")

            analyze_episode_decisions(log_df)

        rewards_history.append(total_reward)

        print(f"Episode {episode+1}: "
              f"Reward: {total_reward:.2f} | "
              f"Net Worth: ${env.net_worth:.2f} | "
              f"Return: {percent_return:.2f}% | "
              f"Trades: {env.num_trades} | "
              f"Epsilon: {epsilon:.2f}")

        # Periodic evaluation on test set
        if (episode + 1) % args.eval_frequency == 0:
            eval_metrics = evaluate_agent(
                policy_net, 
                test_days, 
                args.initial_balance,
                num_eval_episodes=args.eval_episodes
            )
            
            eval_metrics['episode'] = episode + 1
            eval_history.append(eval_metrics)
            
            print_evaluation_results(eval_metrics)
            
            # Save evaluation results
            eval_df = pd.DataFrame(eval_history)
            eval_df.to_csv(os.path.join(eval_dir, 'evaluation_history.csv'), index=False)

        if (episode + 1) % 50 == 0:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards_history, linewidth=2)
            plt.title(f"Training Rewards (Episodes 1-{episode+1})", 
                     fontsize=14, fontweight='bold')
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Total Reward", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(
                rewards_dir, f'training_rewards_ep{episode+1}.png'),
                dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved intermediate plot: training_rewards_ep{episode+1}.png")

    print("\n" + "="*100)
    print("TRAINING COMPLETE")
    print("="*100)

    # Save training metrics
    metrics_df = pd.DataFrame(episode_metrics)
    metrics_df.to_csv(os.path.join(base_dir, 'episode_metrics.csv'), index=False)
    print(f"\nSaved episode metrics to: episode_metrics.csv")

    # Final training plot
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, linewidth=2)
    plt.title("Training Rewards Over All Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(
        rewards_dir, 'training_rewards_final.png'),
        dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Final training plot saved to: training_rewards_final.png")

    # Save model
    torch.save(policy_net.state_dict(),
               os.path.join(base_dir, 'trained_model_simplified.pth'))
    print(f"Model saved to: trained_model_simplified.pth")

    # ========================================================================
    # FINAL EVALUATION ON FULL TEST SET
    # ========================================================================
    print("\n" + "="*100)
    print("FINAL EVALUATION ON FULL TEST SET")
    print("="*100)
    
    final_eval_metrics = evaluate_agent(
        policy_net,
        test_days,
        args.initial_balance,
        num_eval_episodes=200  # Use all test days
    )
    
    print_evaluation_results(final_eval_metrics)
    
    # Save final evaluation
    final_eval_df = pd.DataFrame([final_eval_metrics])
    final_eval_df.to_csv(os.path.join(eval_dir, 'final_test_evaluation.csv'), index=False)

    # ========================================================================
    # TRAINING SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    print(f"\nDATA SPLIT:")
    print(f"  Training Days:          {len(train_days)}")
    print(f"  Test Days:              {len(test_days)}")
    print(f"  Temporal Leakage:       None (chronological split)")
    
    print(f"\nTRAINING PERFORMANCE:")
    print(f"  Total Episodes:         {args.episodes}")
    print(f"  Average Reward:         {np.mean(rewards_history):.2f}")
    print(f"  Best Episode Reward:    {np.max(rewards_history):.2f}")
    print(f"  Worst Episode Reward:   {np.min(rewards_history):.2f}")
    print(f"  Average Return:         {metrics_df['Percent_Return'].mean():.2f}%")
    print(f"  Best Episode Return:    {metrics_df['Percent_Return'].max():.2f}%")
    print(f"  Worst Episode Return:   {metrics_df['Percent_Return'].min():.2f}%")
    print(f"  Average Trades/Episode: {metrics_df['Number_of_Trades'].mean():.1f}")
    print(f"  Final Epsilon:          {epsilon:.4f}")
    
    print(f"\nTEST SET PERFORMANCE:")
    print(f"  Test Episodes:          {final_eval_metrics['num_episodes']}")
    print(f"  Average Return:         {final_eval_metrics['avg_return']:.2f}%")
    print(f"  Win Rate:               {final_eval_metrics['win_rate']:.2f}%")
    print(f"  Best Return:            {final_eval_metrics['best_return']:.2f}%")
    print(f"  Worst Return:           {final_eval_metrics['worst_return']:.2f}%")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    train_agent()