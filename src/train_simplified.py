"""
Main Training Script for Simplified Stock Trading Agent

This script orchestrates the training process with a simplified environment:
- Loads and prepares data
- Initializes DQN agent
- Runs training episodes
- Logs and analyzes performance
- Tracks episode metrics
"""

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


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def train_agent():
    """Main training function for the simplified DQN trading agent."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train simplified stock trading agent.')
    parser.add_argument('--episodes', type=int, default=200, 
                        help='Number of training episodes')
    parser.add_argument('--log-episodes', type=int, default=10, 
                        help='Save detailed logs every N episodes (0 to disable)')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                        help='Directory containing CSV data files')
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial trading balance')
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
    args = parser.parse_args()

    # ========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ========================================================================
    print("="*100)
    print("LOADING DATA")
    print("="*100)
    
    all_files = glob.glob(f'{args.data_dir}/*.csv')
    print(f"Found {len(all_files)} files in {args.data_dir}/")
    
    all_daily_groups = []
    
    for file_path in all_files:
        df = load_data(file_path)
        if df is None:
            continue
        
        # Group by Day
        df['day'] = df['date'].dt.date
        # Only take days with enough data (e.g. at least 60 minutes)
        daily_groups = [group for _, group in df.groupby('day') if len(group) > 60]
        all_daily_groups.extend(daily_groups)
    
    if not all_daily_groups:
        print("No valid trading days found.")
        return
        
    print(f"Loaded {len(all_daily_groups)} trading days total.\n")
    
    # ========================================================================
    # STEP 2: INITIALIZE AGENT
    # ========================================================================
    print("="*100)
    print("INITIALIZING SIMPLIFIED AGENT")
    print("="*100)
    
    # Create dummy environment to get observation/action space dimensions
    dummy_env = SimplifiedStockTradingEnv(all_daily_groups[0], 
                                           initial_balance=args.initial_balance)
    input_dim = dummy_env.observation_space.shape[0]
    output_dim = dummy_env.action_space.n
    
    print(f"Input Dimension:  {input_dim} (price window + shares + balance)")
    print(f"Output Dimension: {output_dim} (Hold, Buy 1, Sell 1)")
    print(f"Initial Balance:  ${args.initial_balance:,.2f}")
    print(f"Commission Rate:  {dummy_env.commission*100:.2f}%\n")
    
    # Create policy and target networks
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Optimizer and replay buffer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(50000)
    
    # Training hyperparameters
    epsilon = 1.0
    rewards_history = []
    
    # Episode metrics logger
    episode_metrics = {
        'Episode': [],
        'Total_Reward': [],
        'Final_Net_Worth': [],
        'Percent_Return': [],
        'Number_of_Trades': []
    }
    
    print(f"Hyperparameters:")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Learning Rate:  {args.learning_rate}")
    print(f"  Gamma:          {args.gamma}")
    print(f"  Epsilon Decay:  {args.epsilon_decay}")
    print(f"  Epsilon Min:    {args.epsilon_min}")
    print(f"  Buffer Size:    50000\n")
    
    # ========================================================================
    # STEP 3: TRAINING LOOP
    # ========================================================================
    print("="*100)
    print("STARTING SIMPLIFIED INTRADAY TRAINING")
    print("="*100)
    
    for episode in range(args.episodes):
        # Pick a random trading day
        day_data = random.choice(all_daily_groups)
        env = SimplifiedStockTradingEnv(day_data, initial_balance=args.initial_balance)
        
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # Initialize episode logger (only if we're logging this episode)
        should_log = (args.log_episodes > 0) and ((episode + 1) % args.log_episodes == 0)
        if should_log:
            episode_logger = create_episode_logger()
        
        step_count = 0
        
        # Episode loop
        while not done:
            # Epsilon-greedy action selection
            was_random = False
            if random.random() < epsilon:
                action = env.action_space.sample()
                was_random = True
                # Get Q-values even for random action (for logging)
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_t).cpu().numpy()[0]
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()
                    q_values = q_values.cpu().numpy()[0]
            
            # Take action in environment
            next_state, reward, done, _, _ = env.step(action)
            
            # Log step (only if logging this episode)
            if should_log:
                log_step(episode_logger, episode + 1, step_count, env, state, 
                        action, q_values, epsilon, reward, was_random)
            
            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Train the network
            if len(replay_buffer) > 2000:  # Wait for buffer to fill a bit
                # Sample batch from replay buffer
                transitions = replay_buffer.sample(args.batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                
                # Convert to tensors
                batch_state = torch.FloatTensor(np.array(batch_state))
                batch_action = torch.LongTensor(batch_action).unsqueeze(1)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                batch_next_state = torch.FloatTensor(np.array(batch_next_state))
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1)
                
                # Compute current Q-values
                curr_q = policy_net(batch_state).gather(1, batch_action)
                
                # Compute target Q-values
                next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                expected_q = batch_reward + args.gamma * next_q * (1 - batch_done)
                
                # Compute loss and update
                loss = nn.MSELoss()(curr_q, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        # Post-episode updates
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        
        # Update target network periodically
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Calculate episode metrics
        percent_return = ((env.net_worth - env.initial_balance) / env.initial_balance) * 100
        
        # Log episode metrics
        episode_metrics['Episode'].append(episode + 1)
        episode_metrics['Total_Reward'].append(total_reward)
        episode_metrics['Final_Net_Worth'].append(env.net_worth)
        episode_metrics['Percent_Return'].append(percent_return)
        episode_metrics['Number_of_Trades'].append(env.num_trades)
        
        # ====================================================================
        # EPISODE LOGGING AND ANALYSIS
        # ====================================================================
        if should_log:
            log_df, log_path = save_episode_log(episode_logger, episode + 1)
            
            print(f"\n{'='*100}")
            print(f"Episode {episode+1} Complete - Detailed Analysis")
            print(f"{'='*100}")
            print(f"Saved detailed log to: {log_path}")
            
            # Print preview
            print_log_preview(log_df, num_rows=10)
            
            # Generate and save plot
            plot_path = plot_episode_performance(log_df, episode + 1)
            print(f"Saved performance plot to: {plot_path}")
            
            # Analyze decisions
            analyze_episode_decisions(log_df)
        
        # Record episode reward
        rewards_history.append(total_reward)
        
        # Print episode summary
        print(f"Episode {episode+1}: "
              f"Reward: {total_reward:.2f} | "
              f"Net Worth: ${env.net_worth:.2f} | "
              f"Return: {percent_return:.2f}% | "
              f"Trades: {env.num_trades} | "
              f"Epsilon: {epsilon:.2f}")
        
        # Generate intermediate training plot every 50 episodes
        if (episode + 1) % 50 == 0:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards_history, linewidth=2)
            plt.title(f"Training Rewards (Episodes 1-{episode+1})", 
                     fontsize=14, fontweight='bold')
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Total Reward", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(f'training_rewards_ep{episode+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved intermediate plot: training_rewards_ep{episode+1}.png")

    # ========================================================================
    # STEP 4: SAVE OVERALL TRAINING RESULTS
    # ========================================================================
    print("\n" + "="*100)
    print("TRAINING COMPLETE")
    print("="*100)
    
    # Save episode metrics to CSV
    metrics_df = pd.DataFrame(episode_metrics)
    metrics_df.to_csv('episode_metrics.csv', index=False)
    print(f"\nSaved episode metrics to: episode_metrics.csv")
    
    # Generate final training plot
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, linewidth=2)
    plt.title("Training Rewards Over All Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('training_rewards_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Final training plot saved to: training_rewards_final.png")
    
    # Save model
    torch.save(policy_net.state_dict(), 'trained_model_simplified.pth')
    print(f"Model saved to: trained_model_simplified.pth")
    
    # ========================================================================
    # PRINT SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    print(f"Total Episodes:         {args.episodes}")
    print(f"Average Reward:         {np.mean(rewards_history):.2f}")
    print(f"Best Episode Reward:    {np.max(rewards_history):.2f}")
    print(f"Worst Episode Reward:   {np.min(rewards_history):.2f}")
    print(f"Average Return:         {metrics_df['Percent_Return'].mean():.2f}%")
    print(f"Best Episode Return:    {metrics_df['Percent_Return'].max():.2f}%")
    print(f"Worst Episode Return:   {metrics_df['Percent_Return'].min():.2f}%")
    print(f"Average Trades/Episode: {metrics_df['Number_of_Trades'].mean():.1f}")
    print(f"Final Epsilon:          {epsilon:.4f}")
    print("="*100 + "\n")


if __name__ == "__main__":
    train_agent()