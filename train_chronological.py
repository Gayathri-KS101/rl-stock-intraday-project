"""
Training Script with Fake PPO (21-bin actions + interpolation)
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

# Import custom modules
from simplified_environment import load_data, SimplifiedStockTradingEnv
from simplified_agent import (DQN, ReplayBuffer, create_episode_logger, log_step,
                               save_episode_log, print_log_preview, 
                               plot_episode_performance, analyze_episode_decisions)
from data_splitter import load_and_split_data, ChronologicalDaySplitter

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# === CHANGE 3: Simple interpolator for smooth transitions ===
class ActionInterpolator:
    """Simple action interpolator for smooth transitions between bins"""
    
    def __init__(self, num_bins=21, interpolation_steps=5):
        self.bin_values = np.linspace(-1.0, 1.0, num_bins)
        self.interpolation_steps = interpolation_steps
        self.current_value = 0.0
        self.target_bin = 0
        self.step_count = 0
        
    def start_interpolation(self, target_bin):
        """Start interpolating to new target bin"""
        self.target_bin = target_bin
        self.step_count = 0
        
    def get_next_value(self):
        """Get next interpolated value"""
        if self.step_count >= self.interpolation_steps:
            return self.bin_values[self.target_bin]
        
        # Linear interpolation
        alpha = self.step_count / self.interpolation_steps
        target_value = self.bin_values[self.target_bin]
        interpolated = self.current_value * (1 - alpha) + target_value * alpha
        
        self.step_count += 1
        return interpolated
    
    def update_current(self, value):
        """Update current value"""
        self.current_value = value


def train_agent():
    """Main training function with Fake PPO"""
    
    parser = argparse.ArgumentParser(description='Train Fake PPO agent.')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--log-episodes', type=int, default=10)
    parser.add_argument('--data-dir', type=str, default='processed_data')
    parser.add_argument('--initial-balance', type=float, default=10000)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--min-minutes', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--epsilon-min', type=float, default=0.05)
    parser.add_argument('--eval-frequency', type=int, default=50)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--decision-interval', type=int, default=5)
    args = parser.parse_args()

    # Load and split data
    print("="*100)
    print("LOADING AND SPLITTING DATA")
    print("="*100)
    
    all_files = glob.glob(f'{args.data_dir}/*.csv')
    train_days, test_days, splitter = load_and_split_data(
        all_files, args.train_ratio, args.min_minutes
    )

    # Initialize agent
    print("="*100)
    print("INITIALIZING FAKE PPO AGENT")
    print("="*100)
    
    dummy_day_data, dummy_stock_name = train_days[0]
    dummy_env = SimplifiedStockTradingEnv(
        dummy_day_data, stock_name=dummy_stock_name, initial_balance=args.initial_balance
    )
    input_dim = dummy_env.observation_space.shape[0]
    output_dim = dummy_env.action_space.n  # This will be 21
    
    print(f"Input Dimension:    {input_dim}")
    print(f"Output Dimension:   {output_dim} (21 bins from -1.0 to +1.0)")
    print(f"Action Range:       -1.0 (short) to +1.0 (long)")
    print(f"Initial Balance:    ${args.initial_balance:,.2f}")
    print(f"Commission Rate:    {dummy_env.commission*100:.2f}%")
    print(f"Interpolation:      {args.decision_interval} steps\n")
    
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(50000)
    
    epsilon = 0.3

    rewards_history = []
    
    episode_metrics = {
        'Episode': [],
        'Total_Reward': [],
        'Final_Net_Worth': [],
        'Percent_Return': [],
        'Number_of_Trades': []
    }

    # Create output directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("output_data", f"fake_ppo_run_{timestamp}")
    episode_csv_dir = os.path.join(base_dir, "episode_logs_csv")
    episode_png_dir = os.path.join(base_dir, "episode_logs_png")
    os.makedirs(episode_csv_dir, exist_ok=True)
    os.makedirs(episode_png_dir, exist_ok=True)

    # Training loop
    print("="*100)
    print("STARTING FAKE PPO TRAINING")
    print("="*100)
    
    for episode in range(args.episodes):
        
        # Sample a trading day
        day_data, stock_name = random.choice(train_days)
        env = SimplifiedStockTradingEnv(
            day_data, stock_name=stock_name, initial_balance=args.initial_balance
        )
        
        print(f"Episode {episode+1} | Trading Stock: {stock_name}")
        
        state, _ = env.reset()
        done = False
        
        should_log = (args.log_episodes > 0) and ((episode + 1) % args.log_episodes == 0)
        if should_log:
            episode_logger = create_episode_logger()
        
        step_count = 0
        total_reward = 0
        
        # Initialize interpolator for smooth actions
        interpolator = ActionInterpolator(interpolation_steps=args.decision_interval)
        
        while not done:
            
            # Make new decision at interval boundaries
            if step_count % args.decision_interval == 0:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_t).cpu().numpy()[0]

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    target_bin = random.randint(0, 20)  # Random bin
                    was_random = True
                else:
                    target_bin = np.argmax(q_values)    # Best bin
                    was_random = False

                # Start interpolation to new target
                interpolator.start_interpolation(target_bin)
                
                # Get first interpolated value
                action_value = interpolator.get_next_value()
                
                # Find nearest bin for environment
                action = np.argmin(np.abs(env.action_bins - action_value))
                
                # Store for logging
                decision_bin = target_bin
            else:
                # Use interpolated value
                action_value = interpolator.get_next_value()
                action = np.argmin(np.abs(env.action_bins - action_value))
                was_random = False
                
                # Recompute q_values for logging
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_t).cpu().numpy()[0]
            
            # Execute action
            next_state, reward, done, _, _ = env.step(action)
            
            # Update interpolator's current value
            interpolator.update_current(action_value)
            
            # Log if needed
            if should_log:
                # Use decision_bin if it exists (from decision step), otherwise use action
                log_action = decision_bin if 'decision_bin' in locals() else action
                log_step(episode_logger, episode + 1, step_count, env, state,
                        log_action, q_values, epsilon, reward, was_random)

            # Store in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Training step
            if len(replay_buffer) > 2000:
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
        
        # Decay epsilon
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Calculate metrics
        percent_return = ((env.net_worth - env.initial_balance) / env.initial_balance) * 100
        
        episode_metrics['Episode'].append(episode + 1)
        episode_metrics['Total_Reward'].append(total_reward)
        episode_metrics['Final_Net_Worth'].append(env.net_worth)
        episode_metrics['Percent_Return'].append(percent_return)
        episode_metrics['Number_of_Trades'].append(env.num_trades)

        # Log detailed episode if requested
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

        print(f"Episode {episode+1}: Reward={total_reward:.2f} | "
              f"Net Worth=${env.net_worth:.2f} | "
              f"Return={percent_return:.2f}% | "
              f"Trades={env.num_trades} | "
              f"Epsilon={epsilon:.2f}")

    # Save model
    torch.save(policy_net.state_dict(), os.path.join(base_dir, 'policy_net.pth'))
    
    print("\n" + "="*100)
    print("TRAINING COMPLETE")
    print("="*100)
    print(f"Model saved to: {base_dir}")


if __name__ == "__main__":
    train_agent()