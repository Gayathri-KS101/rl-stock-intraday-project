"""
Professional Training Script for Maximum Profit Trading
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
from simplified_environment import load_data, ProfessionalTradingEnv
from simplified_agent import (DQN, ReplayBuffer, create_episode_logger, log_step,
                               save_episode_log, print_log_preview, 
                               plot_episode_performance, analyze_episode_decisions)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def validate_agent(policy_net, val_days, initial_balance, window_size, detailed=False):
    """Validate agent on validation days."""
    policy_net.eval()
    returns = []
    
    for day_data in val_days[:20]:  # Limit validation days
        try:
            env = ProfessionalTradingEnv(day_data, initial_balance, window_size)
            state, _ = env.reset()
            done = False
            
            while not done:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_t).argmax().item()
                state, _, done, _, _ = env.step(action)
            
            metrics = env.get_metrics()
            returns.append(metrics['total_return'])
            
            if detailed:
                print(f"  Day return: {metrics['total_return']:.2f}% | Trades: {metrics['num_trades']}")
        except Exception as e:
            print(f"Error in validation: {e}")
            continue
    
    policy_net.train()
    return np.mean(returns) if returns else 0


def plot_training_results(metrics_df, output_dir):
    """Plot training results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns
        axes[0,0].plot(metrics_df['Episode'], metrics_df['Train_Return'], 'b-', label='Train')
        axes[0,0].plot(metrics_df['Episode'], metrics_df['Val_Return'], 'r-', label='Validation')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Return (%)')
        axes[0,0].set_title('Training Progress')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Trades
        axes[0,1].plot(metrics_df['Episode'], metrics_df['Trades'], 'g-')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Number of Trades')
        axes[0,1].set_title('Trading Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # Allocation
        axes[1,0].plot(metrics_df['Episode'], metrics_df['Avg_Alloc'], 'purple')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Avg Allocation (%)')
        axes[1,0].set_title('Average Position Size')
        axes[1,0].grid(True, alpha=0.3)
        
        # Win Rate
        axes[1,1].plot(metrics_df['Episode'], metrics_df['Win_Rate'], 'orange')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Win Rate (%)')
        axes[1,1].set_title('Trading Success Rate')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting results: {e}")


def train_agent():
    """Professional training function optimized for profit."""
    
    parser = argparse.ArgumentParser(description='Professional Trading with Position Sizing')
    parser.add_argument('--episodes', type=int, default=500, help='Training episodes')
    parser.add_argument('--log-episodes', type=int, default=50, help='Log every N episodes')
    parser.add_argument('--data-dir', type=str, default='processed_data', help='Data directory')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-decay', type=float, default=0.997, help='Epsilon decay')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--window-size', type=int, default=20, help='Window size')
    parser.add_argument('--target-update', type=int, default=10, help='Target network update freq')
    parser.add_argument('--replay-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    
    args = parser.parse_args()
    
    # ========== DATA LOADING ==========
    print("="*100)
    print("LOADING DATA")
    print("="*100)
    
    all_files = glob.glob(f'{args.data_dir}/*.csv')
    print(f"Found {len(all_files)} files")
    
    all_days = []
    for file_path in all_files:
        df = load_data(file_path)
        if df is not None and len(df) > 60:
            all_days.append(df)
    
    print(f"Loaded {len(all_days)} trading days")
    
    if len(all_days) == 0:
        print("ERROR: No valid trading days found!")
        return
    
    # Split into train/val (80/20)
    split_idx = int(len(all_days) * 0.8)
    train_days = all_days[:split_idx]
    val_days = all_days[split_idx:]
    
    print(f"Training days: {len(train_days)}, Validation days: {len(val_days)}")
    
    # ========== INITIALIZATION ==========
    print("\n" + "="*100)
    print("INITIALIZING AGENT")
    print("="*100)
    
    # Create dummy environment and get actual state dimension
    dummy_env = ProfessionalTradingEnv(train_days[0], args.initial_balance, args.window_size)
    state, _ = dummy_env.reset()
    input_dim = len(state)  # Should be 32
    output_dim = dummy_env.action_space.n
    
    print(f"State dimension: {input_dim}")
    print(f"Action dimension: {output_dim} (0%, 25%, 50%, 75%, 100%)")
    
    # Networks
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    replay_buffer = ReplayBuffer(args.replay_size)
    
    # Training tracking
    epsilon = 1.0
    best_val_return = -float('inf')
    patience = 50
    patience_counter = 0
    
    # Metrics
    train_rewards = []
    val_returns = []
    episode_metrics = {
        'Episode': [], 'Train_Return': [], 'Val_Return': [],
        'Trades': [], 'Avg_Alloc': [], 'Win_Rate': []
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("output_data", f"run_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'plots'), exist_ok=True)
    
    print(f"\nOutput directory: {base_dir}")
    
    # ========== TRAINING LOOP ==========
    print("\n" + "="*100)
    print("STARTING PROFESSIONAL TRAINING")
    print("="*100)
    
    for episode in range(args.episodes):
        try:
            # === TRAINING EPISODE ===
            day_data = random.choice(train_days)
            env = ProfessionalTradingEnv(day_data, args.initial_balance, args.window_size)
            
            state, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            should_log = (episode + 1) % args.log_episodes == 0
            if should_log:
                logger = create_episode_logger()
            
            while not done:
                # Epsilon-greedy action
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
                        was_random = False
                
                next_state, reward, done, _, _ = env.step(action)
                
                if should_log:
                    log_step(logger, episode+1, step, env, state, action, 
                            q_values, epsilon, reward, was_random)
                
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                step += 1
                
                # Training step
                if len(replay_buffer) > args.batch_size:
                    batch = replay_buffer.sample(args.batch_size)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
                    
                    batch_state = torch.FloatTensor(np.array(batch_state))
                    batch_action = torch.LongTensor(batch_action).unsqueeze(1)
                    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                    batch_next_state = torch.FloatTensor(np.array(batch_next_state))
                    batch_done = torch.FloatTensor(batch_done).unsqueeze(1)
                    
                    # Double DQN update
                    current_q = policy_net(batch_state).gather(1, batch_action)
                    
                    with torch.no_grad():
                        next_actions = policy_net(batch_next_state).argmax(1).unsqueeze(1)
                        next_q = target_net(batch_next_state).gather(1, next_actions)
                        target_q = batch_reward + args.gamma * next_q * (1 - batch_done)
                    
                    loss = nn.MSELoss()(current_q, target_q)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip)
                    optimizer.step()
            
            # Episode complete - update tracking
            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
            scheduler.step()
            
            if episode % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            train_rewards.append(episode_reward)
            metrics = env.get_metrics()
            
            # === VALIDATION ===
            if (episode + 1) % 10 == 0:
                val_return = validate_agent(policy_net, val_days, args.initial_balance, args.window_size)
                val_returns.append(val_return)
                
                # Save best model
                if val_return > best_val_return:
                    best_val_return = val_return
                    torch.save(policy_net.state_dict(), os.path.join(base_dir, 'best_model.pth'))
                    patience_counter = 0
                    print(f"✓ New best model! Val return: {val_return:.2f}%")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter > patience:
                    print(f"\nEarly stopping at episode {episode+1}")
                    break
                
                # Log metrics
                episode_metrics['Episode'].append(episode+1)
                episode_metrics['Train_Return'].append(metrics['total_return'])
                episode_metrics['Val_Return'].append(val_return)
                episode_metrics['Trades'].append(metrics['num_trades'])
                episode_metrics['Avg_Alloc'].append(metrics['avg_allocation'] * 100)
                episode_metrics['Win_Rate'].append(metrics['win_rate'])
            
            # Progress update
            if (episode + 1) % 10 == 0:
                print(f"\nEpisode {episode+1}/{args.episodes}")
                print(f"  Train Return: {metrics['total_return']:.2f}% | Trades: {metrics['num_trades']}")
                print(f"  Win Rate: {metrics['win_rate']:.1f}% | Drawdown: {metrics['max_drawdown']:.1f}%")
                print(f"  Epsilon: {epsilon:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Detailed logging
            if should_log:
                log_df, log_path = save_episode_log(logger, episode+1, os.path.join(base_dir, 'logs'))
                plot_episode_performance(log_df, episode+1, os.path.join(base_dir, 'plots'))
                analyze_episode_decisions(log_df)
                
        except Exception as e:
            print(f"Error in episode {episode+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== FINAL EVALUATION ==========
    print("\n" + "="*100)
    print("TRAINING COMPLETE - FINAL EVALUATION")
    print("="*100)
    
    # Load best model
    best_model_path = os.path.join(base_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        policy_net.load_state_dict(torch.load(best_model_path))
    
    # Final validation
    final_return = validate_agent(policy_net, val_days, args.initial_balance, args.window_size, detailed=True)
    
    # Save results
    if episode_metrics['Episode']:
        metrics_df = pd.DataFrame(episode_metrics)
        metrics_df.to_csv(os.path.join(base_dir, 'metrics.csv'), index=False)
        plot_training_results(metrics_df, base_dir)
    
    # Save final model
    torch.save(policy_net.state_dict(), os.path.join(base_dir, 'final_model.pth'))
    
    print(f"\nBest validation return: {best_val_return:.2f}%")
    print(f"Final validation return: {final_return:.2f}%")
    print(f"All results saved in: {base_dir}")
    print("="*100)


if __name__ == "__main__":
    train_agent()