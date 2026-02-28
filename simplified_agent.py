"""
Simplified DQN Agent and Utilities Module with 21-bin logging
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from collections import deque
import random


class DQN(nn.Module):
    """Deep Q-Network for trading decisions with 21 outputs."""
    
    def __init__(self, input_dim, output_dim=21):  # Changed to 21
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# UPDATED LOGGING FOR 21-BIN ACTIONS
# ============================================================================

def create_episode_logger():
    """Initialize logger for 21-bin actions."""
    return {
        'Episode': [],
        'Step': [],
        'DateTime': [],
        'Current_Price': [],
        'Action_Bin': [],        # 0-20
        'Action_Value': [],       # -1.0 to +1.0
        'Q_Values_Preview': [],   # First few Q-values
        'Chosen_Q_Value': [],
        'Decision_Reason': [],
        'Epsilon': [],
        'Shares_Held': [],
        'Balance': [],
        'Net_Worth': [],
        'Reward': []
    }


def log_step(logger, episode, step, env, state, action, q_values, epsilon, reward, was_random):
    """
    Log a single step with 21-bin action.
    """
    current_row = env.df.iloc[env.current_step - 1]
    
    # Convert action to value
    action_value = env.action_bins[action]

    # Decision reason
    if was_random:
        decision_reason = f"Exploration (Random) -> Bin {action}"
    else:
        max_q_idx = np.argmax(q_values)
        decision_reason = f"Exploitation (Max Q: Bin {max_q_idx}) -> Bin {action}"

    # Format Q-values preview (first 5 values)
    q_preview = ', '.join([f"{q:.4f}" for q in q_values[:5]]) + '...'

    logger['Episode'].append(episode)
    logger['Step'].append(step)
    logger['DateTime'].append(current_row['date'])
    logger['Current_Price'].append(current_row['close'])
    logger['Action_Bin'].append(action)
    logger['Action_Value'].append(action_value)
    logger['Q_Values_Preview'].append(q_preview)
    logger['Chosen_Q_Value'].append(q_values[action])
    logger['Decision_Reason'].append(decision_reason)
    logger['Epsilon'].append(epsilon)
    logger['Shares_Held'].append(env.shares_held)
    logger['Balance'].append(env.balance)
    logger['Net_Worth'].append(env.net_worth)
    logger['Reward'].append(reward)


def save_episode_log(logger, episode, output_dir='episode_logs'):
    """Save episode log to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(logger)
    filepath = f"{output_dir}/episode_{episode}_debug_log.csv"
    df.to_csv(filepath, index=False)
    return df, filepath


def print_log_preview(df, num_rows=10):
    """Print preview of episode log."""
    print("\n" + "="*100)
    print("EPISODE LOG PREVIEW (First {} rows)".format(min(num_rows, len(df))))
    print("="*100)

    preview_cols = ['Step', 'DateTime', 'Action_Value', 'Current_Price',
                    'Shares_Held', 'Net_Worth', 'Reward']

    preview_df = df[preview_cols].head(num_rows).copy()

    preview_df['Current_Price'] = preview_df['Current_Price'].apply(lambda x: f"${x:.2f}")
    preview_df['Action_Value'] = preview_df['Action_Value'].apply(lambda x: f"{x:.3f}")
    preview_df['Net_Worth'] = preview_df['Net_Worth'].apply(lambda x: f"${x:.2f}")
    preview_df['Reward'] = preview_df['Reward'].apply(lambda x: f"{x:.6f}")

    print(preview_df.to_string(index=False))
    print("="*100 + "\n")


def plot_episode_performance(df, episode, output_dir='episode_logs'):
    """Plot episode performance with position coloring."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    steps = df['Step'].values
    net_worth = df['Net_Worth'].values
    actions = df['Action_Value'].values

    # Plot 1: Net Worth with position color
    scatter = ax1.scatter(steps, net_worth, c=actions, cmap='RdYlGn', 
                          s=30, alpha=0.7, vmin=-1, vmax=1)
    ax1.plot(steps, net_worth, 'b-', alpha=0.3, linewidth=1)

    initial_balance = df['Net_Worth'].iloc[0]
    ax1.axhline(y=initial_balance, color='gray', linestyle='--',
                linewidth=1, label=f'Initial Balance (${initial_balance:.2f})')

    ax1.set_ylabel('Net Worth ($)', fontsize=12)
    ax1.set_title(f'Episode {episode} - Fake PPO Performance', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Price
    price = df['Current_Price'].values
    ax2.plot(steps, price, 'k-', linewidth=1.5, label='Price')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Price ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax2, orientation='horizontal', pad=0.2)
    cbar.set_label('Position (-1 Short to +1 Long)', fontsize=10)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/episode_{episode}_performance.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def analyze_episode_decisions(df):
    """Analyze episode decisions for 21-bin actions."""
    print("\n" + "="*100)
    print("EPISODE DECISION ANALYSIS")
    print("="*100)

    # Position Distribution
    print("\n1. POSITION DISTRIBUTION:")
    print("-" * 50)
    
    short_pct = (df['Action_Value'] < -0.1).mean() * 100
    cash_pct = ((df['Action_Value'] >= -0.1) & (df['Action_Value'] <= 0.1)).mean() * 100
    long_pct = (df['Action_Value'] > 0.1).mean() * 100
    
    print(f"   Short (< -0.1):   {short_pct:6.2f}%")
    print(f"   Cash (-0.1 to 0.1): {cash_pct:6.2f}%")
    print(f"   Long (> 0.1):      {long_pct:6.2f}%")
    print(f"   Avg Position:      {df['Action_Value'].mean():6.3f}")

    # Performance summary
    print("\n2. PERFORMANCE SUMMARY:")
    print("-" * 50)
    initial_nw = df['Net_Worth'].iloc[0]
    final_nw = df['Net_Worth'].iloc[-1]
    total_return = ((final_nw - initial_nw) / initial_nw) * 100
    total_reward = df['Reward'].sum()

    print(f"   Initial Net Worth: ${initial_nw:10.2f}")
    print(f"   Final Net Worth:   ${final_nw:10.2f}")
    print(f"   Total Return:      {total_return:9.2f}%")
    print(f"   Total Reward:      {total_reward:10.6f}")

    print("="*100 + "\n")