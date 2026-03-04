"""
Simplified DQN Agent and Utilities Module

This module contains:
- DQN neural network architecture
- Replay buffer implementation
- Episode logging and analysis functions
- Visualization utilities (simplified version)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from collections import deque
import random

#---A2C---
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value
    
# --- DQN Network ---
class DQN(nn.Module):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, input_dim, output_dim=2):  # output_dim=2 for Cash/Invest
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
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# LOGGING AND ANALYSIS UTILITIES
# ============================================================================

def create_episode_logger():
    """Initialize an empty logger for episode steps."""
    return {
        'Episode': [],
        'Step': [],
        'DateTime': [],
        'Current_Price': [],
        'Action': [],
        'Action_Name': [],
        'Q_Cash': [],
        'Q_Invest': [],
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
    Log a single step's information to the episode logger.
    
    Args:
        logger: Dictionary containing lists for each column
        episode: Current episode number
        step: Current step number
        env: Trading environment
        state: Current state (not used but available)
        action: Action taken (0=Cash, 1=Invest)
        q_values: Numpy array of Q-values [Q_Cash, Q_Invest]
        epsilon: Current epsilon value
        reward: Reward received
        was_random: Boolean indicating if action was random exploration
    """
    # Get current row from environment
    current_row = env.df.iloc[env.current_step - 1]

    # Action mapping
    action_names = {0: 'Cash', 1: 'Invest'}

    # Decision reason
    if was_random:
        decision_reason = "Exploration (Random)"
    else:
        max_q_idx = np.argmax(q_values)
        decision_reason = f"Exploitation (Max Q: {action_names[max_q_idx]})"

    # Append to logger
    logger['Episode'].append(episode)
    logger['Step'].append(step)
    logger['DateTime'].append(current_row['date'])
    logger['Current_Price'].append(current_row['close'])
    logger['Action'].append(action)
    logger['Action_Name'].append(action_names[action])
    logger['Q_Cash'].append(q_values[0])
    logger['Q_Invest'].append(q_values[1])
    logger['Chosen_Q_Value'].append(q_values[action])
    logger['Decision_Reason'].append(decision_reason)
    logger['Epsilon'].append(epsilon)
    logger['Shares_Held'].append(env.shares_held)
    logger['Balance'].append(env.balance)
    logger['Net_Worth'].append(env.net_worth)
    logger['Reward'].append(reward)


def save_episode_log(logger, episode, output_dir='episode_logs'):
    """Save episode log to CSV and return DataFrame."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(logger)
    filepath = f"{output_dir}/episode_{episode}_debug_log.csv"
    df.to_csv(filepath, index=False)
    return df, filepath


def print_log_preview(df, num_rows=10):
    """Print a formatted preview of the episode log."""
    print("\n" + "="*100)
    print("EPISODE LOG PREVIEW (First {} rows)".format(min(num_rows, len(df))))
    print("="*100)

    # Select key columns for preview
    preview_cols = ['Step', 'DateTime', 'Action_Name', 'Current_Price',
                    'Q_Cash', 'Q_Invest', 'Decision_Reason',
                    'Shares_Held', 'Net_Worth', 'Reward']

    preview_df = df[preview_cols].head(num_rows).copy()

    # Format for better readability
    preview_df['Current_Price'] = preview_df['Current_Price'].apply(lambda x: f"${x:.2f}")
    preview_df['Q_Cash']        = preview_df['Q_Cash'].apply(lambda x: f"{x:.4f}")
    preview_df['Q_Invest']      = preview_df['Q_Invest'].apply(lambda x: f"{x:.4f}")
    preview_df['Net_Worth']     = preview_df['Net_Worth'].apply(lambda x: f"${x:.2f}")
    preview_df['Reward']        = preview_df['Reward'].apply(lambda x: f"{x:.6f}")

    print(preview_df.to_string(index=False))
    print("="*100 + "\n")


def plot_episode_performance(df, episode, output_dir='episode_logs'):
    """
    Create visualization showing net worth curve with Cash/Invest markers.

    Args:
        df: Episode log DataFrame
        episode: Episode number
        output_dir: Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    steps     = df['Step'].values
    net_worth = df['Net_Worth'].values

    # --- Plot 1: Net Worth with Cash/Invest markers ---
    ax1.plot(steps, net_worth, 'b-', linewidth=2, label='Net Worth')

    # Mark Invest actions (green upward triangles)
    invest_mask = df['Action'] == 1
    if invest_mask.any():
        invest_steps    = df.loc[invest_mask, 'Step'].values
        invest_networth = df.loc[invest_mask, 'Net_Worth'].values
        ax1.scatter(invest_steps, invest_networth, color='green', marker='^',
                    s=100, label='Invest', zorder=5, edgecolors='black', linewidths=1.5)

    # Mark Cash actions (red downward triangles)
    cash_mask = df['Action'] == 0
    if cash_mask.any():
        cash_steps    = df.loc[cash_mask, 'Step'].values
        cash_networth = df.loc[cash_mask, 'Net_Worth'].values
        ax1.scatter(cash_steps, cash_networth, color='red', marker='v',
                    s=100, label='Cash', zorder=5, edgecolors='black', linewidths=1.5)

    # Initial balance reference line
    initial_balance = df['Net_Worth'].iloc[0]
    ax1.axhline(y=initial_balance, color='gray', linestyle='--',
                linewidth=1, label=f'Initial Balance (${initial_balance:.2f})')

    ax1.set_ylabel('Net Worth ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Episode {episode} - Trading Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Price with Cash/Invest markers ---
    price = df['Current_Price'].values
    ax2.plot(steps, price, 'k-', linewidth=1.5, label='Price')

    if invest_mask.any():
        invest_prices = df.loc[invest_mask, 'Current_Price'].values
        ax2.scatter(invest_steps, invest_prices, color='green', marker='^',
                    s=80, alpha=0.7, zorder=5, label='Invest')

    if cash_mask.any():
        cash_prices = df.loc[cash_mask, 'Current_Price'].values
        ax2.scatter(cash_steps, cash_prices, color='red', marker='v',
                    s=80, alpha=0.7, zorder=5, label='Cash')

    ax2.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Price Action', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/episode_{episode}_performance.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def analyze_episode_decisions(df):
    """
    Analyze and print decision statistics from episode log.

    Args:
        df: Episode log DataFrame
    """
    print("\n" + "="*100)
    print("EPISODE DECISION ANALYSIS")
    print("="*100)

    # 1. Action Distribution
    action_counts = df['Action_Name'].value_counts()
    action_pcts   = (action_counts / len(df) * 100).round(2)

    print("\n1. ACTION DISTRIBUTION:")
    print("-" * 50)
    for action in ['Cash', 'Invest']:
        count = action_counts.get(action, 0)
        pct   = action_pcts.get(action, 0.0)
        print(f"   {action:6s}: {count:4d} ({pct:5.2f}%)")

    # 2. Average Q-values per action taken
    print("\n2. AVERAGE Q-VALUES PER ACTION TAKEN:")
    print("-" * 50)
    for action_name, action_code in [('Cash', 0), ('Invest', 1)]:
        action_df = df[df['Action'] == action_code]
        if len(action_df) > 0:
            avg_q_cash   = action_df['Q_Cash'].mean()
            avg_q_invest = action_df['Q_Invest'].mean()
            print(f"   When {action_name:6s}: Q_Cash={avg_q_cash:7.4f}, Q_Invest={avg_q_invest:7.4f}")
        else:
            print(f"   When {action_name:6s}: No actions taken")

    # 3. Average reward per action
    print("\n3. AVERAGE REWARD PER ACTION:")
    print("-" * 50)
    for action_name in ['Cash', 'Invest']:
        action_df = df[df['Action_Name'] == action_name]
        if len(action_df) > 0:
            avg_reward   = action_df['Reward'].mean()
            total_reward = action_df['Reward'].sum()
            print(f"   {action_name:6s}: Avg={avg_reward:10.6f}, Total={total_reward:10.6f}")
        else:
            print(f"   {action_name:6s}: No actions taken")

    # 4. Exploration vs Exploitation
    print("\n4. EXPLORATION vs EXPLOITATION:")
    print("-" * 50)
    total             = len(df)
    exploration_count = df['Decision_Reason'].str.contains('Exploration').sum()
    exploitation_count = df['Decision_Reason'].str.contains('Exploitation').sum()
    print(f"   Exploration (Random): {exploration_count:4d} ({exploration_count/total*100:5.2f}%)")
    print(f"   Exploitation (Max Q): {exploitation_count:4d} ({exploitation_count/total*100:5.2f}%)")

    # 5. Q-value statistics
    print("\n5. Q-VALUE STATISTICS (Overall):")
    print("-" * 50)
    print(f"   Q_Cash   - Mean: {df['Q_Cash'].mean():7.4f}, Std: {df['Q_Cash'].std():7.4f}")
    print(f"   Q_Invest - Mean: {df['Q_Invest'].mean():7.4f}, Std: {df['Q_Invest'].std():7.4f}")

    # 6. Position switching frequency
    print("\n6. POSITION SWITCHING:")
    print("-" * 50)
    switches = (df['Action'] != df['Action'].shift()).sum() - 1  # subtract 1 for the first step
    print(f"   Total switches (Cash ↔ Invest): {max(switches, 0)}")

    # 7. Performance summary
    print("\n7. PERFORMANCE SUMMARY:")
    print("-" * 50)
    initial_nw   = df['Net_Worth'].iloc[0]
    final_nw     = df['Net_Worth'].iloc[-1]
    total_return = ((final_nw - initial_nw) / initial_nw) * 100
    total_reward = df['Reward'].sum()

    print(f"   Initial Net Worth: ${initial_nw:10.2f}")
    print(f"   Final Net Worth:   ${final_nw:10.2f}")
    print(f"   Total Return:      {total_return:9.2f}%")
    print(f"   Total Reward:      {total_reward:10.6f}")

    print("="*100 + "\n")