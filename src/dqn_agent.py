"""
DQN Agent and Analysis Utilities Module

This module contains:
- DQN neural network architecture
- Replay buffer implementation
- Episode logging and analysis functions
- Visualization utilities
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from collections import deque
import random


# --- DQN Network ---
class DQN(nn.Module):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)  # Increased capacity
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
# INTERPRETABILITY & VISUALIZATION HELPERS
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
        'Q_Hold': [],
        'Q_Buy': [],
        'Q_Sell': [],
        'Chosen_Q_Value': [],
        'Decision_Reason': [],
        'Epsilon': [],
        'Shares_Held': [],
        'Balance': [],
        'Net_Worth': [],
        'Reward': [],
        'RSI': [],
        'MACD': [],
        'BB_Pct': [],
        'Trend_5d': [],
        'Trend_20d': [],
        'Change_From_Prev_Close': []
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
        action: Action taken (0=Hold, 1=Buy, 2=Sell)
        q_values: Numpy array of Q-values [Q_Hold, Q_Buy, Q_Sell]
        epsilon: Current epsilon value
        reward: Reward received
        was_random: Boolean indicating if action was random exploration
    """
    # Get current row from environment
    current_row = env.df.iloc[env.current_step - 1]
    
    # Action mapping
    action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    
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
    logger['Q_Hold'].append(q_values[0])
    logger['Q_Buy'].append(q_values[1])
    logger['Q_Sell'].append(q_values[2])
    logger['Chosen_Q_Value'].append(q_values[action])
    logger['Decision_Reason'].append(decision_reason)
    logger['Epsilon'].append(epsilon)
    logger['Shares_Held'].append(env.shares_held)
    logger['Balance'].append(env.balance)
    logger['Net_Worth'].append(env.net_worth)
    logger['Reward'].append(reward)
    logger['RSI'].append(current_row['rsi'])
    logger['MACD'].append(current_row['macd'])
    logger['BB_Pct'].append(current_row['bb_pct'])
    logger['Trend_5d'].append(current_row['trend_5d'])
    logger['Trend_20d'].append(current_row['trend_20d'])
    logger['Change_From_Prev_Close'].append(current_row['change_from_prev_close'])


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
                    'Q_Hold', 'Q_Buy', 'Q_Sell', 'Decision_Reason',
                    'Shares_Held', 'Net_Worth', 'Reward']
    
    preview_df = df[preview_cols].head(num_rows).copy()
    
    # Format for better readability
    preview_df['Current_Price'] = preview_df['Current_Price'].apply(lambda x: f"${x:.2f}")
    preview_df['Q_Hold'] = preview_df['Q_Hold'].apply(lambda x: f"{x:.2f}")
    preview_df['Q_Buy'] = preview_df['Q_Buy'].apply(lambda x: f"{x:.2f}")
    preview_df['Q_Sell'] = preview_df['Q_Sell'].apply(lambda x: f"{x:.2f}")
    preview_df['Net_Worth'] = preview_df['Net_Worth'].apply(lambda x: f"${x:.2f}")
    preview_df['Reward'] = preview_df['Reward'].apply(lambda x: f"{x:.2f}")
    
    print(preview_df.to_string(index=False))
    print("="*100 + "\n")


def plot_episode_performance(df, episode, output_dir='episode_logs'):
    """
    Create visualization showing net worth curve with buy/sell markers.
    
    Args:
        df: Episode log DataFrame
        episode: Episode number
        output_dir: Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Net Worth with Buy/Sell markers
    steps = df['Step'].values
    net_worth = df['Net_Worth'].values
    
    ax1.plot(steps, net_worth, 'b-', linewidth=2, label='Net Worth')
    
    # Mark Buy actions (green upward arrows)
    buy_mask = df['Action'] == 1
    if buy_mask.any():
        buy_steps = df.loc[buy_mask, 'Step'].values
        buy_networth = df.loc[buy_mask, 'Net_Worth'].values
        ax1.scatter(buy_steps, buy_networth, color='green', marker='^', 
                   s=100, label='Buy', zorder=5, edgecolors='black', linewidths=1.5)
    
    # Mark Sell actions (red downward arrows)
    sell_mask = df['Action'] == 2
    if sell_mask.any():
        sell_steps = df.loc[sell_mask, 'Step'].values
        sell_networth = df.loc[sell_mask, 'Net_Worth'].values
        ax1.scatter(sell_steps, sell_networth, color='red', marker='v', 
                   s=100, label='Sell', zorder=5, edgecolors='black', linewidths=1.5)
    
    # Add initial balance line
    initial_balance = df['Net_Worth'].iloc[0]
    ax1.axhline(y=initial_balance, color='gray', linestyle='--', 
                linewidth=1, label=f'Initial Balance (${initial_balance:.2f})')
    
    ax1.set_ylabel('Net Worth ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Episode {episode} - Trading Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price with technical indicators
    price = df['Current_Price'].values
    ax2.plot(steps, price, 'k-', linewidth=1.5, label='Price')
    
    # Mark Buy/Sell on price chart too
    if buy_mask.any():
        buy_prices = df.loc[buy_mask, 'Current_Price'].values
        ax2.scatter(buy_steps, buy_prices, color='green', marker='^', 
                   s=80, alpha=0.7, zorder=5)
    
    if sell_mask.any():
        sell_prices = df.loc[sell_mask, 'Current_Price'].values
        ax2.scatter(sell_steps, sell_prices, color='red', marker='v', 
                   s=80, alpha=0.7, zorder=5)
    
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
    action_pcts = (action_counts / len(df) * 100).round(2)
    
    print("\n1. ACTION DISTRIBUTION:")
    print("-" * 50)
    for action in ['Hold', 'Buy', 'Sell']:
        if action in action_counts.index:
            print(f"   {action:6s}: {action_counts[action]:4d} ({action_pcts[action]:5.2f}%)")
        else:
            print(f"   {action:6s}: {0:4d} ({0:5.2f}%)")
    
    # 2. Average Q-values per action
    print("\n2. AVERAGE Q-VALUES PER ACTION TAKEN:")
    print("-" * 50)
    for action_name, action_code in [('Hold', 0), ('Buy', 1), ('Sell', 2)]:
        action_df = df[df['Action'] == action_code]
        if len(action_df) > 0:
            avg_q_hold = action_df['Q_Hold'].mean()
            avg_q_buy = action_df['Q_Buy'].mean()
            avg_q_sell = action_df['Q_Sell'].mean()
            print(f"   When {action_name:4s}: Q_Hold={avg_q_hold:7.2f}, Q_Buy={avg_q_buy:7.2f}, Q_Sell={avg_q_sell:7.2f}")
        else:
            print(f"   When {action_name:4s}: No actions taken")
    
    # 3. Average reward per action
    print("\n3. AVERAGE REWARD PER ACTION:")
    print("-" * 50)
    for action_name in ['Hold', 'Buy', 'Sell']:
        action_df = df[df['Action_Name'] == action_name]
        if len(action_df) > 0:
            avg_reward = action_df['Reward'].mean()
            total_reward = action_df['Reward'].sum()
            print(f"   {action_name:6s}: Avg={avg_reward:8.2f}, Total={total_reward:10.2f}")
        else:
            print(f"   {action_name:6s}: No actions taken")
    
    # 4. Exploration vs Exploitation
    print("\n4. EXPLORATION vs EXPLOITATION:")
    print("-" * 50)
    exploration_count = df['Decision_Reason'].str.contains('Exploration').sum()
    exploitation_count = df['Decision_Reason'].str.contains('Exploitation').sum()
    total = len(df)
    print(f"   Exploration (Random): {exploration_count:4d} ({exploration_count/total*100:5.2f}%)")
    print(f"   Exploitation (Max Q): {exploitation_count:4d} ({exploitation_count/total*100:5.2f}%)")
    
    # 5. Correlation: RSI and Buy decisions
    print("\n5. TECHNICAL INDICATOR CORRELATIONS:")
    print("-" * 50)
    
    # Binary indicators for actions
    df_analysis = df.copy()
    df_analysis['is_buy'] = (df_analysis['Action'] == 1).astype(int)
    df_analysis['is_sell'] = (df_analysis['Action'] == 2).astype(int)
    
    # RSI correlation with Buy
    rsi_buy_corr = df_analysis[['RSI', 'is_buy']].corr().iloc[0, 1]
    print(f"   RSI vs Buy decisions:      {rsi_buy_corr:7.4f}")
    
    # Trend_20d correlation with Sell
    trend20_sell_corr = df_analysis[['Trend_20d', 'is_sell']].corr().iloc[0, 1]
    print(f"   Trend_20d vs Sell decisions: {trend20_sell_corr:7.4f}")
    
    # Additional correlations
    macd_buy_corr = df_analysis[['MACD', 'is_buy']].corr().iloc[0, 1]
    print(f"   MACD vs Buy decisions:     {macd_buy_corr:7.4f}")
    
    bb_buy_corr = df_analysis[['BB_Pct', 'is_buy']].corr().iloc[0, 1]
    print(f"   BB_Pct vs Buy decisions:   {bb_buy_corr:7.4f}")
    
    # 6. Q-value statistics
    print("\n6. Q-VALUE STATISTICS (Overall):")
    print("-" * 50)
    print(f"   Q_Hold - Mean: {df['Q_Hold'].mean():7.2f}, Std: {df['Q_Hold'].std():7.2f}")
    print(f"   Q_Buy  - Mean: {df['Q_Buy'].mean():7.2f}, Std: {df['Q_Buy'].std():7.2f}")
    print(f"   Q_Sell - Mean: {df['Q_Sell'].mean():7.2f}, Std: {df['Q_Sell'].std():7.2f}")
    
    # 7. Performance summary
    print("\n7. PERFORMANCE SUMMARY:")
    print("-" * 50)
    initial_nw = df['Net_Worth'].iloc[0]
    final_nw = df['Net_Worth'].iloc[-1]
    total_return = ((final_nw - initial_nw) / initial_nw) * 100
    total_reward = df['Reward'].sum()
    
    print(f"   Initial Net Worth: ${initial_nw:10.2f}")
    print(f"   Final Net Worth:   ${final_nw:10.2f}")
    print(f"   Total Return:      {total_return:9.2f}%")
    print(f"   Total Reward:      {total_reward:10.2f}")
    
    print("="*100 + "\n")