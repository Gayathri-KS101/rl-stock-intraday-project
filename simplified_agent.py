"""
Simplified DQN Agent and Utilities Module
Enhanced version with professional DQN architecture for maximum profit trading
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from collections import deque
import random


# ============================================================================
# ENHANCED DQN NETWORK WITH DROPOUT FOR BETTER GENERALIZATION
# ============================================================================

class DQN(nn.Module):
    """
    Enhanced Deep Q-Network for trading decisions with dropout for regularization.
    """
    
    def __init__(self, input_dim, output_dim=5, dropout_rate=0.2):
        """
        Args:
            input_dim: Dimension of state space (should be 32)
            output_dim: Number of actions (default 5 for position sizing)
            dropout_rate: Dropout probability for regularization
        """
        super(DQN, self).__init__()
        
        # Remove BatchNorm for stability with single samples
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Q-values for each action
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
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
    """Initialize an empty logger for episode steps with enhanced tracking."""
    return {
        'Episode': [],
        'Step': [],
        'DateTime': [],
        'Current_Price': [],
        'Action': [],
        'Action_Name': [],
        'Q_0': [],  # Q-value for 0% allocation
        'Q_25': [],  # Q-value for 25% allocation
        'Q_50': [],  # Q-value for 50% allocation
        'Q_75': [],  # Q-value for 75% allocation
        'Q_100': [],  # Q-value for 100% allocation
        'Chosen_Q_Value': [],
        'Decision_Reason': [],
        'Epsilon': [],
        'Current_Allocation': [],
        'Balance': [],
        'Portfolio_Value': [],
        'Reward': [],
        'Cumulative_Return': [],
        'Drawdown': []
    }


def log_step(logger, episode, step, env, state, action, q_values, epsilon, reward, was_random):
    """
    Log a single step's information to the episode logger with enhanced metrics.
    """
    # Get current row from environment
    try:
        current_row = env.df.iloc[env.current_step - 1]
        current_price = current_row['close']
        current_date = current_row['date']
    except:
        current_price = 0
        current_date = None

    # Action mapping for position sizing
    action_names = {
        0: '0% Allocation', 
        1: '25% Allocation', 
        2: '50% Allocation', 
        3: '75% Allocation', 
        4: '100% Allocation'
    }

    # Decision reason
    if was_random:
        decision_reason = "Exploration (Random)"
    else:
        max_q_idx = np.argmax(q_values)
        decision_reason = f"Exploitation (Max Q: {action_names[max_q_idx]})"

    # Calculate cumulative return
    cumulative_return = ((env.portfolio_value / env.initial_balance) - 1) * 100
    
    # Calculate current drawdown
    if hasattr(env, 'peak_value'):
        current_drawdown = max(0, (env.peak_value - env.portfolio_value) / env.peak_value * 100)
    else:
        current_drawdown = 0

    # Append to logger
    logger['Episode'].append(episode)
    logger['Step'].append(step)
    logger['DateTime'].append(current_date)
    logger['Current_Price'].append(current_price)
    logger['Action'].append(action)
    logger['Action_Name'].append(action_names[action])
    
    # Handle q_values (ensure we have 5 values)
    if len(q_values) >= 5:
        logger['Q_0'].append(q_values[0])
        logger['Q_25'].append(q_values[1])
        logger['Q_50'].append(q_values[2])
        logger['Q_75'].append(q_values[3])
        logger['Q_100'].append(q_values[4])
    else:
        # Pad with zeros if we have fewer q_values
        q_values_padded = np.pad(q_values, (0, 5 - len(q_values)), 'constant', constant_values=0)
        logger['Q_0'].append(q_values_padded[0])
        logger['Q_25'].append(q_values_padded[1])
        logger['Q_50'].append(q_values_padded[2])
        logger['Q_75'].append(q_values_padded[3])
        logger['Q_100'].append(q_values_padded[4])
    
    logger['Chosen_Q_Value'].append(q_values[action] if action < len(q_values) else 0)
    logger['Decision_Reason'].append(decision_reason)
    logger['Epsilon'].append(epsilon)
    logger['Current_Allocation'].append(env.current_allocation)
    logger['Balance'].append(env.balance)
    logger['Portfolio_Value'].append(env.portfolio_value)
    logger['Reward'].append(reward)
    logger['Cumulative_Return'].append(cumulative_return)
    logger['Drawdown'].append(current_drawdown)


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
                    'Q_0', 'Q_25', 'Q_50', 'Q_75', 'Q_100', 'Decision_Reason',
                    'Current_Allocation', 'Portfolio_Value', 'Cumulative_Return', 
                    'Drawdown', 'Reward']

    # Only include columns that exist
    available_cols = [col for col in preview_cols if col in df.columns]
    preview_df = df[available_cols].head(num_rows).copy()

    # Format for better readability
    if 'Current_Price' in preview_df.columns:
        preview_df['Current_Price'] = preview_df['Current_Price'].apply(lambda x: f"${x:.2f}")
    if 'Q_0' in preview_df.columns:
        for q_col in ['Q_0', 'Q_25', 'Q_50', 'Q_75', 'Q_100']:
            if q_col in preview_df.columns:
                preview_df[q_col] = preview_df[q_col].apply(lambda x: f"{x:.4f}")
    if 'Portfolio_Value' in preview_df.columns:
        preview_df['Portfolio_Value'] = preview_df['Portfolio_Value'].apply(lambda x: f"${x:.2f}")
    if 'Current_Allocation' in preview_df.columns:
        preview_df['Current_Allocation'] = preview_df['Current_Allocation'].apply(lambda x: f"{x*100:.0f}%")
    if 'Cumulative_Return' in preview_df.columns:
        preview_df['Cumulative_Return'] = preview_df['Cumulative_Return'].apply(lambda x: f"{x:.2f}%")
    if 'Drawdown' in preview_df.columns:
        preview_df['Drawdown'] = preview_df['Drawdown'].apply(lambda x: f"{x:.2f}%")
    if 'Reward' in preview_df.columns:
        preview_df['Reward'] = preview_df['Reward'].apply(lambda x: f"{x:.4f}")

    print(preview_df.to_string(index=False))
    print("="*100 + "\n")


def plot_episode_performance(df, episode, output_dir='episode_logs'):
    """
    Create enhanced visualization showing portfolio value with allocation markers.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    steps = df['Step'].values
    
    # --- Plot 1: Portfolio Value with allocation markers ---
    if 'Portfolio_Value' in df.columns:
        portfolio_value = df['Portfolio_Value'].values
    else:
        portfolio_value = df['Net_Worth'].values
    
    ax1.plot(steps, portfolio_value, 'b-', linewidth=2, label='Portfolio Value')

    # Mark different allocation levels
    if 'Action' in df.columns:
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        labels = ['0%', '25%', '50%', '75%', '100%']
        
        for action in range(5):
            mask = df['Action'] == action
            if mask.any():
                action_steps = df.loc[mask, 'Step'].values
                action_values = portfolio_value[mask.values]
                ax1.scatter(action_steps, action_values, color=colors[action], 
                          marker='o', s=50, label=labels[action], alpha=0.6, edgecolors='black', linewidths=1)

    # Initial balance reference
    initial_balance = portfolio_value[0]
    ax1.axhline(y=initial_balance, color='gray', linestyle='--',
                linewidth=1, label=f'Start (${initial_balance:.2f})')

    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.set_title(f'Episode {episode} - Portfolio Performance', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Allocation over time ---
    if 'Current_Allocation' in df.columns:
        allocation = df['Current_Allocation'].values * 100
        ax2.plot(steps, allocation, 'purple', linewidth=2)
        ax2.fill_between(steps, 0, allocation, alpha=0.3, color='purple')
    
    ax2.set_ylabel('Allocation (%)', fontsize=11)
    ax2.set_title('Position Size Over Time', fontsize=13, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal lines for allocation levels
    for level in [0, 25, 50, 75, 100]:
        ax2.axhline(y=level, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Plot 3: Cumulative Return with drawdown shading ---
    if 'Cumulative_Return' in df.columns:
        cum_return = df['Cumulative_Return'].values
        ax3.plot(steps, cum_return, 'g-', linewidth=2, label='Cumulative Return')
        
        # Shade drawdown periods
        if 'Drawdown' in df.columns:
            drawdown = df['Drawdown'].values
            ax3.fill_between(steps, cum_return, cum_return - drawdown, 
                            alpha=0.3, color='red', label='Drawdown')
    
    ax3.set_ylabel('Return (%)', fontsize=11)
    ax3.set_title('Cumulative Return & Drawdown', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Q-values evolution ---
    q_cols = ['Q_0', 'Q_25', 'Q_50', 'Q_75', 'Q_100']
    available_q = [col for col in q_cols if col in df.columns]
    
    if available_q:
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        for i, q_col in enumerate(available_q):
            ax4.plot(steps, df[q_col].values, color=colors[i], 
                    linewidth=1.5, label=f'{i*25}% Q-value')
    
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Q-Value', fontsize=11)
    ax4.set_title('Q-Values Evolution', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/episode_{episode}_performance.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def analyze_episode_decisions(df):
    """
    Enhanced analysis of episode decisions with profit-focused metrics.
    """
    print("\n" + "="*100)
    print("PROFESSIONAL EPISODE DECISION ANALYSIS")
    print("="*100)

    # 1. Action Distribution
    if 'Action_Name' in df.columns:
        action_counts = df['Action_Name'].value_counts()
    else:
        action_counts = df['Action'].value_counts().sort_index()
        action_counts.index = [f'{int(i*25)}%' for i in action_counts.index]
    
    action_pcts = (action_counts / len(df) * 100).round(2)

    print("\n1. POSITION SIZING DISTRIBUTION:")
    print("-" * 50)
    alloc_levels = ['0%', '25%', '50%', '75%', '100%']
    for action in alloc_levels:
        count = action_counts.get(f'{action} Allocation', action_counts.get(action, 0))
        pct = action_pcts.get(f'{action} Allocation', action_pcts.get(action, 0.0))
        print(f"   {action:>4} allocation: {count:4d} times ({pct:5.2f}%)")

    # 2. Profitability by Allocation Level
    print("\n2. PROFITABILITY BY ALLOCATION LEVEL:")
    print("-" * 50)
    if 'Action' in df.columns and 'Reward' in df.columns:
        best_allocation = None
        best_avg_reward = -float('inf')
        
        for action in range(5):
            action_df = df[df['Action'] == action]
            if len(action_df) > 0:
                avg_reward = action_df['Reward'].mean()
                total_reward = action_df['Reward'].sum()
                win_rate = (action_df['Reward'] > 0).mean() * 100
                
                print(f"   {action*25:3d}% allocation:")
                print(f"      Avg Reward: {avg_reward:.4f}")
                print(f"      Total Reward: {total_reward:.4f}")
                print(f"      Win Rate: {win_rate:.1f}%")
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_allocation = action * 25
        
        if best_allocation is not None:
            print(f"\n   🏆 BEST PERFORMING: {best_allocation}% allocation")

    # 3. Market Timing Analysis
    print("\n3. MARKET TIMING ANALYSIS:")
    print("-" * 50)
    if 'Current_Price' in df.columns and 'Action' in df.columns:
        # Calculate price changes
        price_changes = df['Current_Price'].pct_change().fillna(0)
        
        # Up days performance
        up_days = price_changes > 0.001
        if up_days.any():
            up_allocation = df.loc[up_days, 'Current_Allocation'].mean() * 100
            print(f"   During UP moves (>{0.1}%): Avg allocation = {up_allocation:.1f}%")
        
        # Down days performance
        down_days = price_changes < -0.001
        if down_days.any():
            down_allocation = df.loc[down_days, 'Current_Allocation'].mean() * 100
            print(f"   During DOWN moves (<-0.1%): Avg allocation = {down_allocation:.1f}%")
        
        # Market timing score
        if 'up_allocation' in locals() and 'down_allocation' in locals():
            timing_score = up_allocation - down_allocation
            print(f"   Market Timing Score: {timing_score:.1f} (higher is better)")

    # 4. Exploration vs Exploitation
    print("\n4. EXPLORATION vs EXPLOITATION:")
    print("-" * 50)
    total = len(df)
    if 'Decision_Reason' in df.columns:
        exploration_count = df['Decision_Reason'].str.contains('Exploration').sum()
        exploitation_count = df['Decision_Reason'].str.contains('Exploitation').sum()
        print(f"   Exploration (Random): {exploration_count:4d} ({exploration_count/total*100:5.2f}%)")
        print(f"   Exploitation (Max Q): {exploitation_count:4d} ({exploitation_count/total*100:5.2f}%)")

    # 5. Risk Metrics
    print("\n5. RISK METRICS:")
    print("-" * 50)
    if 'Portfolio_Value' in df.columns:
        values = df['Portfolio_Value'].values
        returns = np.diff(values) / values[:-1]
        
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            max_drawdown = ((values.max() - values.min()) / values.max()) * 100
            volatility = np.std(returns) * 100
            
            print(f"   Sharpe Ratio: {sharpe:.3f}")
            print(f"   Max Drawdown: {max_drawdown:.2f}%")
            print(f"   Volatility: {volatility:.3f}%")
    
    if 'Drawdown' in df.columns:
        avg_drawdown = df['Drawdown'].mean()
        max_drawdown = df['Drawdown'].max()
        print(f"   Avg Drawdown: {avg_drawdown:.2f}%")
        print(f"   Peak Drawdown: {max_drawdown:.2f}%")

    # 6. Trading Efficiency
    print("\n6. TRADING EFFICIENCY:")
    print("-" * 50)
    if 'Action' in df.columns:
        switches = (df['Action'] != df['Action'].shift()).sum() - 1
        trades_per_step = switches / len(df)
        print(f"   Total allocation changes: {max(switches, 0)}")
        print(f"   Changes per 100 steps: {trades_per_step*100:.1f}")
        
        if switches > 0 and 'Portfolio_Value' in df.columns:
            profit_per_trade = (df['Portfolio_Value'].iloc[-1] - df['Portfolio_Value'].iloc[0]) / switches
            print(f"   Profit per trade: ${profit_per_trade:.2f}")

    # 7. Final Performance Summary
    print("\n7. PERFORMANCE SUMMARY:")
    print("-" * 50)
    if 'Portfolio_Value' in df.columns:
        initial_value = df['Portfolio_Value'].iloc[0]
        final_value = df['Portfolio_Value'].iloc[-1]
    else:
        initial_value = df['Net_Worth'].iloc[0]
        final_value = df['Net_Worth'].iloc[-1]
    
    total_return = ((final_value - initial_value) / initial_value) * 100
    total_reward = df['Reward'].sum()
    
    # Win rate
    if 'Reward' in df.columns:
        win_rate = (df['Reward'] > 0).mean() * 100
        print(f"   Win Rate: {win_rate:.1f}%")
    
    print(f"   Initial Portfolio: ${initial_value:10.2f}")
    print(f"   Final Portfolio:   ${final_value:10.2f}")
    print(f"   Total Return:      {total_return:9.2f}%")
    print(f"   Total Reward:      {total_reward:10.4f}")
    
    # Performance rating
    if total_return > 0:
        if total_return > 10:
            rating = "🌟 EXCELLENT"
        elif total_return > 5:
            rating = "👍 GOOD"
        elif total_return > 2:
            rating = "👌 SATISFACTORY"
        elif total_return > 0:
            rating = "⚖️ BREAK EVEN"
    else:
        if total_return > -5:
            rating = "📉 MINOR LOSS"
        elif total_return > -10:
            rating = "⚠️ MODERATE LOSS"
        else:
            rating = "🔥 SIGNIFICANT LOSS"
    
    print(f"\n   PERFORMANCE RATING: {rating}")
    print("="*100 + "\n")