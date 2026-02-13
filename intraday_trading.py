import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import glob
import argparse
from datetime import datetime

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Technical Indicator Helpers ---
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def add_technical_indicators(df):
    df = df.copy()
    # Ensure no division by zero in returns
    df['rsi'] = calculate_rsi(df['close'], period=14).fillna(50) # Default to neutral 50
    macd, signal = calculate_macd(df['close'])
    df['macd'] = macd.fillna(0)
    df['macd_signal'] = signal.fillna(0)
    
    # Bollinger Bands (20, 2)
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)
    # %B indicator which nicely normalizes price location within bands
    # Avoid division by zero
    diff = upper - lower
    df['bb_pct'] = ((df['close'] - lower) / diff).fillna(0.5) 
     
    return df.fillna(0)

# Load Data
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Pre-calculate indicators for the whole dataset for efficiency
    df = add_technical_indicators(df)
    
    # --- Add Daily Context (Long-term trends) ---
    # Resample to daily to get broader trends
    daily_df = df.resample('1D', on='date').agg({'close': 'last'}).dropna()
    
    # Calculate Daily SMAs and Previous Close
    daily_df['daily_sma_5'] = daily_df['close'].rolling(window=5).mean()
    daily_df['daily_sma_20'] = daily_df['close'].rolling(window=20).mean()
    daily_df['prev_close'] = daily_df['close'].shift(1)
    
    # Reset index to merge
    daily_df = daily_df.reset_index()
    daily_df['day_date'] = daily_df['date'].dt.date
    
    # Create a mapping key in original df
    df['day_date'] = df['date'].dt.date
    
    # Merge daily stats back to intraday df
    df = df.merge(daily_df[['day_date', 'daily_sma_5', 'daily_sma_20', 'prev_close']], on='day_date', how='left')
    
    # Calculate Context Features (available for every minute)
    # 1. Trend 5d: How far is price above/below 5-day average
    df['trend_5d'] = (df['close'] / df['daily_sma_5']) - 1.0
    
    # 2. Trend 20d: How far is price above/below 20-day average
    df['trend_20d'] = (df['close'] / df['daily_sma_20']) - 1.0
    
    # 3. Gap: How much did we gap from yesterday's close?
    # Uses the OPEN of the CURRENT day vs Close of PREV day. 
    # Since we are row-by-row, we can just use current open / prev_close - 1
    # But let's use current close / prev_close - 1 for a dynamic "change from yesterday"
    df['change_from_prev_close'] = (df['close'] / df['prev_close']) - 1.0
    
    # Fill NaNs (first few days won't have headers)
    df = df.fillna(0)
    
    return df

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, window_size=20):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.n_steps = len(df)
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # State: 
        # Price History (Normalized): window_size
        # Volume History (Normalized): window_size
        # Technicals: [RSI, MACD, Signal, BB%] * 1 (Current Step) -> 4 features
        # Account: [SharesHeld (Normalized), Balance (Normalized), Unrealized PnL (Normalized)] -> 3 features
        # Intraday Context: [TimeProgress] -> 1 feature
        # Daily Context: [Trend5d, Trend20d, ChangeFromPrev] -> 3 features
        
        self.n_historics = 2 # Price, Volume
        self.n_technicals = 4 # RSI, MACD, Signal, BB%
        self.n_account = 3
        self.n_context = 1 + 3 # Time + 3 Daily Context features
        
        total_features = (self.n_historics * window_size) + self.n_technicals + self.n_account + self.n_context
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32)
        
        # Commission fee (0.1%)
        self.commission = 0.001 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = self.window_size
        self.history = []
        self.trades = []
        
        return self._next_observation(), {}
    
    def _next_observation(self):
        # Window of data
        frame = self.df.iloc[self.current_step - self.window_size : self.current_step]
        current_row = self.df.iloc[self.current_step - 1]
        
        # 1. Price History (Normalized relative to start of window)
        # This makes the agent robust to any price level (100 or 10000)
        window_start_price = frame['close'].iloc[0]
        if window_start_price == 0: window_start_price = 1e-8 # Safety
        
        prices_norm = (frame['close'].values / window_start_price) - 1.0
        
        # 2. Volume History (Normalized by window mean)
        vol_mean = frame['volume'].mean()
        if vol_mean == 0: vol_mean = 1.0
        volumes_norm = (frame['volume'].values / vol_mean) - 1.0
        
        # 3. Technicals (Already somewhat normalized, but let's scale)
        # RSI is 0-100 -> scale to 0-1
        rsi = current_row['rsi'] / 100.0
        # MACD values are small, usually < 1% of price. Normalize by price
        macd = current_row['macd'] / current_row['close'] * 100 # Multiplied for visibility
        signal = current_row['macd_signal'] / current_row['close'] * 100
        bb_pct = current_row['bb_pct'] # Already normalized approx 0-1
        
        technicals = np.array([rsi, macd, signal, bb_pct])
        
        # 4. Account Data
        # Shares held: Normalize by theoretical max shares (balance / price)
        # But balance changes. Let's use a loose normalization or log scale.
        # Simple: Shares / (InitialBalance / CurrentPrice)
        max_possible_shares = self.initial_balance / current_row['close']
        shares_norm = self.shares_held / max_possible_shares if max_possible_shares > 0 else 0
        
        balance_norm = self.balance / self.initial_balance
        
        unrealized_pnl = 0
        if self.shares_held > 0:
            unrealized_pnl = (current_row['close'] - self.cost_basis) / self.cost_basis
            
        account_state = np.array([shares_norm, balance_norm, unrealized_pnl])
        
        # 5. Context
        time_progress = self.current_step / self.n_steps
        
        # Daily Context
        trend_5d = current_row['trend_5d']
        trend_20d = current_row['trend_20d']
        change_prev = current_row['change_from_prev_close']
        
        context = np.array([time_progress, trend_5d, trend_20d, change_prev])
        
        state = np.concatenate([
            prices_norm, 
            volumes_norm, 
            technicals, 
            account_state,
            context
        ])
        
        # Safety clip to avoid infs messing up NN
        state = np.clip(state, -10, 10)
        
        return state.astype(np.float32)
    
    def step(self, action):
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['close']
        
        reward = 0
        done = False
        
        # Force sell at end of day
        if self.current_step >= self.n_steps - 1:
            done = True
            if self.shares_held > 0:
                action = 2 
        
        trade_occurred = False
        
        # Execute Action
        if action == 1: # Buy
            # Logic: Buy 25% of current buying power
            # To allow accumulation
            spend = self.balance * 0.25
            
            # Simple logic: If we have enough for 1 share and spend > price
            if self.balance >= current_price:
                # Ensure we buy at least 1 share if we have money
                if spend < current_price: spend = current_price
                
                shares_bought = int(spend // current_price)
                if shares_bought > 0:
                    cost = shares_bought * current_price
                    fee = cost * self.commission
                    total_outflow = cost + fee
                    
                    if self.balance >= total_outflow:
                        self.balance -= total_outflow
                        
                        # Update Avg Cost
                        prev_total_cost = self.shares_held * self.cost_basis
                        self.shares_held += shares_bought
                        self.cost_basis = (prev_total_cost + cost) / self.shares_held
                        trade_occurred = True
        
        elif action == 2: # Sell
             # Logic: Sell 50% of holding - allows scaling out
             # Or simplify to Sell All for clearer learning sign
            if self.shares_held > 0:
                # Sell ALL for now to simplify learning the "Exit" signal
                shares_to_sell = self.shares_held 
                revenue = shares_to_sell * current_price
                fee = revenue * self.commission
                net_revenue = revenue - fee
                
                profit = net_revenue - (shares_to_sell * self.cost_basis)
                
                self.balance += net_revenue
                self.shares_held -= shares_to_sell
                if self.shares_held == 0:
                    self.cost_basis = 0
                
                # Reward for realizing profit? 
                # Better to reward total net worth change, but realizing profit is a strong signal
                trade_occurred = True

        self.current_step += 1
        
        new_net_worth = self.balance + (self.shares_held * current_price)
        
        # Reward Function
        # 1. Change in Net Worth (The ultimate goal)
        reward = (new_net_worth - self.net_worth)
        
        # 2. Small penalty for holding over long time with no gain? 
        # Or penalty for trading too much? Commission handles trading penalty.
        
        self.net_worth = new_net_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            
        self.history.append(self.net_worth)
            
        return self._next_observation(), reward, done, False, {}

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256) # Increased capacity
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# INTERPRETABILITY & VISUALIZATION HELPERS (NEW CODE ONLY)
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


# ============================================================================
# MODIFIED TRAINING FUNCTION (Minimal changes, only adds logging calls)
# ============================================================================

def train_agent():
    parser = argparse.ArgumentParser(description='Train stock trading agent.')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--log-episodes', type=int, default=5, 
                        help='Save detailed logs every N episodes (0 to disable)')
    args = parser.parse_args()

    # Load and Group Data
    all_files = glob.glob('processed_data/*.csv')
    print(f"Found {len(all_files)} files in processed_data/")
    
    all_daily_groups = []
    
    for file_path in all_files:
        df = load_data(file_path)
        if df is None: continue
        
        # Group by Day
        df['day'] = df['date'].dt.date
        # Only take days with enough data (e.g. at least 100 minutes)
        daily_groups = [group for _, group in df.groupby('day') if len(group) > 60]
        all_daily_groups.extend(daily_groups)
    
    if not all_daily_groups:
        print("No valid trading days found.")
        return
        
    print(f"Loaded {len(all_daily_groups)} trading days total.")
    
    # Init Agent (using dummy env to get shapes)
    dummy_env = StockTradingEnv(all_daily_groups[0])
    input_dim = dummy_env.observation_space.shape[0]
    output_dim = dummy_env.action_space.n
    
    print(f"Input Dimension: {input_dim}, Output Dimension: {output_dim}")
    
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005) # Lower LR for stability
    replay_buffer = ReplayBuffer(50000) # Larger buffer
    
    # Hyperparams
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995 # Slower decay
    epsilon_min = 0.05
    episodes = args.episodes
    
    rewards_history = []
    
    print("Starting Intraday Training...")
    
    # Training Loop
    for episode in range(episodes):
        # Pick a random day
        day_data = random.choice(all_daily_groups)
        env = StockTradingEnv(day_data)
        
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # Initialize episode logger (only if we're logging this episode)
        should_log = (args.log_episodes > 0) and ((episode + 1) % args.log_episodes == 0)
        if should_log:
            episode_logger = create_episode_logger()
        
        step_count = 0
        
        while not done:
            # Decide action
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
            
            next_state, reward, done, _, _ = env.step(action)
            
            # Log step (only if logging this episode)
            if should_log:
                log_step(episode_logger, episode + 1, step_count, env, state, 
                        action, q_values, epsilon, reward, was_random)
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if len(replay_buffer) > 2000: # Wait for buffer to fill a bit
                transitions = replay_buffer.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                
                batch_state = torch.FloatTensor(np.array(batch_state))
                batch_action = torch.LongTensor(batch_action).unsqueeze(1)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                batch_next_state = torch.FloatTensor(np.array(batch_next_state))
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1)
                
                curr_q = policy_net(batch_state).gather(1, batch_action)
                next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                expected_q = batch_reward + gamma * next_q * (1 - batch_done)
                
                loss = nn.MSELoss()(curr_q, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0) # Gradient clipping
                optimizer.step()
        
        # Post-episode processing
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Save and analyze episode log
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
            
        rewards_history.append(total_reward)
        print(f"Episode {episode+1}: Day {env.df.iloc[0]['date'].date()} | Reward: {total_reward:.2f} | Final Balance: {env.balance:.2f} | Net Worth: {env.net_worth:.2f} | Epsilon: {epsilon:.2f}")

    # Plot overall training
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history)
    plt.title("Intraday Training Rewards Over All Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('training_rewards.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nTraining complete. Overall plot saved to training_rewards.png")

if __name__ == "__main__":
    train_agent()