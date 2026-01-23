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

def train_agent():
    parser = argparse.ArgumentParser(description='Train stock trading agent.')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
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
        
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()
            
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
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
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        rewards_history.append(total_reward)
        print(f"Episode {episode+1}: Day {env.df.iloc[0]['date'].date()} | Reward: {total_reward:.2f} | Final Balance: {env.balance:.2f} | Net Worth: {env.net_worth:.2f} | Epsilon: {epsilon:.2f}")

    # Plot
    plt.plot(rewards_history)
    plt.title("Intraday Training Rewards")
    plt.savefig('training_rewards.png') # Save instead of show
    print("Training complete. Plot saved to training_rewards.png")

if __name__ == "__main__":
    train_agent()
