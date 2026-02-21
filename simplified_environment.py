"""
Simplified Trading Environment Module

This module contains:
- Minimal data loading
- Simplified stock trading environment with basic state representation
- Single-share buy/sell actions
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os


def load_data(file_path):
    """
    Load CSV data and prepare for training.
    
    Args:
        file_path: Path to CSV file containing stock data
        
    Returns:
        DataFrame with date column converted to datetime and sorted
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    return df


class SimplifiedStockTradingEnv(gym.Env):
    """
    Simplified trading environment with minimal state representation.
    
    State Features:
    - Normalized price window (window_size features)
    - Shares held (normalized, 1 feature)
    - Balance (normalized, 1 feature)
    
    Actions:
    - 0 = Hold (do nothing)
    - 1 = Buy exactly 1 share
    - 2 = Sell exactly 1 share
    
    Reward:
    - Profit component (change in net worth)
    - Drawdown penalty (discourages large losses)
    - Exposure penalty (discourages over-concentration)
    """
    
    def __init__(self, df, stock_name="UNKNOWN", initial_balance=10000, window_size=20):
        super(SimplifiedStockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.stock_name = stock_name
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.n_steps = len(df)
        
        # Actions: 0=Hold, 1=Buy 1 share, 2=Sell 1 share
        self.action_space = spaces.Discrete(3)
        
        # State: Price History + Shares Held + Balance
        # - Price window (normalized): window_size features
        # - Shares held (normalized): 1 feature
        # - Balance (normalized): 1 feature
        total_features = window_size + 4
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_features,), 
            dtype=np.float32
        )
        
        # Commission fee (0.2%)
        self.commission = 0.002
        
        # Track number of trades
        self.num_trades = 0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = self.window_size
        self.history = []
        self.num_trades = 0
        
        return self._next_observation(), {}
    
    def _next_observation(self):
        """
        Construct minimal state representation.
        
        Returns:
            Normalized state vector containing:
            1. Price window (normalized to start of window)
            2. Shares held (normalized by max possible shares)
            3. Balance (normalized by initial balance)
        """
        # Window of price data
        frame = self.df.iloc[self.current_step - self.window_size : self.current_step]
        current_row = self.df.iloc[self.current_step - 1]
        
       # 1. Price History
        window_start_price = frame['close'].iloc[0]
        if window_start_price == 0:
            window_start_price = 1e-8

        prices_norm = (frame['close'].values / window_start_price) - 1.0

        # 2. Shares Held
        max_possible_shares = self.initial_balance / current_row['close']
        shares_norm = self.shares_held / max_possible_shares if max_possible_shares > 0 else 0

        # 3. Balance
        balance_norm = self.balance / self.initial_balance

        # 4. Momentum & Volatility
        returns = frame['close'].pct_change().dropna().values
        momentum = returns.mean() if len(returns) > 0 else 0.0
        volatility = returns.std() if len(returns) > 0 else 0.0

        momentum = np.clip(momentum, -1, 1)
        volatility = np.clip(volatility, 0, 1)

        state = np.concatenate([
            prices_norm,
            [shares_norm],
            [balance_norm],
            [momentum],
            [volatility]
        ])
        
        # Safety clip to avoid infs
        state = np.clip(state, -10, 10)
        
        return state.astype(np.float32)
    
    def step(self, action):
        """
        Execute one trading step.
        
        Args:
            action: 0=Hold, 1=Buy 1 share, 2=Sell 1 share
            
        Returns:
            observation, reward, done, truncated, info
        """
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['close']
        
        done = False
        
        # Force sell at end of episode
        if self.current_step >= self.n_steps - 1:
            done = True
            if self.shares_held > 0:
                action = 2
        
        # Execute Action
        if action == 1:  # Buy exactly 1 share
            cost = current_price
            fee = cost * self.commission
            total_outflow = cost + fee
            
            if self.balance >= total_outflow:
                self.balance -= total_outflow
                
                # Update average cost basis
                prev_total_cost = self.shares_held * self.cost_basis
                self.shares_held += 1
                self.cost_basis = (prev_total_cost + cost) / self.shares_held
                self.num_trades += 1
        
        elif action == 2:  # Sell exactly 1 share
            if self.shares_held > 0:
                revenue = current_price
                fee = revenue * self.commission
                net_revenue = revenue - fee
                
                self.balance += net_revenue
                self.shares_held -= 1
                if self.shares_held == 0:
                    self.cost_basis = 0
                
                self.num_trades += 1

        self.current_step += 1
        
        # Calculate new net worth
        new_net_worth = self.balance + (self.shares_held * current_price)

        # Update max net worth FIRST (for drawdown calculation)
        if new_net_worth > self.max_net_worth:
            self.max_net_worth = new_net_worth

        # --- Reward Components ---
        
        # 1. Profit component
        profit = (new_net_worth - self.net_worth) / self.initial_balance

        # 2. Small transaction penalty
        trade_penalty = -0.0005 if action in [1, 2] else 0

        reward = profit + trade_penalty
        reward = np.clip(reward, -5, 5)
        
        # Update net worth tracking
        self.net_worth = new_net_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            
        self.history.append(self.net_worth)
            
        return self._next_observation(), reward, done, False, {}