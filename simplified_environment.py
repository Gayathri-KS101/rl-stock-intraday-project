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

        # Actions: 0 = Full cash, 1 = Fully invested
        self.action_space = spaces.Discrete(2)

        # State: Price History + Shares Held + Balance
        # - Price window (normalized): window_size features
        # - Shares held (normalized): 1 feature
        # - Balance (normalized): 1 feature
        total_features = window_size + 2

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        # Commission fee (0.2%)
        self.commission = 0.000

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
        
        # 1. Price History (Normalized relative to start of window)
        window_start_price = frame['close'].iloc[0]
        if window_start_price == 0:
            window_start_price = 1e-8  # Safety
        
        prices_norm = (frame['close'].values / window_start_price) - 1.0
        
        # 2. Shares Held (Normalized by theoretical max shares)
        max_possible_shares = self.initial_balance / current_row['close']
        shares_norm = self.shares_held / max_possible_shares if max_possible_shares > 0 else 0
        
        # 3. Balance (Normalized by initial balance)
        balance_norm = self.balance / self.initial_balance
        
        # Concatenate state
        state = np.concatenate([
            prices_norm,
            [shares_norm],
            [balance_norm]
        ])
        
        # Safety clip to avoid infs
        state = np.clip(state, -10, 10)
        
        return state.astype(np.float32)
    
    def step(self, action):
        """
        Execute one trading step.

        Args:
            action: 0 = Stay fully in cash, 1 = Fully invested
        """

        current_price = self.df.iloc[self.current_step]['close']
        done = False

        # --- Execute action at current_price ---
        if action == 1 and self.shares_held == 0:
            shares_to_buy = int(self.balance / (current_price * (1 + self.commission)))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                fee = cost * self.commission
                self.balance -= (cost + fee)
                self.shares_held = shares_to_buy
                self.cost_basis = current_price
                self.num_trades += 1

        elif action == 0 and self.shares_held > 0:
            revenue = self.shares_held * current_price
            fee = revenue * self.commission
            self.balance += revenue - fee
            self.shares_held = 0
            self.cost_basis = 0
            self.num_trades += 1

        # --- Move to next timestep ---
        self.current_step += 1

        # Check terminal
        if self.current_step >= self.n_steps - 1:
            done = True

        # Use NEXT price for valuation
        next_price = self.df.iloc[self.current_step]['close']

        # --- Calculate new net worth using next price ---
        new_net_worth = self.balance + (self.shares_held * next_price)

        if new_net_worth > self.max_net_worth:
            self.max_net_worth = new_net_worth

        reward = (new_net_worth - self.net_worth) / self.initial_balance
        reward = reward * 100

        self.net_worth = new_net_worth
        self.history.append(self.net_worth)

        return self._next_observation(), reward, done, False, {}
