"""
Simplified Trading Environment Module with 21-bin actions and proper short selling
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
    Simplified trading environment with 21-bin continuous-like actions.
    
    State Features:
    - Normalized price window (window_size features)
    - Shares held (normalized, 1 feature)
    - Balance (normalized, 1 feature)
    
    Actions:
    - 21 bins from -1.0 to +1.0 representing position percentage
      - -1.0 = 100% short
      - 0.0 = 100% cash  
      - +1.0 = 100% long
    """
    
    def __init__(self, df, stock_name="UNKNOWN", initial_balance=10000, window_size=20):
        super(SimplifiedStockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.stock_name = stock_name
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.n_steps = len(df)

        # 21 bins from -1.0 to +1.0
        self.action_space = spaces.Discrete(21)
        self.action_bins = np.linspace(-1.0, 1.0, 21)  # [-1.0, -0.9, ..., 0.9, 1.0]

        # State: Price History + Shares Held + Balance
        total_features = window_size + 2

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        # === FIXED: Commission rate (set to 0 to disable temporarily) ===
        self.commission = 0.0  # 0% commission for testing

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
            window_start_price = 1e-8
        
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
        
        # Safety clip
        state = np.clip(state, -10, 10)
        
        return state.astype(np.float32)
    
    def _action_to_position(self, action_idx):
        """Convert action index to position value from -1.0 to +1.0"""
        return self.action_bins[action_idx]
    
    def step(self, action_idx):
        """
        Execute one trading step with continuous-like action and proper short selling.
        """
        current_price = self.df.iloc[self.current_step]['close']
        done = False

        # Convert action index to target position
        target_position_pct = self._action_to_position(action_idx)
        
        # Calculate target shares (positive for long, negative for short)
        target_value = abs(target_position_pct) * self.net_worth
        target_shares = int(target_value / current_price)
        
        # Apply direction
        if target_position_pct > 0:
            target_shares = target_shares  # Long
        elif target_position_pct < 0:
            target_shares = -target_shares  # Short (negative)
        else:
            target_shares = 0  # Cash
        
        # Calculate shares to trade
        shares_to_trade = target_shares - self.shares_held

        # Execute trades
        if shares_to_trade > 0:  # Buy (increase long position)
            cost = shares_to_trade * current_price
            fee = cost * self.commission
            
            if cost + fee <= self.balance:
                self.balance -= (cost + fee)
                self.shares_held += shares_to_trade
                self.cost_basis = current_price
                self.num_trades += 1
                
        elif shares_to_trade < 0:  # Sell (decrease long OR open short)
            shares_to_sell = -shares_to_trade
            
            if self.shares_held >= 0:  # Currently long or flat
                if shares_to_sell <= self.shares_held:
                    # Selling existing long shares
                    revenue = shares_to_sell * current_price
                    fee = revenue * self.commission
                    self.balance += revenue - fee
                    self.shares_held -= shares_to_sell
                    
                    if self.shares_held == 0:
                        self.cost_basis = 0
                    self.num_trades += 1
                else:
                    # Selling more than we have = opening short position
                    # First sell all long shares
                    if self.shares_held > 0:
                        revenue = self.shares_held * current_price
                        fee = revenue * self.commission
                        self.balance += revenue - fee
                        shares_to_sell -= self.shares_held
                        self.shares_held = 0
                        self.num_trades += 1
                    
                    # Then open short position for remaining
                    if shares_to_sell > 0:
                        # For short, we receive money now but owe shares later
                        revenue = shares_to_sell * current_price
                        fee = revenue * self.commission
                        self.balance += revenue - fee
                        self.shares_held = -shares_to_sell  # Negative = short
                        self.cost_basis = current_price  # Entry price for short
                        self.num_trades += 1
            else:  # Currently short (shares_held is negative)
                # Closing short position (buying to cover)
                if shares_to_sell <= abs(self.shares_held):
                    cost = shares_to_sell * current_price
                    fee = cost * self.commission
                    self.balance -= (cost + fee)
                    self.shares_held += shares_to_sell  # Adding positive reduces short
                    
                    if self.shares_held == 0:
                        self.cost_basis = 0
                    self.num_trades += 1

        # Move to next timestep
        self.current_step += 1

        # Check terminal
        if self.current_step >= self.n_steps - 1:
            done = True

        # Use NEXT price for valuation
        next_price = self.df.iloc[self.current_step]['close']

        # Calculate new net worth (handle short positions)
        if self.shares_held >= 0:  # Long or flat
            position_value = self.shares_held * next_price
        else:  # Short
            # For short: value = initial cash from sale - current cost to buy back
            position_value = abs(self.shares_held) * (2 * self.cost_basis - next_price)

        new_net_worth = self.balance + position_value

        # Update max net worth
        if new_net_worth > self.max_net_worth:
            self.max_net_worth = new_net_worth

        # Reward (simple profit-based)
        reward = (new_net_worth - self.net_worth) / self.initial_balance

        self.net_worth = new_net_worth
        self.history.append(self.net_worth)

        return self._next_observation(), reward, done, False, {}