"""
Professional Trading Environment with Advanced Position Sizing
Optimized for maximum profit with proper risk management
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
from collections import deque


def load_data(file_path):
    """Load and prepare CSV data with enhanced features."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Add technical indicators with safety checks
        df['returns'] = df['close'].pct_change().fillna(0)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['returns'].rolling(10).std().fillna(0)
        df['rsi'] = calculate_rsi(df['close'], 14).fillna(50)
        df['macd'] = calculate_macd(df['close']).fillna(0)
        
        # Replace any infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD technical indicator."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


class ProfessionalTradingEnv(gym.Env):
    """
    Professional trading environment optimized for maximum profit.
    
    Key Features:
    - 5 allocation levels (0%, 25%, 50%, 75%, 100%)
    - Advanced reward shaping for profit maximization
    - Strict risk management with drawdown control
    - Technical indicators in state space
    - Position persistence to reduce overtrading
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=10000, window_size=20):
        super(ProfessionalTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.window_size = window_size
        self.n_steps = len(df)
        
        # Action space: 5 allocation levels
        self.action_space = spaces.Discrete(5)
        self.allocation_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Enhanced state space with technical indicators
        # Features: price_history(20) + returns(5) + rsi(1) + macd(1) + volatility(1) + allocation(1) + balance(1) + drawdown(1) + performance(1)
        self.total_features = window_size + 12  # 20 + 5 + 3 + 2 + 2 = 32
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.total_features,), dtype=np.float32
        )
        
        # Trading parameters
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005   # 0.05% slippage estimate
        
        # Position management
        self.current_allocation = 0.0
        self.portfolio_value = self.initial_balance
        self.balance = self.initial_balance
        self.peak_value = self.initial_balance
        
        # Performance tracking
        self.num_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.total_return = 0.0
        
        # Position persistence (CRITICAL for reducing overtrading)
        self.min_steps_per_position = 10
        self.steps_in_position = 0
        self.current_position_action = 0
        
        # Trade cooldown
        self.steps_since_last_trade = 0
        self.min_steps_between_trades = 5
        
        # Performance history
        self.returns_history = deque(maxlen=100)
        self.equity_curve = []
        
        # Current step tracking
        self.current_step = self.window_size
        
    def reset(self, seed=None, options=None):
        """Reset environment with enhanced tracking."""
        super().reset(seed=seed)
        
        self.current_allocation = 0.0
        self.portfolio_value = self.initial_balance
        self.balance = self.initial_balance
        self.peak_value = self.initial_balance
        self.current_step = self.window_size
        self.num_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.steps_in_position = 0
        self.current_position_action = 0
        self.steps_since_last_trade = 0
        self.returns_history.clear()
        self.equity_curve = [self.portfolio_value]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """
        Construct rich state representation with fixed dimensions.
        Returns exactly 32-dimensional state vector.
        """
        try:
            # Get window of data
            start_idx = max(0, self.current_step - self.window_size)
            frame = self.df.iloc[start_idx:self.current_step]
            current_row = self.df.iloc[max(0, self.current_step - 1)]
            
            # 1. Normalized price history (EXACTLY window_size values)
            if len(frame) > 0:
                base_price = float(frame['close'].iloc[0]) if len(frame) > 0 else 1.0
                prices_norm = (frame['close'].values / base_price) - 1.0
                
                # Pad or truncate to exactly window_size
                if len(prices_norm) < self.window_size:
                    # Pad with zeros at the beginning
                    prices_norm = np.pad(prices_norm, (self.window_size - len(prices_norm), 0), 'constant', constant_values=0)
                elif len(prices_norm) > self.window_size:
                    # Take last window_size values
                    prices_norm = prices_norm[-self.window_size:]
            else:
                prices_norm = np.zeros(self.window_size)
            
            # 2. Recent returns (EXACTLY 5 values)
            recent_returns = self.df['returns'].iloc[
                max(0, self.current_step-5):self.current_step
            ].values
            if len(recent_returns) < 5:
                recent_returns = np.pad(recent_returns, (5 - len(recent_returns), 0), 'constant', constant_values=0)
            elif len(recent_returns) > 5:
                recent_returns = recent_returns[-5:]
            
            # 3. Technical indicators (3 values)
            rsi = float(current_row.get('rsi', 50)) / 100.0
            macd = float(np.tanh(current_row.get('macd', 0)))
            volatility = float(min(current_row.get('volatility', 0.01), 0.1)) / 0.1
            
            # 4. Portfolio state (2 values)
            allocation_norm = float(self.current_allocation)
            balance_norm = float(self.balance / max(self.initial_balance, 1))
            
            # 5. Risk metrics (2 values)
            drawdown = float((self.peak_value - self.portfolio_value) / max(self.peak_value, 1)) if self.peak_value > 0 else 0
            recent_performance = float(np.mean(list(self.returns_history)[-10:])) if self.returns_history else 0
            
            # Combine all features
            state_parts = [
                prices_norm.astype(np.float32),                    # 20 values
                recent_returns.astype(np.float32),                 # 5 values
                np.array([rsi, macd, volatility], dtype=np.float32), # 3 values
                np.array([allocation_norm, balance_norm], dtype=np.float32), # 2 values
                np.array([drawdown, recent_performance], dtype=np.float32)   # 2 values
            ]
            
            state = np.concatenate(state_parts)
            
            # Ensure exactly 32 dimensions
            if len(state) != 32:
                print(f"Warning: State dimension is {len(state)}, expected 32. Adjusting...")
                if len(state) < 32:
                    state = np.pad(state, (0, 32 - len(state)), 'constant', constant_values=0)
                else:
                    state = state[:32]
            
            return state.astype(np.float32)
            
        except Exception as e:
            print(f"Error in _get_observation: {e}")
            # Return zero state as fallback
            return np.zeros(32, dtype=np.float32)
    
    @property
    def net_worth(self):
        """Compatibility property for older code"""
        return self.portfolio_value
    
    @property
    def shares_held(self):
        """Compatibility property for older code"""
        return self.current_allocation
    
    def step(self, action):
        """
        Execute trade with professional risk management.
        """
        try:
            current_price = float(self.df.iloc[self.current_step]['close'])
            prev_price = float(self.df.iloc[self.current_step - 1]['close'])
        except:
            current_price = prev_price = 1.0
        
        done = False
        
        # === POSITION PERSISTENCE (Prevents overtrading) ===
        if self.steps_in_position < self.min_steps_per_position:
            # Must maintain current position
            action = self.current_position_action
            self.steps_in_position += 1
        else:
            # Can change position
            if action != self.current_position_action:
                self.current_position_action = action
                self.steps_in_position = 1
            else:
                self.steps_in_position += 1
        
        # === TRADE COOLDOWN ===
        target_allocation = self.allocation_levels[action]
        allocation_change = abs(target_allocation - self.current_allocation)
        
        trade_executed = False
        if allocation_change > 0.01:  # Significant change
            if self.steps_since_last_trade >= self.min_steps_between_trades:
                trade_executed = True
                self.num_trades += 1
                self.steps_since_last_trade = 0
            else:
                # Cooldown - maintain allocation
                target_allocation = self.current_allocation
        
        # Update allocation
        old_allocation = self.current_allocation
        self.current_allocation = target_allocation
        self.steps_since_last_trade += 1
        
        # Calculate returns
        if prev_price != 0:
            price_return = (current_price - prev_price) / prev_price
        else:
            price_return = 0
            
        portfolio_return = self.current_allocation * price_return
        
        # Apply transaction costs
        if trade_executed:
            trade_value = self.portfolio_value * abs(self.current_allocation - old_allocation)
            transaction_cost = trade_value * (self.commission + self.slippage)
            portfolio_return -= transaction_cost / max(self.portfolio_value, 1)
        
        # Update portfolio
        old_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)
        self.balance = self.portfolio_value * (1 - self.current_allocation)
        
        # Track peak for drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        # Track returns
        step_return = (self.portfolio_value - old_value) / max(old_value, 1)
        self.returns_history.append(step_return)
        self.equity_curve.append(self.portfolio_value)
        
        # Update step
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True
        
        # === PROFESSIONAL REWARD FUNCTION ===
        reward = self._calculate_reward(portfolio_return, step_return, price_return, trade_executed)
        
        return self._get_observation(), reward, done, False, {}
    
    def _calculate_reward(self, portfolio_return, step_return, price_return, trade_executed):
        """
        Advanced reward function designed for maximum profit.
        """
        # === 1. BASE RETURN ===
        base_reward = portfolio_return * 100
        
        # === 2. RISK-ADJUSTED BONUS ===
        if len(self.returns_history) > 20:
            recent_returns = np.array(list(self.returns_history)[-20:])
            volatility = np.std(recent_returns) + 1e-8
            sharpe = np.mean(recent_returns) / volatility
            risk_bonus = sharpe * 10
        else:
            risk_bonus = 0
        
        # === 3. DRAWDOWN PENALTY ===
        current_drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1)
        if current_drawdown > 0.05:  # 5% drawdown starts penalty
            drawdown_penalty = -current_drawdown * 50
        elif current_drawdown > 0.02:  # 2-5% drawdown
            drawdown_penalty = -current_drawdown * 20
        else:
            drawdown_penalty = 0
        
        # === 4. MARKET TIMING BONUS ===
        if price_return > 0.001:  # Market up
            if self.current_allocation > 0.5:  # We're invested
                timing_bonus = price_return * 30
            elif self.current_allocation < 0.25:  # We're in cash
                timing_bonus = -price_return * 20
            else:
                timing_bonus = 0
        elif price_return < -0.001:  # Market down
            if self.current_allocation < 0.25:  # We're in cash
                timing_bonus = -price_return * 30
            elif self.current_allocation > 0.5:  # We're invested
                timing_bonus = price_return * 20
            else:
                timing_bonus = 0
        else:
            timing_bonus = 0
        
        # === 5. CONSISTENCY BONUS ===
        if len(self.returns_history) > 10:
            positive_streak = sum(1 for r in list(self.returns_history)[-10:] if r > 0)
            consistency_bonus = (positive_streak - 5) * 0.5
        else:
            consistency_bonus = 0
        
        # === 6. TRADE EFFICIENCY ===
        if self.num_trades > 0:
            avg_trade_value = abs(self.portfolio_value - self.initial_balance) / self.num_trades
            trade_efficiency = avg_trade_value / max(self.initial_balance, 1)
            efficiency_bonus = trade_efficiency * 5
        else:
            efficiency_bonus = 0
        
        # === 7. WIN/LOSS RATIO BONUS ===
        if self.num_trades > 5:
            win_rate = self.winning_trades / max(1, self.num_trades)
            win_bonus = (win_rate - 0.5) * 10
        else:
            win_bonus = 0
        
        # Combine all components
        reward = (
            base_reward * 0.4 +
            risk_bonus * 0.3 +
            drawdown_penalty * 0.2 +
            timing_bonus * 0.1 +
            consistency_bonus * 0.05 +
            efficiency_bonus * 0.05 +
            win_bonus * 0.1
        )
        
        # Update win/loss tracking
        if step_return > 0.001:
            self.winning_trades += 1
            self.consecutive_losses = 0
        elif step_return < -0.001:
            self.losing_trades += 1
            self.consecutive_losses += 1
        
        # Emergency stop-loss
        if self.consecutive_losses > 5:
            reward -= 5
            self.current_allocation = 0
            
        return np.clip(reward, -20, 20)
    
    def get_metrics(self):
        """Return comprehensive performance metrics."""
        returns_array = np.array(list(self.returns_history)) if self.returns_history else np.array([0])
        
        return {
            'portfolio_value': self.portfolio_value,
            'total_return': ((self.portfolio_value / self.initial_balance) - 1) * 100,
            'num_trades': self.num_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / max(1, self.num_trades)) * 100,
            'max_drawdown': self.max_drawdown * 100,
            'avg_allocation': float(np.mean([self.current_allocation])),
            'sharpe_ratio': float(np.mean(returns_array) / (np.std(returns_array) + 1e-8)) if len(returns_array) > 0 else 0
        }