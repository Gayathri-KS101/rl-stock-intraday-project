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

# Load Data
def load_data(file_path):
    """Load CSV data and prepare for training."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

class SimplifiedStockTradingEnv(gym.Env):
    """
    Simplified trading environment with minimal state:
    - Normalized price window
    - Shares held (normalized)
    - Balance (normalized)
    
    Actions:
    - 0 = Hold
    - 1 = Buy exactly 1 share
    - 2 = Sell exactly 1 share
    
    Reward: Change in net worth between steps
    """
    def __init__(self, df, initial_balance=10000, window_size=20):
        super(SimplifiedStockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.n_steps = len(df)
        
        # Actions: 0=Hold, 1=Buy 1 share, 2=Sell 1 share
        self.action_space = spaces.Discrete(3)
        
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
        
        # Commission fee (0.1%)
        #self.commission = 0.001
        self.commission = 0.002

        
        # Track number of trades
        self.num_trades = 0

    def reset(self, seed=None, options=None):
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
        Construct minimal state:
        1. Normalized price window
        2. Normalized shares held
        3. Normalized balance
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

# Update max net worth FIRST
        if new_net_worth > self.max_net_worth:
            self.max_net_worth = new_net_worth

# --- Profit component ---
        profit = (new_net_worth - self.net_worth) / self.initial_balance

# --- Drawdown penalty ---
        drawdown = (self.max_net_worth - new_net_worth) / self.max_net_worth
        drawdown_penalty = -3.0 * drawdown

# --- Exposure penalty ---
        exposure = (self.shares_held * current_price) / new_net_worth if new_net_worth > 0 else 0
        exposure_penalty = -0.2 * exposure

# --- Final reward ---
        reward = profit + drawdown_penalty + exposure_penalty
        reward = np.clip(reward, -5, 5)

        # Reward: Change in net worth
        #reward = new_net_worth - self.net_worth
        
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
        self.fc2 = nn.Linear(128, 256)
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
    
    # 5. Q-value statistics
    print("\n5. Q-VALUE STATISTICS (Overall):")
    print("-" * 50)
    print(f"   Q_Hold - Mean: {df['Q_Hold'].mean():7.2f}, Std: {df['Q_Hold'].std():7.2f}")
    print(f"   Q_Buy  - Mean: {df['Q_Buy'].mean():7.2f}, Std: {df['Q_Buy'].std():7.2f}")
    print(f"   Q_Sell - Mean: {df['Q_Sell'].mean():7.2f}, Std: {df['Q_Sell'].std():7.2f}")
    
    # 6. Performance summary
    print("\n6. PERFORMANCE SUMMARY:")
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
# TRAINING FUNCTION
# ============================================================================

def train_agent():
    parser = argparse.ArgumentParser(description='Train simplified stock trading agent.')
    parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('--log-episodes', type=int, default=10, 
                        help='Save detailed logs every N episodes (0 to disable)')
    args = parser.parse_args()

    # Load and Group Data
    all_files = glob.glob('processed_data/*.csv')
    print(f"Found {len(all_files)} files in processed_data/")
    
    all_daily_groups = []
    
    for file_path in all_files:
        df = load_data(file_path)
        if df is None:
            continue
        
        # Group by Day
        df['day'] = df['date'].dt.date
        # Only take days with enough data (e.g. at least 60 minutes)
        daily_groups = [group for _, group in df.groupby('day') if len(group) > 60]
        all_daily_groups.extend(daily_groups)
    
    if not all_daily_groups:
        print("No valid trading days found.")
        return
        
    print(f"Loaded {len(all_daily_groups)} trading days total.")
    
    # Initialize Agent
    dummy_env = SimplifiedStockTradingEnv(all_daily_groups[0])
    input_dim = dummy_env.observation_space.shape[0]
    output_dim = dummy_env.action_space.n
    
    print(f"Input Dimension: {input_dim}, Output Dimension: {output_dim}")
    
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    replay_buffer = ReplayBuffer(50000)
    
    # Hyperparameters
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    episodes = args.episodes
    
    rewards_history = []
    
    # Episode metrics logger
    episode_metrics = {
        'Episode': [],
        'Total_Reward': [],
        'Final_Net_Worth': [],
        'Percent_Return': [],
        'Number_of_Trades': []
    }
    
    print("Starting Simplified Intraday Training...")
    
    # Training Loop
    for episode in range(episodes):
        # Pick a random day
        day_data = random.choice(all_daily_groups)
        env = SimplifiedStockTradingEnv(day_data)
        
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
            
            # Train from replay buffer
            if len(replay_buffer) > 2000:
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
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        # Post-episode processing
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Calculate episode metrics
        percent_return = ((env.net_worth - env.initial_balance) / env.initial_balance) * 100
        
        # Log episode metrics
        episode_metrics['Episode'].append(episode + 1)
        episode_metrics['Total_Reward'].append(total_reward)
        episode_metrics['Final_Net_Worth'].append(env.net_worth)
        episode_metrics['Percent_Return'].append(percent_return)
        episode_metrics['Number_of_Trades'].append(env.num_trades)
        
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
        print(f"Episode {episode+1}: Reward: {total_reward:.2f} | Net Worth: ${env.net_worth:.2f} | Return: {percent_return:.2f}% | Trades: {env.num_trades} | Epsilon: {epsilon:.2f}")
        
        # Generate training plot every 50 episodes
        if (episode + 1) % 50 == 0:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards_history)
            plt.title(f"Training Rewards (Episodes 1-{episode+1})", fontsize=14, fontweight='bold')
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Total Reward", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(f'training_rewards_ep{episode+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved training plot: training_rewards_ep{episode+1}.png")

    # Save episode metrics to CSV
    metrics_df = pd.DataFrame(episode_metrics)
    metrics_df.to_csv('episode_metrics.csv', index=False)
    print(f"\nSaved episode metrics to: episode_metrics.csv")
    
    # Generate final training plot
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history)
    plt.title("Training Rewards Over All Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('training_rewards_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training complete. Final plot saved to training_rewards_final.png")
    
    # Print summary statistics
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    print(f"Total Episodes:        {episodes}")
    print(f"Average Reward:        {np.mean(rewards_history):.2f}")
    print(f"Best Episode Reward:   {np.max(rewards_history):.2f}")
    print(f"Worst Episode Reward:  {np.min(rewards_history):.2f}")
    print(f"Average Return:        {metrics_df['Percent_Return'].mean():.2f}%")
    print(f"Average Trades/Episode: {metrics_df['Number_of_Trades'].mean():.1f}")
    print("="*100)

if __name__ == "__main__":
    train_agent()