"""
PPO Training Script for Your Online Trading Project
- Chronological train/test split
- Decision interval
- Reward = change in net worth per step
- Tracks Reward, Net Worth, Trades, Return, Win Rate
"""

import os
import glob
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from simplified_environment import SimplifiedStockTradingEnv
from data_splitter import load_and_split_data
from performance_commentary import generate_performance_summary


# -------------------------------
# PPO Environment Wrapper
# -------------------------------
class PPOTradingEnv(SimplifiedStockTradingEnv):
    def __init__(self, day_data, stock_name, initial_balance=10000):
        super().__init__(day_data, stock_name, initial_balance)
        self.prev_net_worth = initial_balance
        self.total_reward = 0
        self.num_trades = 0

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        # Reward = change in net worth
        reward = self.net_worth - self.prev_net_worth
        self.prev_net_worth = self.net_worth
        self.total_reward += reward
        self.num_trades = self.shares_held  # assume shares_held updates on buy/sell

        info.update({
            "net_worth": self.net_worth,
            "total_reward": self.total_reward,
            "num_trades": self.num_trades
        })

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.prev_net_worth = self.initial_balance
        self.total_reward = 0
        self.num_trades = 0
        return obs, info


# -------------------------------
# Evaluate PPO Agent
# -------------------------------
def evaluate_ppo_agent(model, test_days, initial_balance=10000, decision_interval=5):
    rewards, net_worths, trades_list, returns = [], [], [], []

    for day_data, stock_name in test_days:
        env = PPOTradingEnv(day_data, stock_name, initial_balance)
        obs, _ = env.reset()
        done = False
        step_count = 0
        previous_action = 0

        while not done:
            if step_count % decision_interval == 0:
                action, _ = model.predict(obs, deterministic=True)
                previous_action = action
            else:
                action = previous_action

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        percent_return = ((env.net_worth - initial_balance) / initial_balance) * 100
        rewards.append(env.total_reward)
        net_worths.append(env.net_worth)
        trades_list.append(env.num_trades)
        returns.append(percent_return)

    eval_metrics = {
        "num_episodes": len(test_days),
        "avg_reward": np.mean(rewards),
        "avg_net_worth": np.mean(net_worths),
        "avg_trades": np.mean(trades_list),
        "avg_return": np.mean(returns),
        "win_rate": sum(1 for r in returns if r > 0) / len(returns) * 100,
        "best_return": np.max(returns),
        "worst_return": np.min(returns)
    }
    return eval_metrics


# -------------------------------
# PPO Training
# -------------------------------
def train_ppo():
    INITIAL_BALANCE = 10000
    DECISION_INTERVAL = 5
    EPISODES = 200
    TRAIN_RATIO = 0.7

    all_files = glob.glob('processed_data/*.csv')
    if not all_files:
        print("ERROR: No CSV files found in processed_data/")
        return

    train_days, test_days, splitter = load_and_split_data(all_files, train_ratio=TRAIN_RATIO)
    print(f"Training Days: {len(train_days)} | Test Days: {len(test_days)}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("ppo_output", f"run_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    rewards_dir = os.path.join(base_dir, "training_rewards")
    os.makedirs(rewards_dir, exist_ok=True)

    eval_dir = os.path.join(base_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)

    # Save split info
    pd.DataFrame([splitter.get_split_info()]).to_csv(
        os.path.join(base_dir, 'data_split_info.csv'),
        index=False
    )

    # --------------------------
    # Dashboard Logging File
    # --------------------------
    episode_metrics_file = os.path.join(base_dir, "episode_metrics.csv")

    pd.DataFrame(columns=[
        "episode",
        "stock",
        "reward",
        "net_worth",
        "return_pct",
        "trades",
        "win_rate"
    ]).to_csv(episode_metrics_file, index=False)

    dummy_day, dummy_stock = train_days[0]
    vec_env = DummyVecEnv([lambda: PPOTradingEnv(dummy_day, dummy_stock, INITIAL_BALANCE)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=0.0005,
        batch_size=64,
        n_steps=128,
        n_epochs=10,
        gamma=0.99,
        verbose=0
    )

    rewards_history = []
    eval_history = []

    for ep in range(EPISODES):
        day_data, stock_name = random.choice(train_days)

        env = PPOTradingEnv(day_data, stock_name, INITIAL_BALANCE)
        vec_env = DummyVecEnv([lambda: env])
        model.set_env(vec_env)

        model.learn(total_timesteps=len(day_data) * 20)

        metrics = evaluate_ppo_agent(
            model,
            [(day_data, stock_name)],
            INITIAL_BALANCE,
            DECISION_INTERVAL
        )

        rewards_history.append(metrics['avg_reward'])
        eval_history.append(metrics)

        # --------------------------
        # Log for Dashboard
        # --------------------------
        episode_row = {
            "episode": ep + 1,
            "stock": stock_name,
            "reward": metrics['avg_reward'],
            "net_worth": metrics['avg_net_worth'],
            "return_pct": metrics['avg_return'],
            "trades": metrics['avg_trades'],
            "win_rate": metrics['win_rate']
        }

        pd.DataFrame([episode_row]).to_csv(
            episode_metrics_file,
            mode='a',
            header=False,
            index=False
        )

        print(f"Episode {ep+1} | Stock: {stock_name} | "
              f"Reward: {metrics['avg_reward']:.2f} | "
              f"Net Worth: ${metrics['avg_net_worth']:.2f} | "
              f"Return: {metrics['avg_return']:.2f}% | "
              f"Trades: {metrics['avg_trades']} | "
              f"Win Rate: {metrics['win_rate']:.2f}%")

        if (ep + 1) % 50 == 0:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards_history, linewidth=2)
            plt.title(f"Training Rewards (Episodes 1-{ep+1})")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(rewards_dir, f'training_rewards_ep{ep+1}.png'))
            plt.close()

    # Final reward plot
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, linewidth=2)
    plt.title("Training Rewards Over All Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(rewards_dir, 'training_rewards_final.png'))
    plt.close()

    os.makedirs("ppo_models", exist_ok=True)
    model.save(os.path.join("ppo_models", "ppo_trading_model"))

    final_metrics = evaluate_ppo_agent(
        model,
        test_days,
        INITIAL_BALANCE,
        DECISION_INTERVAL
    )

    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(eval_dir, 'final_test_evaluation.csv'),
        index=False
    )

    print("\nFinal Evaluation on Test Set:")
    print(f"Avg Reward: {final_metrics['avg_reward']:.2f} | "
          f"Net Worth: ${final_metrics['avg_net_worth']:.2f} | "
          f"Avg Return: {final_metrics['avg_return']:.2f}% | "
          f"Avg Trades: {final_metrics['avg_trades']:.2f} | "
          f"Win Rate: {final_metrics['win_rate']:.2f}% | "
          f"Best Return: {final_metrics['best_return']:.2f}% | "
          f"Worst Return: {final_metrics['worst_return']:.2f}%")

    generate_performance_summary(
        pd.DataFrame([final_metrics]),
        output_dir=base_dir,
        enable_voice=False
    )


if __name__ == "__main__":
    train_ppo()