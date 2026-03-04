"""
A2C Training Script with Chronological Train/Test Split
Hybrid Version — Keeps Logging + Analytics
"""

import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from simplified_environment import SimplifiedStockTradingEnv
from simplified_agent import (
    ActorCritic,
    create_episode_logger,
    log_step,
    save_episode_log,
    print_log_preview,
    analyze_episode_decisions,
)
from visualizations import plot_episode_results
from data_splitter import load_and_split_data
from advanced_analytics import AdvancedAnalytics
from performance_commentary import generate_performance_summary


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def train_agent():

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--log-episodes', type=int, default=10)
    parser.add_argument('--data-dir', type=str, default='processed_data')
    parser.add_argument('--initial-balance', type=float, default=10000)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--decision-interval', type=int, default=5)
    args = parser.parse_args()

    print("=" * 100)
    print("LOADING DATA WITH CHRONOLOGICAL SPLIT")
    print("=" * 100)

    all_files = glob.glob(f'{args.data_dir}/*.csv')

    train_days, test_days, splitter = load_and_split_data(
        all_files,
        train_ratio=args.train_ratio
    )

    dummy_day_data, dummy_stock = train_days[0]
    dummy_env = SimplifiedStockTradingEnv(
        dummy_day_data,
        stock_name=dummy_stock,
        initial_balance=args.initial_balance
    )

    input_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n

    print(f"Input Dim: {input_dim}")
    print(f"Action Dim: {action_dim}")

    policy_net = ActorCritic(input_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("output_data", f"a2c_run_{timestamp}")
    log_dir = os.path.join(base_dir, "episode_logs_csv")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    advanced_analytics = AdvancedAnalytics(base_dir)

    rewards_history = []
    episode_metrics = {
        'Episode': [],
        'Total_Reward': [],
        'Final_Net_Worth': [],
        'Percent_Return': [],
        'Number_of_Trades': []
    }

    print("=" * 100)
    print("STARTING A2C TRAINING")
    print("=" * 100)

    for episode in range(args.episodes):

        day_data, stock_name = random.choice(train_days)
        env = SimplifiedStockTradingEnv(
            day_data,
            stock_name=stock_name,
            initial_balance=args.initial_balance
        )

        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        previous_action = 0
        minute_log = []

        should_log = (args.log_episodes > 0) and ((episode + 1) % args.log_episodes == 0)
        if should_log:
            episode_logger = create_episode_logger()

        while not done:

            if step_count % args.decision_interval == 0:

                state_t = torch.FloatTensor(state).unsqueeze(0)

                logits, state_value = policy_net(state_t)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.item()

                previous_action = action

            else:
                action = previous_action
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    logits, _ = policy_net(state_t)
                    probs = torch.softmax(logits, dim=-1)

            next_state, reward, done, _, info = env.step(action)
            minute_log.append(info)
            total_reward += reward

            if should_log:
                log_step(
                    episode_logger,
                    episode + 1,
                    step_count,
                    env,
                    state,
                    action,
                    probs.detach().numpy()[0],
                    0.0,
                    reward,
                    False
                )

            if step_count % args.decision_interval == 0:

                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value = policy_net(next_state_t)

                if done:
                    target = torch.tensor([[reward]], dtype=torch.float32)
                else:
                    target = reward + args.gamma * next_value.detach()

                advantage = target - state_value

                # Normalize advantage
                #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                actor_loss = -(log_prob * advantage.detach())
                critic_loss = advantage.pow(2).mean()

                entropy = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            state = next_state
            step_count += 1

        percent_return = ((env.net_worth - args.initial_balance) / args.initial_balance) * 100
        minute_df = pd.DataFrame(minute_log)
        minute_df.to_csv(os.path.join(base_dir, f"episode_{episode+1}_minute_log.csv"), index=False)

        episode_metrics['Episode'].append(episode + 1)
        episode_metrics['Total_Reward'].append(total_reward)
        episode_metrics['Final_Net_Worth'].append(env.net_worth)
        episode_metrics['Percent_Return'].append(percent_return)
        episode_metrics['Number_of_Trades'].append(env.num_trades)

        rewards_history.append(total_reward)

        print(f"Episode {episode+1}: "
              f"Reward: {total_reward:.2f} | "
              f"Net Worth: ${env.net_worth:.2f} | "
              f"Return: {percent_return:.2f}% | "
              f"Trades: {env.num_trades}")

        if should_log:
            log_df, filepath = save_episode_log(
                episode_logger,
                episode + 1,
                output_dir=log_dir
            )

            print_log_preview(log_df)
            analyze_episode_decisions(log_df)
            advanced_analytics.process_episode(log_df, episode + 1)

            # 🔥 Generate Visualization
            plot_episode_results(
                filepath,
                os.path.join(base_dir, f"episode_{episode+1}_visualization.png")
            )

    print("=" * 100)
    print("TRAINING COMPLETE")
    print("=" * 100)

    torch.save(policy_net.state_dict(),
               os.path.join(base_dir, 'trained_model_a2c.pth'))

    metrics_df = pd.DataFrame(episode_metrics)
    metrics_df.to_csv(os.path.join(base_dir, 'episode_metrics.csv'), index=False)

    advanced_analytics.process_training(metrics_df)

    print("A2C model saved successfully.")


if __name__ == "__main__":
    train_agent()