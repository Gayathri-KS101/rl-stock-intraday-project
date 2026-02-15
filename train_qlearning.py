import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# =====================================
# LOAD DATA
# =====================================

file_name = "reliance-minute.csv"   # Change if needed
file_path = os.path.join("processed_data", file_name)

data = pd.read_csv(file_path)
prices = data["close"].values   # Change to "Close" if needed


# =====================================
# PARAMETERS
# =====================================

n_states = 3      # 0=Down, 1=Up, 2=Flat
n_actions = 3     # 0=Hold, 1=Buy, 2=Sell

Q = np.zeros((n_states, n_actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.1

episodes = 50
initial_balance = 10000


# =====================================
# STATE FUNCTION
# =====================================

def get_state(price_diff):
    if price_diff > 0:
        return 1
    elif price_diff < 0:
        return 0
    else:
        return 2


# =====================================
# TRAINING
# =====================================

episode_rewards = []
episode_balances = []
episode_networths = []

for episode in range(episodes):

    balance = initial_balance
    shares_held = 0
    total_reward = 0

    for t in range(1, len(prices)-1):

        price_diff = prices[t] - prices[t-1]
        state = get_state(price_diff)

        # Epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[state])

        next_price_diff = prices[t+1] - prices[t]
        next_state = get_state(next_price_diff)

        reward = 0

        # ======================
        # ACTION LOGIC
        # ======================

        if action == 1:  # Buy
            if balance > prices[t]:
                shares_held += 1
                balance -= prices[t]

        elif action == 2:  # Sell
            if shares_held > 0:
                shares_held -= 1
                balance += prices[t]

        # Net worth calculation
        net_worth = balance + shares_held * prices[t]

        reward = net_worth - initial_balance

        # Q update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        total_reward += reward

    episode_rewards.append(total_reward)
    episode_balances.append(balance)
    episode_networths.append(net_worth)

    print(f"Episode {episode+1} | "
          f"Balance: {round(balance,2)} | "
          f"Net Worth: {round(net_worth,2)} | "
          f"Total Reward: {round(total_reward,2)}")


# =====================================
# SAVE OUTPUT
# =====================================

os.makedirs("output_qlearning", exist_ok=True)

results_df = pd.DataFrame({
    "Episode": range(1, episodes+1),
    "Final_Balance": episode_balances,
    "Final_NetWorth": episode_networths,
    "Total_Reward": episode_rewards
})

results_df.to_csv("output_qlearning/episode_metrics.csv", index=False)

print("\nSaved metrics to CSV.")


# =====================================
# PLOTS
# =====================================

# Reward Curve
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Reward Curve")
plt.savefig("output_qlearning/reward_curve.png")
plt.show()

# Net Worth Curve
plt.figure()
plt.plot(episode_networths)
plt.xlabel("Episode")
plt.ylabel("Net Worth")
plt.title("Net Worth Over Episodes")
plt.savefig("output_qlearning/networth_curve.png")
plt.show()

# Q-table Heatmap
plt.figure()
plt.imshow(Q)
plt.colorbar()
plt.xlabel("Actions (0=Hold, 1=Buy, 2=Sell)")
plt.ylabel("States (0=Down, 1=Up, 2=Flat)")
plt.title("Q-Table Heatmap")
plt.savefig("output_qlearning/q_table_heatmap.png")
plt.show()

print("\nFinal Q-Table:")
print(Q)

print("\nAll outputs saved in 'output_qlearning' folder.")
