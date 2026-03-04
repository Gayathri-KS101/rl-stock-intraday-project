import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_episode_results(csv_path, save_path=None):
    """
    Generate professional visualization for one episode.
    """

    df = pd.read_csv(csv_path)

    # Convert datetime column
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Identify buy/sell points
    buy_signals = df[df['Action'] == 1]
    sell_signals = df[df['Action'] == 0]

    plt.style.use('seaborn-v0_8-darkgrid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # ------------------ PRICE PLOT ------------------ #
    ax1.plot(df['DateTime'], df['Current_Price'], 
             label='Market Price', linewidth=2)

    ax1.scatter(buy_signals['DateTime'], buy_signals['Current_Price'],
                marker='^', s=100, label='Buy', alpha=0.8)

    ax1.scatter(sell_signals['DateTime'], sell_signals['Current_Price'],
                marker='v', s=100, label='Sell', alpha=0.8)

    ax1.set_title("Intraday Market Price with Agent Decisions", fontsize=14)
    ax1.set_ylabel("Price")
    ax1.legend()

    # ---------------- NET WORTH PLOT ---------------- #
    ax2.plot(df['DateTime'], df['Net_Worth'], 
             linewidth=2)

    ax2.set_title("Portfolio Net Worth Over Time", fontsize=14)
    ax2.set_ylabel("Net Worth")
    ax2.set_xlabel("Time")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()