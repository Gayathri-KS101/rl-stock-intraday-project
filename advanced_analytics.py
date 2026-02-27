"""
Advanced Analytics Module for DQN Stock Trading System

Provides three analytics classes:
    - EpisodeAdvancedAnalytics
    - TrainingAdvancedAnalytics
    - EvaluationAdvancedAnalytics

All outputs are saved to disk only. Terminal output is a single clean summary line.
Nothing in this module alters trading logic, agent behavior, or existing outputs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


# ============================================================================
# HELPERS
# ============================================================================

def _safe_sharpe(series: pd.Series) -> float:
    """Mean / std with guard against zero std."""
    std = series.std()
    return float(series.mean() / std) if std > 1e-10 else 0.0


def _max_drawdown(net_worth: np.ndarray) -> float:
    """Maximum drawdown as a fraction (negative value)."""
    peak = np.maximum.accumulate(net_worth)
    drawdown = (net_worth - peak) / np.where(peak == 0, 1, peak)
    return float(drawdown.min())


def _holding_durations(actions: pd.Series) -> dict:
    """
    Compute statistics on how long each position (Cash=0, Invest=1) is held
    consecutively.
    Returns dict with mean, std, min, max duration (in steps).
    """
    durations = []
    if len(actions) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
    current = actions.iloc[0]
    count = 1
    for a in actions.iloc[1:]:
        if a == current:
            count += 1
        else:
            durations.append(count)
            current = a
            count = 1
    durations.append(count)
    arr = np.array(durations)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def _profit_per_trade(df: pd.DataFrame) -> float:
    """
    Approximate profit per trade from net worth changes at position switches.
    Returns average net-worth delta at each Cash→Invest→Cash round trip.
    """
    switches = df[df['Action'] != df['Action'].shift()].copy()
    if len(switches) < 2:
        return 0.0
    deltas = switches['Net_Worth'].diff().dropna().values
    return float(np.mean(deltas)) if len(deltas) > 0 else 0.0


def _trade_frequency_per_hour(df: pd.DataFrame) -> float:
    """Number of position switches per hour of trading."""
    switches = int((df['Action'] != df['Action'].shift()).sum()) - 1
    switches = max(switches, 0)
    total_steps = len(df)
    # Assume 1 step = 1 minute
    hours = total_steps / 60.0
    return switches / hours if hours > 0 else 0.0


# ============================================================================
# EPISODE ANALYTICS
# ============================================================================

class EpisodeAdvancedAnalytics:
    """
    Computes and saves advanced per-episode analytics.

    Outputs (saved to <base_dir>/episode_analysis/):
        episode_XXX_summary.csv
        episode_XXX_statistics.csv
        episode_XXX_readable_summary.txt
        episode_XXX_advanced_plot.png
    """

    def __init__(self, base_dir: str):
        self.out_dir = os.path.join(base_dir, "episode_analysis")
        os.makedirs(self.out_dir, exist_ok=True)

    def process(self, df: pd.DataFrame, episode: int) -> None:
        """Run all episode analytics and save outputs."""
        summary = self._compute_summary(df, episode)
        stats = self._compute_statistics(df)
        self._save_summary_csv(summary, episode)
        self._save_statistics_csv(stats, episode)
        self._save_readable_txt(summary, stats, episode)
        self._save_advanced_plot(df, summary, episode)

    # ------------------------------------------------------------------
    def _compute_summary(self, df: pd.DataFrame, episode: int) -> dict:
        rewards = df['Reward']
        net_worth = df['Net_Worth'].values
        total_steps = len(df)

        # Exploration / Exploitation
        n_explore = df['Decision_Reason'].str.contains('Exploration').sum()
        n_exploit = total_steps - n_explore

        # Position switches
        switches = int((df['Action'] != df['Action'].shift()).sum()) - 1
        switches = max(switches, 0)

        return {
            "episode": episode,
            "total_steps": total_steps,
            "avg_reward_per_step": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "total_reward": float(rewards.sum()),
            "sharpe_like_ratio": _safe_sharpe(rewards),
            "max_drawdown": _max_drawdown(net_worth),
            "net_worth_volatility": float(np.std(net_worth)),
            "avg_q_cash": float(df['Q_Cash'].mean()),
            "avg_q_invest": float(df['Q_Invest'].mean()),
            "exploration_pct": float(n_explore / total_steps * 100),
            "exploitation_pct": float(n_exploit / total_steps * 100),
            "position_switches": switches,
            "profit_per_trade": _profit_per_trade(df),
            "trade_frequency_per_hour": _trade_frequency_per_hour(df),
            "initial_net_worth": float(df['Net_Worth'].iloc[0]),
            "final_net_worth": float(df['Net_Worth'].iloc[-1]),
            "percent_return": float(
                (df['Net_Worth'].iloc[-1] - df['Net_Worth'].iloc[0])
                / df['Net_Worth'].iloc[0] * 100
            ),
        }

    def _compute_statistics(self, df: pd.DataFrame) -> dict:
        hold_stats = _holding_durations(df['Action'])
        return {
            "holding_duration_mean": hold_stats["mean"],
            "holding_duration_std": hold_stats["std"],
            "holding_duration_min": hold_stats["min"],
            "holding_duration_max": hold_stats["max"],
            "reward_min": float(df['Reward'].min()),
            "reward_max": float(df['Reward'].max()),
            "reward_median": float(df['Reward'].median()),
            "q_cash_std": float(df['Q_Cash'].std()),
            "q_invest_std": float(df['Q_Invest'].std()),
            "cash_action_count": int((df['Action'] == 0).sum()),
            "invest_action_count": int((df['Action'] == 1).sum()),
        }

    # ------------------------------------------------------------------
    def _save_summary_csv(self, summary: dict, episode: int) -> None:
        pd.DataFrame([summary]).to_csv(
            os.path.join(self.out_dir, f"episode_{episode:03d}_summary.csv"),
            index=False
        )

    def _save_statistics_csv(self, stats: dict, episode: int) -> None:
        pd.DataFrame([stats]).to_csv(
            os.path.join(self.out_dir, f"episode_{episode:03d}_statistics.csv"),
            index=False
        )

    def _save_readable_txt(self, summary: dict, stats: dict, episode: int) -> None:
        lines = [
            f"EPISODE {episode} — ADVANCED ANALYTICS SUMMARY",
            "=" * 60,
            f"  Total Steps            : {summary['total_steps']}",
            f"  Avg Reward / Step      : {summary['avg_reward_per_step']:.6f}",
            f"  Std Reward             : {summary['std_reward']:.6f}",
            f"  Total Reward           : {summary['total_reward']:.6f}",
            f"  Sharpe-Like Ratio      : {summary['sharpe_like_ratio']:.4f}",
            f"  Max Drawdown           : {summary['max_drawdown']*100:.2f}%",
            f"  Net Worth Volatility   : ${summary['net_worth_volatility']:.2f}",
            "",
            "  Q-VALUE AVERAGES",
            f"    Avg Q_Cash           : {summary['avg_q_cash']:.4f}  (std {stats['q_cash_std']:.4f})",
            f"    Avg Q_Invest         : {summary['avg_q_invest']:.4f}  (std {stats['q_invest_std']:.4f})",
            "",
            "  EXPLORATION / EXPLOITATION",
            f"    Exploration          : {summary['exploration_pct']:.2f}%",
            f"    Exploitation         : {summary['exploitation_pct']:.2f}%",
            "",
            "  POSITION MANAGEMENT",
            f"    Position Switches    : {summary['position_switches']}",
            f"    Holding Duration     : mean {stats['holding_duration_mean']:.1f} | "
            f"std {stats['holding_duration_std']:.1f} | "
            f"min {stats['holding_duration_min']} | "
            f"max {stats['holding_duration_max']} steps",
            f"    Profit / Trade       : ${summary['profit_per_trade']:.4f}",
            f"    Trade Freq / Hour    : {summary['trade_frequency_per_hour']:.2f}",
            "",
            "  PERFORMANCE",
            f"    Initial Net Worth    : ${summary['initial_net_worth']:.2f}",
            f"    Final Net Worth      : ${summary['final_net_worth']:.2f}",
            f"    Percent Return       : {summary['percent_return']:.2f}%",
            "=" * 60,
        ]
        path = os.path.join(self.out_dir, f"episode_{episode:03d}_readable_summary.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _save_advanced_plot(self, df: pd.DataFrame, summary: dict, episode: int) -> None:
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        steps = df['Step'].values

        # 1. Net worth
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(steps, df['Net_Worth'].values, 'b-', linewidth=1.5, label='Net Worth')
        invest_mask = df['Action'] == 1
        cash_mask = df['Action'] == 0
        if invest_mask.any():
            ax1.scatter(df.loc[invest_mask, 'Step'], df.loc[invest_mask, 'Net_Worth'],
                        color='green', marker='^', s=60, zorder=5, label='Invest', alpha=0.7)
        if cash_mask.any():
            ax1.scatter(df.loc[cash_mask, 'Step'], df.loc[cash_mask, 'Net_Worth'],
                        color='red', marker='v', s=60, zorder=5, label='Cash', alpha=0.7)
        ax1.axhline(y=summary['initial_net_worth'], color='gray', linestyle='--', linewidth=1)
        ax1.set_title(f'Episode {episode} — Net Worth with Drawdown Overlay', fontweight='bold')
        ax1.set_ylabel('Net Worth ($)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        # Shade drawdown regions
        nw = df['Net_Worth'].values
        peak = np.maximum.accumulate(nw)
        ax1.fill_between(steps, nw, peak, where=(nw < peak), color='red', alpha=0.15, label='Drawdown')

        # 2. Q-values over time
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(steps, df['Q_Cash'].values, label='Q_Cash', color='red', linewidth=1)
        ax2.plot(steps, df['Q_Invest'].values, label='Q_Invest', color='green', linewidth=1)
        ax2.set_title('Q-Values Over Time', fontweight='bold')
        ax2.set_ylabel('Q-Value')
        ax2.set_xlabel('Step')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. Reward distribution
        ax3 = fig.add_subplot(gs[1, 1])
        rewards = df['Reward'].values
        ax3.hist(rewards, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
        ax3.axvline(x=rewards.mean(), color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean: {rewards.mean():.4f}')
        ax3.set_title('Reward Distribution', fontweight='bold')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. Action timeline (heatmap-style)
        ax4 = fig.add_subplot(gs[2, 0])
        actions = df['Action'].values.reshape(1, -1)
        ax4.imshow(actions, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[steps[0], steps[-1], 0, 1])
        ax4.set_title('Action Timeline (Green=Invest, Red=Cash)', fontweight='bold')
        ax4.set_xlabel('Step')
        ax4.set_yticks([])

        # 5. Holding duration bar chart
        ax5 = fig.add_subplot(gs[2, 1])
        hold_stats = _holding_durations(df['Action'])
        categories = ['Mean', 'Std', 'Min', 'Max']
        values = [hold_stats['mean'], hold_stats['std'], hold_stats['min'], hold_stats['max']]
        bars = ax5.bar(categories, values, color=['steelblue', 'orange', 'green', 'red'], alpha=0.8)
        ax5.set_title('Holding Duration Stats (steps)', fontweight='bold')
        ax5.set_ylabel('Steps')
        for bar, val in zip(bars, values):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Episode {episode} Advanced Analytics', fontsize=15, fontweight='bold', y=1.01)
        path = os.path.join(self.out_dir, f"episode_{episode:03d}_advanced_plot.png")
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# TRAINING ANALYTICS
# ============================================================================

class TrainingAdvancedAnalytics:
    """
    Computes and saves training-level analytics.

    Outputs (saved to <base_dir>/training_analysis/):
        training_full_metrics.csv
        training_summary.csv
        training_summary.txt
        reward_curve.png
        reward_curve_moving_avg_20.png
        return_distribution.png
        trade_distribution.png
    """

    def __init__(self, base_dir: str):
        self.out_dir = os.path.join(base_dir, "training_analysis")
        os.makedirs(self.out_dir, exist_ok=True)

    def process(self, metrics_df: pd.DataFrame) -> None:
        """Run all training analytics and save outputs."""
        enriched = self._enrich_metrics(metrics_df)
        summary = self._compute_summary(enriched)
        self._save_full_metrics(enriched)
        self._save_summary_csv(summary)
        self._save_summary_txt(summary)
        self._plot_reward_curve(enriched)
        self._plot_reward_moving_avg(enriched)
        self._plot_return_distribution(enriched)
        self._plot_trade_distribution(enriched)

    # ------------------------------------------------------------------
    def _enrich_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rolling_mean_20'] = df['Total_Reward'].rolling(window=20, min_periods=1).mean()
        df['rolling_volatility_20'] = df['Total_Reward'].rolling(window=20, min_periods=1).std()
        return df

    def _compute_summary(self, df: pd.DataFrame) -> dict:
        best_idx = df['Percent_Return'].idxmax()
        worst_idx = df['Percent_Return'].idxmin()
        return {
            "total_episodes": int(len(df)),
            "global_avg_reward": float(df['Total_Reward'].mean()),
            "global_std_reward": float(df['Total_Reward'].std()),
            "global_avg_return": float(df['Percent_Return'].mean()),
            "global_std_return": float(df['Percent_Return'].std()),
            "best_episode": int(df.loc[best_idx, 'Episode']),
            "best_episode_return": float(df.loc[best_idx, 'Percent_Return']),
            "worst_episode": int(df.loc[worst_idx, 'Episode']),
            "worst_episode_return": float(df.loc[worst_idx, 'Percent_Return']),
            "avg_trades_per_episode": float(df['Number_of_Trades'].mean()),
            "win_rate": float((df['Percent_Return'] > 0).sum() / len(df) * 100),
            "avg_final_net_worth": float(df['Final_Net_Worth'].mean()),
        }

    # ------------------------------------------------------------------
    def _save_full_metrics(self, df: pd.DataFrame) -> None:
        df.to_csv(os.path.join(self.out_dir, "training_full_metrics.csv"), index=False)

    def _save_summary_csv(self, summary: dict) -> None:
        pd.DataFrame([summary]).to_csv(
            os.path.join(self.out_dir, "training_summary.csv"), index=False)

    def _save_summary_txt(self, summary: dict) -> None:
        lines = [
            "TRAINING ADVANCED ANALYTICS SUMMARY",
            "=" * 60,
            f"  Total Episodes           : {summary['total_episodes']}",
            f"  Global Avg Reward        : {summary['global_avg_reward']:.4f}",
            f"  Global Std Reward        : {summary['global_std_reward']:.4f}",
            f"  Global Avg Return        : {summary['global_avg_return']:.2f}%",
            f"  Global Std Return        : {summary['global_std_return']:.2f}%",
            f"  Win Rate                 : {summary['win_rate']:.2f}%",
            f"  Best Episode             : #{summary['best_episode']} ({summary['best_episode_return']:.2f}%)",
            f"  Worst Episode            : #{summary['worst_episode']} ({summary['worst_episode_return']:.2f}%)",
            f"  Avg Trades / Episode     : {summary['avg_trades_per_episode']:.1f}",
            f"  Avg Final Net Worth      : ${summary['avg_final_net_worth']:.2f}",
            "=" * 60,
        ]
        with open(os.path.join(self.out_dir, "training_summary.txt"), "w") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    def _plot_reward_curve(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Episode'], df['Total_Reward'], linewidth=1.2, color='steelblue', alpha=0.8)
        ax.set_title('Training Reward Curve', fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "reward_curve.png"), dpi=130, bbox_inches='tight')
        plt.close(fig)

    def _plot_reward_moving_avg(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Episode'], df['Total_Reward'], linewidth=0.8, color='steelblue', alpha=0.4, label='Raw')
        ax.plot(df['Episode'], df['rolling_mean_20'], linewidth=2, color='orange', label='20-ep Moving Avg')
        if 'rolling_volatility_20' in df.columns:
            upper = df['rolling_mean_20'] + df['rolling_volatility_20']
            lower = df['rolling_mean_20'] - df['rolling_volatility_20']
            ax.fill_between(df['Episode'], lower, upper, alpha=0.2, color='orange', label='±1 Std')
        ax.set_title('Training Reward — 20-Episode Moving Average', fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "reward_curve_moving_avg_20.png"), dpi=130, bbox_inches='tight')
        plt.close(fig)

    def _plot_return_distribution(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Percent_Return'], bins=40, color='steelblue', edgecolor='black', alpha=0.8)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, label='Break-even')
        ax.axvline(x=df['Percent_Return'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean: {df['Percent_Return'].mean():.2f}%")
        ax.set_title('Episode Return Distribution', fontsize=13, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "return_distribution.png"), dpi=130, bbox_inches='tight')
        plt.close(fig)

    def _plot_trade_distribution(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Number_of_Trades'], bins=30, color='darkorange', edgecolor='black', alpha=0.8)
        ax.axvline(x=df['Number_of_Trades'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean: {df['Number_of_Trades'].mean():.1f}")
        ax.set_title('Trade Count Distribution per Episode', fontsize=13, fontweight='bold')
        ax.set_xlabel('Number of Trades')
        ax.set_ylabel('Count')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "trade_distribution.png"), dpi=130, bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# EVALUATION ANALYTICS
# ============================================================================

class EvaluationAdvancedAnalytics:
    """
    Computes and saves evaluation-level analytics.

    Outputs (saved to <base_dir>/evaluation_analysis/):
        evaluation_summary.csv
        evaluation_readable_summary.txt
        evaluation_distribution.png
    """

    def __init__(self, base_dir: str):
        self.out_dir = os.path.join(base_dir, "evaluation_analysis")
        os.makedirs(self.out_dir, exist_ok=True)

    def process(self, eval_df: pd.DataFrame) -> None:
        """
        Run all evaluation analytics and save outputs.

        eval_df must have columns produced by the evaluation loop, e.g.:
            num_episodes, avg_reward, std_reward, avg_return, std_return,
            avg_trades, win_rate, best_return, worst_return, final_avg_net_worth, episode
        """
        summary = self._compute_summary(eval_df)
        self._save_summary_csv(summary)
        self._save_readable_txt(summary, eval_df)
        self._save_distribution_plot(eval_df)

    # ------------------------------------------------------------------
    def _compute_summary(self, df: pd.DataFrame) -> dict:
        """Aggregate across all periodic evaluation checkpoints."""
        return {
            "num_eval_checkpoints": int(len(df)),
            "overall_avg_return": float(df['avg_return'].mean()),
            "overall_std_return": float(df['std_return'].mean()),
            "overall_win_rate": float(df['win_rate'].mean()),
            "best_checkpoint_return": float(df['best_return'].max()),
            "worst_checkpoint_return": float(df['worst_return'].min()),
            "avg_trades_per_episode": float(df['avg_trades'].mean()),
            "avg_final_net_worth": float(df['final_avg_net_worth'].mean()),
            "best_checkpoint_episode": int(df.loc[df['avg_return'].idxmax(), 'episode'])
                if 'episode' in df.columns else -1,
        }

    def _save_summary_csv(self, summary: dict) -> None:
        pd.DataFrame([summary]).to_csv(
            os.path.join(self.out_dir, "evaluation_summary.csv"), index=False)

    def _save_readable_txt(self, summary: dict, df: pd.DataFrame) -> None:
        lines = [
            "EVALUATION ADVANCED ANALYTICS SUMMARY",
            "=" * 60,
            f"  Eval Checkpoints         : {summary['num_eval_checkpoints']}",
            f"  Overall Avg Return       : {summary['overall_avg_return']:.2f}%",
            f"  Overall Std Return       : {summary['overall_std_return']:.2f}%",
            f"  Overall Win Rate         : {summary['overall_win_rate']:.2f}%",
            f"  Best Checkpoint Return   : {summary['best_checkpoint_return']:.2f}%",
            f"  Worst Checkpoint Return  : {summary['worst_checkpoint_return']:.2f}%",
            f"  Avg Trades / Episode     : {summary['avg_trades_per_episode']:.1f}",
            f"  Avg Final Net Worth      : ${summary['avg_final_net_worth']:.2f}",
            f"  Best Checkpoint Episode  : #{summary['best_checkpoint_episode']}",
            "=" * 60,
            "",
            "PER-CHECKPOINT DETAIL:",
            "-" * 60,
        ]
        for _, row in df.iterrows():
            ep_str = f"Ep {int(row['episode'])}" if 'episode' in df.columns else ""
            lines.append(
                f"  {ep_str:8s} | Avg Return: {row['avg_return']:6.2f}% "
                f"| Win Rate: {row['win_rate']:5.1f}% "
                f"| Avg Trades: {row['avg_trades']:5.1f}"
            )
        lines.append("=" * 60)
        with open(os.path.join(self.out_dir, "evaluation_readable_summary.txt"), "w") as f:
            f.write("\n".join(lines))

    def _save_distribution_plot(self, df: pd.DataFrame) -> None:
        has_episode = 'episode' in df.columns
        ncols = 3
        fig, axes = plt.subplots(1, ncols, figsize=(15, 5))

        # 1. Return distribution across checkpoints
        axes[0].bar(
            range(len(df)),
            df['avg_return'].values,
            color=['green' if v >= 0 else 'red' for v in df['avg_return'].values],
            alpha=0.8, edgecolor='black'
        )
        if has_episode:
            axes[0].set_xticks(range(len(df)))
            axes[0].set_xticklabels([str(int(e)) for e in df['episode']], rotation=45, fontsize=8)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0].set_title('Avg Return per Eval Checkpoint', fontweight='bold')
        axes[0].set_xlabel('Training Episode')
        axes[0].set_ylabel('Return (%)')
        axes[0].grid(True, alpha=0.3, axis='y')

        # 2. Win rate across checkpoints
        axes[1].plot(
            df['episode'].values if has_episode else range(len(df)),
            df['win_rate'].values,
            marker='o', linewidth=2, color='steelblue'
        )
        axes[1].axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50%')
        axes[1].set_title('Win Rate per Eval Checkpoint', fontweight='bold')
        axes[1].set_xlabel('Training Episode')
        axes[1].set_ylabel('Win Rate (%)')
        axes[1].set_ylim(0, 100)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # 3. Avg net worth per checkpoint
        axes[2].plot(
            df['episode'].values if has_episode else range(len(df)),
            df['final_avg_net_worth'].values,
            marker='s', linewidth=2, color='darkorange'
        )
        axes[2].axhline(
            y=df['final_avg_net_worth'].iloc[0] if len(df) > 0 else 10000,
            color='gray', linestyle='--', linewidth=1, label='Baseline'
        )
        axes[2].set_title('Avg Final Net Worth per Checkpoint', fontweight='bold')
        axes[2].set_xlabel('Training Episode')
        axes[2].set_ylabel('Net Worth ($)')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle('Evaluation Performance Across Training', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "evaluation_distribution.png"), dpi=130, bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# UNIFIED FACADE
# ============================================================================

class AdvancedAnalytics:
    """
    Single entry-point used by train_chronological.py.

    Usage:
        analytics = AdvancedAnalytics(base_dir)
        analytics.process_episode(log_df, episode)        # inside episode loop
        analytics.process_training(metrics_df)            # after training loop
        analytics.process_evaluation(eval_history_df)     # after training loop
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._episode = EpisodeAdvancedAnalytics(base_dir)
        self._training = TrainingAdvancedAnalytics(base_dir)
        self._evaluation = EvaluationAdvancedAnalytics(base_dir)

    def process_episode(self, log_df: pd.DataFrame, episode: int) -> None:
        """Save advanced analytics for one episode. Silent — no terminal spam."""
        self._episode.process(log_df, episode)
        print(f"Advanced analytics saved to {os.path.join(self.base_dir, 'episode_analysis')}")

    def process_training(self, metrics_df: pd.DataFrame) -> None:
        """Save training-level analytics. Prints one summary line."""
        self._training.process(metrics_df)
        print(f"Advanced analytics saved to {os.path.join(self.base_dir, 'training_analysis')}")

    def process_evaluation(self, eval_history: list) -> None:
        """
        Save evaluation analytics from the eval_history list of dicts.
        Prints one summary line.
        """
        if not eval_history:
            return
        eval_df = pd.DataFrame(eval_history)
        self._evaluation.process(eval_df)
        print(f"Advanced analytics saved to {os.path.join(self.base_dir, 'evaluation_analysis')}")