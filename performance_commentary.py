"""
Performance Commentary Engine
==============================
Generates professional quant-style written commentary from evaluation results
and optionally narrates it aloud using pyttsx3 (fully offline).

Public API:
    generate_performance_summary(eval_df, output_dir) -> str
    narrate_performance(summary_text, enable_voice=True)
"""

import os
import threading
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================================
# INTERNAL METRIC COMPUTATIONS
# ============================================================================

def _classify_win_rate(win_rate: float) -> str:
    if win_rate >= 65:
        return "strong"
    if win_rate >= 55:
        return "above-average"
    if win_rate >= 50:
        return "marginal"
    return "sub-50"


def _classify_volatility(std_return: float) -> str:
    if std_return < 0.5:
        return "low"
    if std_return < 1.5:
        return "moderate"
    if std_return < 3.0:
        return "elevated"
    return "high"


def _classify_tail_risk(worst_return: float) -> str:
    if worst_return > -1.0:
        return "contained"
    if worst_return > -3.0:
        return "moderate"
    if worst_return > -6.0:
        return "material"
    return "severe"


def _classify_trade_aggression(avg_trades: float) -> str:
    if avg_trades < 5:
        return "passive"
    if avg_trades < 15:
        return "moderate"
    if avg_trades < 30:
        return "active"
    return "hyper-active"


def _stability_trend(eval_df: pd.DataFrame) -> str:
    """Describe trend of avg_return across checkpoints if available."""
    if "avg_return" not in eval_df.columns or len(eval_df) < 3:
        return None
    returns = eval_df["avg_return"].values
    # Simple linear slope via polyfit
    x = np.arange(len(returns))
    slope = float(np.polyfit(x, returns, 1)[0])
    if slope > 0.05:
        return "improving"
    if slope < -0.05:
        return "deteriorating"
    return "stable"


def _compute_sharpe(avg_return: float, std_return: float) -> float:
    return avg_return / std_return if std_return > 1e-10 else 0.0


def _compute_profit_factor(avg_return: float, worst_return: float) -> float:
    """Simplified profit factor proxy: avg gain / worst loss magnitude."""
    if abs(worst_return) < 1e-10:
        return float("inf") if avg_return > 0 else 0.0
    return avg_return / abs(worst_return)


# ============================================================================
# COMMENTARY BUILDER
# ============================================================================

def _build_commentary(metrics: dict, eval_df: pd.DataFrame) -> str:
    """
    Compose a multi-paragraph quant-style performance report.
    All language is generated programmatically from metric thresholds.
    """

    wr        = metrics["win_rate"]
    avg_ret   = metrics["avg_return"]
    std_ret   = metrics["std_return"]
    best_ret  = metrics["best_return"]
    worst_ret = metrics["worst_return"]
    avg_trades = metrics["avg_trades"]
    sharpe    = metrics["sharpe"]
    pf        = metrics["profit_factor"]
    n_eps     = metrics["num_episodes"]

    wr_class   = _classify_win_rate(wr)
    vol_class  = _classify_volatility(std_ret)
    tail_class = _classify_tail_risk(worst_ret)
    trade_class = _classify_trade_aggression(avg_trades)
    trend      = _stability_trend(eval_df)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Section 1: Header ────────────────────────────────────────────────────
    s1 = f"""\
AGENT PERFORMANCE COMMENTARY
Generated: {timestamp}
Evaluation Episodes: {n_eps}
{"=" * 72}
"""

    # ── Section 2: Win Rate ───────────────────────────────────────────────────
    if wr_class == "strong":
        wr_sentence = (
            f"The agent demonstrates strong directional accuracy, achieving a win rate of "
            f"{wr:.1f}% across {n_eps} evaluation episodes. This level of consistency "
            f"suggests the policy has captured a meaningful market edge."
        )
    elif wr_class == "above-average":
        wr_sentence = (
            f"The agent records an above-average win rate of {wr:.1f}% over {n_eps} episodes. "
            f"While directional bias is positive, the margin is not yet sufficient to confirm "
            f"structural alpha without further out-of-sample validation."
        )
    elif wr_class == "marginal":
        wr_sentence = (
            f"The win rate of {wr:.1f}% across {n_eps} episodes is marginally above the random "
            f"baseline of 50%. The edge is statistically weak and may not be robust under "
            f"regime shifts or transaction cost sensitivity analysis."
        )
    else:
        wr_sentence = (
            f"The agent's win rate of {wr:.1f}% across {n_eps} episodes falls below the break-even "
            f"threshold. The policy is currently net-negative in directional accuracy, indicating "
            f"insufficient learned signal or reward misalignment."
        )

    s2 = f"""\
1. WIN RATE AND DIRECTIONAL ACCURACY
{"-" * 72}
{wr_sentence}
"""

    # ── Section 3: Return Profile ─────────────────────────────────────────────
    ret_direction = "positive" if avg_ret > 0 else "negative"
    ret_magnitude = "modest" if abs(avg_ret) < 0.5 else ("material" if abs(avg_ret) < 2.0 else "significant")

    s3 = f"""\
2. RETURN PROFILE
{"-" * 72}
The agent generates a {ret_magnitude} {ret_direction} average return of {avg_ret:.2f}% per episode, \
with a return standard deviation of {std_ret:.2f}%. The resulting Sharpe-like ratio of {sharpe:.3f} \
{'indicates favorable risk-adjusted performance' if sharpe > 0.5 else 'reflects limited compensation for variance taken'}.

The best observed episode return was {best_ret:.2f}%, while the worst was {worst_ret:.2f}%, \
yielding a return range of {best_ret - worst_ret:.2f} percentage points. \
{'This range is consistent with disciplined position management.' if (best_ret - worst_ret) < 5 else 'This wide range indicates high dispersion and inconsistent episode outcomes.'}
"""

    # ── Section 4: Volatility and Risk Exposure ───────────────────────────────
    vol_desc = {
        "low":      "tightly clustered around the mean, suggesting consistent execution with limited variance.",
        "moderate": "consistent with typical intraday return distributions, indicating acceptable regime stability.",
        "elevated": "materially dispersed, indicating variable episode outcomes driven by market sensitivity or policy instability.",
        "high":     "severely dispersed, raising concerns about policy convergence and exposure to adverse market conditions.",
    }[vol_class]

    s4 = f"""\
3. VOLATILITY AND RISK EXPOSURE
{"-" * 72}
Return volatility is classified as {vol_class} (std: {std_ret:.2f}%). Episode outcomes are {vol_desc}

The profit factor proxy — defined as the ratio of mean return to worst-case loss magnitude — \
stands at {pf:.3f}. {'A value above 1.0 indicates that average gains outpace worst-case losses, a necessary condition for long-run viability.' if pf >= 1.0 else 'A value below 1.0 signals that worst-case losses exceed average gains, which is unsustainable without structural improvement.'}
"""

    # ── Section 5: Drawdown and Tail Risk ─────────────────────────────────────
    tail_desc = {
        "contained": "Tail risk is contained; worst-case episodes remain within acceptable loss bounds.",
        "moderate":  "Moderate tail exposure is present. The agent is vulnerable to occasional drawdowns that may test capital preservation under live conditions.",
        "material":  "Material tail risk is evident. Drawdown events of this magnitude suggest the policy has not adequately penalized catastrophic loss scenarios during training.",
        "severe":    "Severe tail risk is observed. The worst-case return implies significant capital destruction risk. The reward structure and exploration policy require review before deployment.",
    }[tail_class]

    s5 = f"""\
4. DRAWDOWN AND TAIL-RISK ANALYSIS
{"-" * 72}
Worst observed episode return: {worst_ret:.2f}%. {tail_desc}

{'The asymmetry between best and worst returns (' + f"{best_ret:.2f}% / {worst_ret:.2f}%" + ') is acceptable and reflects normal intraday variance.' if abs(best_ret) > abs(worst_ret) else 'The downside exceeds the upside in absolute terms, indicating a negatively skewed return distribution. This profile requires attention before live deployment.'}
"""

    # ── Section 6: Trade Aggressiveness ───────────────────────────────────────
    trade_desc = {
        "passive":    "The agent exhibits low trade frequency, consistent with a conservative positioning strategy. While this reduces transaction cost exposure, it may also limit the agent's ability to capitalize on short-lived intraday opportunities.",
        "moderate":   "Trade frequency is moderate, reflecting a balanced decision cadence. The agent appears to distinguish between high-conviction signals and noise effectively.",
        "active":     "High trade frequency is observed, which increases transaction cost sensitivity. Performance metrics may deteriorate materially when realistic slippage and commission models are applied.",
        "hyper-active": "The agent exhibits hyper-active trading behavior. At this trade frequency, frictional costs will likely dominate any gross return edge. Reward shaping to penalize unnecessary position changes is strongly recommended.",
    }[trade_class]

    s6 = f"""\
5. TRADE AGGRESSIVENESS AND FREQUENCY
{"-" * 72}
Average trades per episode: {avg_trades:.1f} (classification: {trade_class}). {trade_desc}
"""

    # ── Section 7: Stability Trend ────────────────────────────────────────────
    if trend is not None:
        trend_desc = {
            "improving":     "Evaluation performance shows an improving trend across training checkpoints. The agent continues to refine its policy and demonstrates positive learning momentum.",
            "stable":        "Performance is stable across evaluation checkpoints, indicating policy convergence. The agent shows neither meaningful improvement nor regression in out-of-sample conditions.",
            "deteriorating": "A deteriorating trend is observed across evaluation checkpoints. This pattern is consistent with policy overfitting to the training distribution or reward hacking behavior. Further regularization or data augmentation is advised.",
        }[trend]
        s7 = f"""\
6. STABILITY TREND ACROSS TRAINING
{"-" * 72}
{trend_desc}
"""
    else:
        s7 = f"""\
6. STABILITY TREND ACROSS TRAINING
{"-" * 72}
Insufficient evaluation checkpoints to assess trend (minimum 3 required).
"""

    # ── Section 8: Overall Robustness Verdict ────────────────────────────────
    score = 0
    if wr >= 55:       score += 2
    elif wr >= 50:     score += 1
    if avg_ret > 0:    score += 2
    if sharpe > 0.5:   score += 2
    if pf >= 1.0:      score += 1
    if tail_class in ("contained", "moderate"): score += 1
    if trade_class in ("passive", "moderate"):  score += 1
    if trend == "improving": score += 1

    if score >= 9:
        verdict = "ROBUST — The agent demonstrates consistent, risk-adjusted performance suitable for further validation under live market conditions."
    elif score >= 6:
        verdict = "DEVELOPING — Core signals are present but require additional training stability and risk reduction before deployment consideration."
    elif score >= 4:
        verdict = "EARLY STAGE — The policy shows partial learning but lacks the consistency and risk control required for reliable operation."
    else:
        verdict = "INSUFFICIENT — The current policy does not meet minimum performance thresholds. Reward design, data pipeline, or architecture review is recommended."

    s8 = f"""\
7. OVERALL ROBUSTNESS VERDICT
{"-" * 72}
Composite robustness score: {score}/10

{verdict}
{"=" * 72}
"""

    return "\n".join([s1, s2, s3, s4, s5, s6, s7, s8])


# ============================================================================
# TTS NARRATION — pyttsx3, non-blocking, safe
# ============================================================================

def _build_spoken_summary(metrics: dict) -> str:
    """Distil metrics into a short, natural speech string."""
    wr      = metrics["win_rate"]
    avg_ret = metrics["avg_return"]
    std_ret = metrics["std_return"]
    sharpe  = metrics["sharpe"]
    worst   = metrics["worst_return"]
    trades  = metrics["avg_trades"]

    vol_word = _classify_volatility(std_ret)
    wr_word  = "strong" if wr >= 60 else ("adequate" if wr >= 50 else "below threshold")

    return (
        f"Evaluation complete. "
        f"The agent achieved an average return of {avg_ret:.1f} percent "
        f"with {vol_word} volatility. "
        f"Win rate is {wr:.0f} percent, which is {wr_word}. "
        f"The Sharpe-like ratio stands at {sharpe:.2f}. "
        f"Worst-case episode return was {worst:.1f} percent. "
        f"Average trades per episode: {trades:.0f}. "
        f"{'Risk-adjusted performance is acceptable.' if sharpe > 0.3 else 'Risk-adjusted performance requires improvement.'}"
    )


def narrate_performance(summary_text: str, enable_voice: bool = True) -> None:
    """
    Read a spoken summary aloud using pyttsx3 (offline, non-blocking).

    Args:
        summary_text: The full written commentary (used only to derive spoken text here).
        enable_voice: If False, silently skips narration.
    """
    if not enable_voice:
        return

    def _speak(text: str) -> None:
        try:
            import pyttsx3  # lazy import — keeps module loadable without pyttsx3 installed
            engine = pyttsx3.init()
            engine.setProperty("rate", 155)    # words per minute
            engine.setProperty("volume", 0.95)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except ImportError:
            print("[narrate_performance] pyttsx3 not installed. "
                  "Run `pip install pyttsx3` to enable voice narration.")
        except Exception as e:
            print(f"[narrate_performance] TTS failed silently: {e}")

    # Extract the spoken digest from the full commentary
    # (We re-parse the metrics section so spoken text stays concise)
    thread = threading.Thread(target=_speak, args=(summary_text,), daemon=True)
    thread.start()


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_performance_summary(
    eval_df: pd.DataFrame,
    output_dir: str,
    enable_voice: bool = False,
) -> str:
    """
    Compute metrics, generate quant-style commentary, save to disk, and
    optionally narrate.

    Args:
        eval_df:     DataFrame with evaluation metrics (from final_test_evaluation.csv
                     or evaluation_history.csv). Must contain at minimum:
                         avg_return, std_return, win_rate, best_return,
                         worst_return, avg_trades, num_episodes
        output_dir:  Directory to save performance_summary.txt
        enable_voice: Whether to trigger pyttsx3 narration (default False;
                      set True in train_chronological.py integration)

    Returns:
        The full commentary string.
    """
    if eval_df is None or len(eval_df) == 0:
        return ""

    # Use last row if multiple checkpoints passed in
    row = eval_df.iloc[-1]

    avg_ret   = float(row["avg_return"])
    std_ret   = float(row.get("std_return", 0.0))
    win_rate  = float(row["win_rate"])
    best_ret  = float(row["best_return"])
    worst_ret = float(row["worst_return"])
    avg_trades = float(row["avg_trades"])
    n_eps     = int(row.get("num_episodes", len(eval_df)))

    metrics = {
        "avg_return":    avg_ret,
        "std_return":    std_ret,
        "win_rate":      win_rate,
        "best_return":   best_ret,
        "worst_return":  worst_ret,
        "avg_trades":    avg_trades,
        "num_episodes":  n_eps,
        "sharpe":        _compute_sharpe(avg_ret, std_ret),
        "profit_factor": _compute_profit_factor(avg_ret, worst_ret),
    }

    commentary = _build_commentary(metrics, eval_df)

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "performance_summary.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(commentary)

    print(f"Performance summary saved to: {save_path}")

    # Voice narration — build a short spoken digest
    if enable_voice:
        spoken = _build_spoken_summary(metrics)
        narrate_performance(spoken, enable_voice=True)

    return commentary