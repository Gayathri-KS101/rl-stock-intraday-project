"""
Live Training Dashboard
=======================
Reads training artifacts from disk and renders a live view.
Runs completely independently of the training process.

Usage:
    streamlit run dashboard.py

Auto-detects the latest run under output_data/ and refreshes every 2 seconds.
Safe against partially-written CSV files at all times.
"""

import os
import glob
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================================
st.set_page_config(
    page_title="DQN Trading — Live Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CONSTANTS
# ============================================================================
OUTPUT_ROOT = "output_data"
REFRESH_INTERVAL_SEC = 2

COLOUR_INVEST  = "#00c853"
COLOUR_CASH    = "#ff1744"
COLOUR_NET_WORTH = "#1565c0"
COLOUR_PRICE   = "#424242"
COLOUR_DRAW    = "rgba(255,23,68,0.15)"

# ============================================================================
# DISK HELPERS — all I/O is safe against partial writes
# ============================================================================

def _safe_read_csv(path: str) -> pd.DataFrame | None:
    """Read CSV; return None on any I/O or parse error."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def detect_latest_run(root: str = OUTPUT_ROOT) -> str | None:
    """Return the path of the most-recently-modified run folder."""
    pattern = os.path.join(root, "*run_*")
    runs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return runs[0] if runs else None


def load_episode_metrics(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "episode_metrics.csv")
    return _safe_read_csv(path)


def load_latest_episode_log(run_dir: str) -> pd.DataFrame | None:
    """Find and load the highest-numbered episode debug log."""
    pattern = os.path.join(run_dir, "episode_*_debug_log.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return _safe_read_csv(files[-1])


def load_evaluation_history(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "evaluation_results", "evaluation_history.csv")
    return _safe_read_csv(path)


def load_final_eval(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "evaluation_results", "final_test_evaluation.csv")
    return _safe_read_csv(path)


# ============================================================================
# METRIC COMPUTATIONS — stateless, operate on DataFrames
# ============================================================================

def compute_max_drawdown(net_worth: np.ndarray) -> float:
    if len(net_worth) == 0:
        return 0.0
    peak = np.maximum.accumulate(net_worth)
    dd = (net_worth - peak) / np.where(peak == 0, 1, peak)
    return float(dd.min()) * 100  # as %


def compute_rolling_volatility(rewards: pd.Series, window: int = 20) -> float:
    if len(rewards) < 2:
        return 0.0
    return float(rewards.rolling(window, min_periods=2).std().iloc[-1])


def compute_sharpe(rewards: pd.Series) -> float:
    if len(rewards) < 2:
        return 0.0
    std = rewards.std()
    return float(rewards.mean() / std) if std > 1e-10 else 0.0


# ============================================================================
# PLOT BUILDERS — all return Plotly figures
# ============================================================================

def build_equity_curve(ep_log: pd.DataFrame) -> go.Figure:
    steps     = ep_log["Step"].values
    net_worth = ep_log["Net_Worth"].values
    price     = ep_log["Current_Price"].values

    peak = np.maximum.accumulate(net_worth)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Net Worth + Drawdown", "Price Action"),
    )

    # Drawdown fill
    fig.add_trace(go.Scatter(
        x=np.concatenate([steps, steps[::-1]]),
        y=np.concatenate([peak, net_worth[::-1]]),
        fill="toself",
        fillcolor=COLOUR_DRAW,
        line=dict(width=0),
        name="Drawdown",
        showlegend=True,
    ), row=1, col=1)

    # Net worth line
    fig.add_trace(go.Scatter(
        x=steps, y=net_worth,
        line=dict(color=COLOUR_NET_WORTH, width=2),
        name="Net Worth",
    ), row=1, col=1)

    # Invest markers
    invest_mask = ep_log["Action"] == 1
    if invest_mask.any():
        fig.add_trace(go.Scatter(
            x=ep_log.loc[invest_mask, "Step"],
            y=ep_log.loc[invest_mask, "Net_Worth"],
            mode="markers",
            marker=dict(symbol="triangle-up", color=COLOUR_INVEST, size=7, line=dict(width=0.5, color="black")),
            name="Invest",
        ), row=1, col=1)

    # Cash markers
    cash_mask = ep_log["Action"] == 0
    if cash_mask.any():
        fig.add_trace(go.Scatter(
            x=ep_log.loc[cash_mask, "Step"],
            y=ep_log.loc[cash_mask, "Net_Worth"],
            mode="markers",
            marker=dict(symbol="triangle-down", color=COLOUR_CASH, size=7, line=dict(width=0.5, color="black")),
            name="Cash",
        ), row=1, col=1)

    # Initial balance line
    fig.add_hline(y=float(net_worth[0]), line_dash="dot", line_color="gray",
                  annotation_text="Initial", row=1, col=1)

    # Price
    fig.add_trace(go.Scatter(
        x=steps, y=price,
        line=dict(color=COLOUR_PRICE, width=1.5),
        name="Price",
    ), row=2, col=1)

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    fig.update_xaxes(gridcolor="#1e2130", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2130", zeroline=False)
    return fig


def build_reward_curve(metrics_df: pd.DataFrame) -> go.Figure:
    episodes = metrics_df["Episode"].values
    rewards  = metrics_df["Total_Reward"].values
    rolling  = pd.Series(rewards).rolling(20, min_periods=1).mean().values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards,
        line=dict(color="#546e7a", width=1),
        opacity=0.5,
        name="Raw Reward",
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=rolling,
        line=dict(color="#ffa726", width=2.5),
        name="20-ep MA",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Training Reward Curve",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    fig.update_xaxes(title="Episode", gridcolor="#1e2130")
    fig.update_yaxes(title="Reward", gridcolor="#1e2130")
    return fig


def build_return_histogram(metrics_df: pd.DataFrame) -> go.Figure:
    returns = metrics_df["Percent_Return"].values
    colours = [COLOUR_INVEST if r >= 0 else COLOUR_CASH for r in returns]

    fig = go.Figure(go.Histogram(
        x=returns,
        nbinsx=40,
        marker_color="#42a5f5",
        marker_line=dict(color="#1565c0", width=0.5),
        opacity=0.85,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=float(np.mean(returns)), line_dash="dot", line_color="#ffa726",
                  annotation_text=f"μ={np.mean(returns):.2f}%",
                  annotation_position="top right")
    fig.update_layout(
        title="Return Distribution",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    fig.update_xaxes(title="Return (%)", gridcolor="#1e2130")
    fig.update_yaxes(title="Count", gridcolor="#1e2130")
    return fig


def build_trade_histogram(metrics_df: pd.DataFrame) -> go.Figure:
    trades = metrics_df["Number_of_Trades"].values
    fig = go.Figure(go.Histogram(
        x=trades,
        nbinsx=30,
        marker_color="#ab47bc",
        marker_line=dict(color="#7b1fa2", width=0.5),
        opacity=0.85,
    ))
    fig.add_vline(x=float(np.mean(trades)), line_dash="dot", line_color="#ffa726",
                  annotation_text=f"μ={np.mean(trades):.1f}",
                  annotation_position="top right")
    fig.update_layout(
        title="Trade Count Distribution",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    fig.update_xaxes(title="Trades / Episode", gridcolor="#1e2130")
    fig.update_yaxes(title="Count", gridcolor="#1e2130")
    return fig


def build_eval_timeline(eval_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Avg Return per Checkpoint", "Win Rate per Checkpoint"),
    )
    episodes = eval_df["episode"].values if "episode" in eval_df.columns else np.arange(len(eval_df))

    colours = [COLOUR_INVEST if v >= 0 else COLOUR_CASH for v in eval_df["avg_return"].values]
    fig.add_trace(go.Bar(x=episodes, y=eval_df["avg_return"].values,
                         marker_color=colours, name="Avg Return"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=episodes, y=eval_df["win_rate"].values,
                             mode="lines+markers",
                             line=dict(color="#42a5f5", width=2),
                             name="Win Rate"), row=1, col=2)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=2)

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    fig.update_xaxes(gridcolor="#1e2130", title="Episode")
    fig.update_yaxes(gridcolor="#1e2130")
    return fig


def build_q_value_plot(ep_log: pd.DataFrame) -> go.Figure:
    steps = ep_log["Step"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=ep_log["Q_Cash"].values,
                             line=dict(color=COLOUR_CASH, width=1.5), name="Q_Cash"))
    fig.add_trace(go.Scatter(x=steps, y=ep_log["Q_Invest"].values,
                             line=dict(color=COLOUR_INVEST, width=1.5), name="Q_Invest"))
    fig.update_layout(
        title="Q-Values (Latest Episode)",
        height=240,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    fig.update_xaxes(title="Step", gridcolor="#1e2130")
    fig.update_yaxes(title="Q-Value", gridcolor="#1e2130")
    return fig


# ============================================================================
# STYLED METRIC CARD
# ============================================================================

def metric_card(label: str, value: str, delta: str = "", colour: str = "#fafafa") -> None:
    delta_html = f"<p style='font-size:0.78rem;color:#9e9e9e;margin:0'>{delta}</p>" if delta else ""
    st.markdown(f"""
    <div style='background:#1e2130;border-radius:8px;padding:14px 18px;margin-bottom:6px'>
        <p style='font-size:0.75rem;color:#9e9e9e;margin:0;text-transform:uppercase;letter-spacing:0.08em'>{label}</p>
        <p style='font-size:1.55rem;font-weight:700;color:{colour};margin:2px 0'>{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN DASHBOARD RENDER
# ============================================================================

def render_dashboard() -> None:
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Dashboard Controls")
        run_dir = detect_latest_run()

        if run_dir:
            run_name = os.path.basename(run_dir)
            st.success(f"**Active run:**\n`{run_name}`")
        else:
            st.warning("No runs found under `output_data/`")
            st.stop()

        auto_refresh = st.toggle("Auto-refresh (2 s)", value=True)
        st.caption(f"Root: `{os.path.abspath(OUTPUT_ROOT)}`")
        st.divider()
        st.markdown("**Run:**")
        all_runs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "*run_*")),
                          key=os.path.getmtime, reverse=True)
        selected_run = st.selectbox(
            "Select run", options=[os.path.basename(r) for r in all_runs],
            index=0,
        )
        run_dir = os.path.join(OUTPUT_ROOT, selected_run)

    # ── Load data ─────────────────────────────────────────────────────────────
    metrics_df = load_episode_metrics(run_dir)
    ep_log     = load_latest_episode_log(run_dir)
    eval_df    = load_evaluation_history(run_dir)
    final_eval = load_final_eval(run_dir)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0'>📈 DQN Trading — Live Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Run: `{os.path.basename(run_dir)}` · Last refresh: `{time.strftime('%H:%M:%S')}`")
    st.divider()

    # ── SECTION A: Live Episode Monitor ───────────────────────────────────────
    st.subheader("A — Live Episode Monitor")

    if metrics_df is not None and len(metrics_df) > 0:
        last = metrics_df.iloc[-1]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            metric_card("Episode", str(int(last["Episode"])))
        with c2:
            nw = last["Final_Net_Worth"]
            metric_card("Net Worth", f"${nw:,.2f}",
                        colour=COLOUR_INVEST if nw >= 10000 else COLOUR_CASH)
        with c3:
            ret = last["Percent_Return"]
            metric_card("Return", f"{ret:+.2f}%",
                        colour=COLOUR_INVEST if ret >= 0 else COLOUR_CASH)
        with c4:
            metric_card("Trades", str(int(last["Number_of_Trades"])))
        with c5:
            metric_card("Reward", f"{last['Total_Reward']:.4f}")
        with c6:
            total_eps = len(metrics_df)
            win_rate  = (metrics_df["Percent_Return"] > 0).mean() * 100
            metric_card("Win Rate", f"{win_rate:.1f}%",
                        colour=COLOUR_INVEST if win_rate >= 50 else COLOUR_CASH)
    else:
        st.info("Waiting for first episode to complete…")

    st.divider()

    # ── SECTION B: Equity Curve ────────────────────────────────────────────────
    st.subheader("B — Live Equity Curve (Latest Episode)")

    if ep_log is not None:
        st.plotly_chart(build_equity_curve(ep_log), width="stretch")
    else:
        st.info("No episode logs found yet. Waiting for first logged episode…")

    # ── SECTION C: Risk Metrics ────────────────────────────────────────────────
    st.subheader("C — Risk Metrics Panel")

    if ep_log is not None and metrics_df is not None:
        nw_arr   = ep_log["Net_Worth"].values
        rewards  = metrics_df["Total_Reward"]
        max_dd   = compute_max_drawdown(nw_arr)
        roll_vol = compute_rolling_volatility(rewards)
        sharpe   = compute_sharpe(rewards)
        n_total  = len(ep_log)
        if "Decision_Reason" in ep_log.columns:
            expl_pct = ep_log["Decision_Reason"].str.contains("Exploration").mean() * 100
        else:
            expl_pct = 0.0
        avg_qc = ep_log["Q_Cash"].mean() if "Q_Cash" in ep_log.columns else 0.0
        avg_qi = ep_log["Q_Invest"].mean() if "Q_Invest" in ep_log.columns else 0.0

        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
        with rc1:
            metric_card("Max Drawdown", f"{max_dd:.2f}%",
                        colour=COLOUR_CASH if max_dd < -2 else "#ffa726")
        with rc2:
            metric_card("Rolling Vol (20)", f"{roll_vol:.4f}")
        with rc3:
            metric_card("Sharpe-Like", f"{sharpe:.3f}",
                        colour=COLOUR_INVEST if sharpe > 0 else COLOUR_CASH)
        with rc4:
            metric_card("Exploration", f"{expl_pct:.1f}%")
        with rc5:
            metric_card("Avg Q_Cash", f"{avg_qc:.4f}")
        with rc6:
            metric_card("Avg Q_Invest", f"{avg_qi:.4f}")

        st.plotly_chart(build_q_value_plot(ep_log), width="stretch")
    else:
        st.info("Risk metrics available after first logged episode.")

    st.divider()

    # ── SECTION D: Distributions ───────────────────────────────────────────────
    st.subheader("D — Distributions")

    if metrics_df is not None and len(metrics_df) >= 2:
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(build_return_histogram(metrics_df), width="stretch")
        with col_right:
            st.plotly_chart(build_trade_histogram(metrics_df), width="stretch")

        st.plotly_chart(build_reward_curve(metrics_df), width="stretch")
    else:
        st.info("Distributions available after at least 2 episodes complete.")

    st.divider()

    # ── SECTION E: Evaluation Summary ─────────────────────────────────────────
    st.subheader("E — Evaluation Summary")

    eval_source = final_eval if final_eval is not None else eval_df

    if eval_source is not None and len(eval_source) > 0:
        row = eval_source.iloc[-1]

        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
        with ec1:
            wr = row["win_rate"]
            metric_card("Win Rate", f"{wr:.1f}%",
                        colour=COLOUR_INVEST if wr >= 50 else COLOUR_CASH)
        with ec2:
            metric_card("Avg Return", f"{row['avg_return']:.2f}%",
                        colour=COLOUR_INVEST if row["avg_return"] >= 0 else COLOUR_CASH)
        with ec3:
            metric_card("Std Return", f"{row['std_return']:.2f}%")
        with ec4:
            metric_card("Best Return", f"{row['best_return']:.2f}%", colour=COLOUR_INVEST)
        with ec5:
            metric_card("Worst Return", f"{row['worst_return']:.2f}%", colour=COLOUR_CASH)
        with ec6:
            metric_card("Avg Trades", f"{row['avg_trades']:.1f}")

        if eval_df is not None and len(eval_df) >= 2:
            st.plotly_chart(build_eval_timeline(eval_df), width="stretch")
    else:
        st.info("Evaluation data available after first evaluation checkpoint.")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL_SEC)
        st.rerun()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    render_dashboard()
else:
    # Called by `streamlit run dashboard.py`
    render_dashboard()