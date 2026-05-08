# dashboard/pages/3_Risk_Analysis.py
# Page 4: Risk Analysis

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.config import TICKERS
from src.preprocessing import get_processed_data
from src.risk import compute_risk_metrics, compare_all_tickers

st.set_page_config(
    page_title="Risk Analysis",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ Risk Analysis")
st.caption("Quantitative risk metrics — Volatility, VaR, Sharpe Ratio")
st.divider()

# ── Stock selector ────────────────────────────────────────────
ticker  = st.selectbox("Select a stock", TICKERS)
metrics = compute_risk_metrics(ticker)

# ── Risk level badge ──────────────────────────────────────────
risk_color = {
    'Low':    '🟢', 'Medium': '🟡', 'High': '🔴'
}
st.markdown(
    f"### Risk Level: "
    f"{risk_color.get(metrics['risk_level'], '⚪')} "
    f"**{metrics['risk_level']}**"
)

# ── Metric cards ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Annualized Volatility", f"{metrics['volatility']:.2f}%")
c2.metric("VaR (95%)",             f"{metrics['var_95']:.2f}%",
          delta="Max daily loss",   delta_color="inverse")
c3.metric("Sharpe Ratio",          f"{metrics['sharpe_ratio']:.3f}")
c4.metric("Max Drawdown",          f"{metrics['max_drawdown']:.2f}%",
          delta_color="inverse")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Total Return",  f"{metrics['total_return']:.2f}%")
c6.metric("Best Day",      f"+{metrics['best_day']:.2f}%")
c7.metric("Worst Day",     f"{metrics['worst_day']:.2f}%",
          delta_color="inverse")
c8.metric("Avg Daily Return", f"{metrics['avg_return']:.4f}%")

st.divider()

# ── Charts ────────────────────────────────────────────────────
df = get_processed_data(ticker)

col_left, col_right = st.columns(2)

# Return distribution histogram
with col_left:
    st.subheader("Return Distribution")
    fig_hist = px.histogram(
        df, x='daily_return',
        nbins=80,
        title=f"{ticker} — Daily Return Distribution",
        template="plotly_white",
        color_discrete_sequence=['#534AB7']
    )
    fig_hist.add_vline(
        x=metrics['var_95'] / 100,
        line_dash="dash", line_color="#E24B4A",
        annotation_text=f"VaR 95%: {metrics['var_95']:.2f}%"
    )
    fig_hist.update_layout(height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

# Rolling volatility over time
with col_right:
    st.subheader("Rolling Volatility Over Time")
    df['rolling_vol'] = (
        df['daily_return'].rolling(20).std() * (252 ** 0.5) * 100
    )
    fig_vol = px.line(
        df, x='date', y='rolling_vol',
        title=f"{ticker} — 20-Day Rolling Volatility (%)",
        template="plotly_white",
        color_discrete_sequence=['#F5A623']
    )
    fig_vol.add_hline(
        y=35, line_dash="dash", line_color="#E24B4A",
        annotation_text="High risk threshold"
    )
    fig_vol.update_layout(height=380)
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# ── Drawdown chart ────────────────────────────────────────────
st.subheader("Drawdown Over Time — How far did it fall from peak?")

returns     = df['daily_return'].dropna()
cumulative  = (1 + returns).cumprod()
rolling_max = cumulative.cummax()
drawdown    = ((cumulative - rolling_max) / rolling_max * 100)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=df['date'].iloc[-len(drawdown):],
    y=drawdown.values,
    fill='tozeroy',
    fillcolor='rgba(226, 75, 74, 0.2)',
    line=dict(color='#E24B4A', width=1),
    name='Drawdown %'
))
fig_dd.update_layout(
    title=f"{ticker} — Historical Drawdown",
    xaxis_title="Date",
    yaxis_title="Drawdown %",
    height=300,
    template="plotly_white"
)
st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# ── All stocks comparison ─────────────────────────────────────
st.subheader("All Stocks — Risk Comparison")
risk_df = compare_all_tickers(TICKERS)

display_cols = [
    'risk_level', 'volatility', 'var_95',
    'sharpe_ratio', 'max_drawdown', 'total_return'
]
risk_display = risk_df[display_cols].copy()
risk_display.columns = [
    'Risk Level', 'Volatility %', 'VaR 95%',
    'Sharpe Ratio', 'Max Drawdown %', 'Total Return %'
]
st.dataframe(
    risk_display.style.format({
        'Volatility %':   '{:.2f}%',
        'VaR 95%':        '{:.2f}%',
        'Sharpe Ratio':   '{:.3f}',
        'Max Drawdown %': '{:.2f}%',
        'Total Return %': '{:.2f}%'
    }),
    use_container_width=True
)