# dashboard/app.py
# Page 1: Market Overview

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.config import TICKERS
from src.preprocessing import get_processed_data
from src.risk import compare_all_tickers
from src.sentiment import get_sentiment_summary

st.set_page_config(
    page_title="Financial AI Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Market Overview")
st.caption("AI-Driven Financial Market Prediction and Risk Analysis System")
st.divider()

# ── Top metric cards ──────────────────────────────────────────
# One card per stock showing current price and daily change
st.subheader("Live Market Summary")

cols = st.columns(4)
indian_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

for i, ticker in enumerate(indian_tickers):
    df      = get_processed_data(ticker)
    current = df['close'].iloc[-1]
    prev    = df['close'].iloc[-2]
    change  = ((current - prev) / prev) * 100
    arrow   = "▲" if change > 0 else "▼"
    name    = ticker.replace('.NS', '')

    cols[i].metric(
        label=name,
        value=f"₹{current:,.2f}",
        delta=f"{arrow} {abs(change):.2f}%"
    )

st.divider()

# ── Price chart with moving averages ─────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    selected = st.selectbox(
        "Select stock to view",
        TICKERS,
        key="overview_ticker"
    )

with col2:
    period = st.selectbox(
        "Time period",
        ["3 months", "6 months", "1 year", "All"],
        index=2
    )

df = get_processed_data(selected)

# Filter by period
period_map = {"3 months": 63, "6 months": 126, "1 year": 252, "All": len(df)}
n_days = period_map[period]
df_plot = df.tail(n_days)

# Candlestick chart
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_plot['date'],
    open=df_plot['open'],
    high=df_plot['high'],
    low=df_plot['low'],
    close=df_plot['close'],
    name='Price',
    increasing_line_color='#1D9E75',
    decreasing_line_color='#E24B4A'
))
fig.add_trace(go.Scatter(
    x=df_plot['date'], y=df_plot['MA20'],
    name='MA20', line=dict(color='#F5A623', width=1.5)
))
fig.add_trace(go.Scatter(
    x=df_plot['date'], y=df_plot['MA50'],
    name='MA50', line=dict(color='#534AB7', width=1.5)
))
fig.update_layout(
    title=f"{selected} — Price History",
    xaxis_title="Date",
    yaxis_title="Price",
    height=450,
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Sentiment overview ────────────────────────────────────────
st.subheader("Market Sentiment Overview")
sent_cols = st.columns(4)

for i, ticker in enumerate(indian_tickers):
    summary = get_sentiment_summary(ticker)
    name    = ticker.replace('.NS', '')
    color   = ("🟢" if summary['label'] == 'Bullish'
                else "🔴" if summary['label'] == 'Bearish'
                else "🟡")
    sent_cols[i].metric(
        label=f"{name} Sentiment",
        value=f"{color} {summary['label']}",
        delta=f"Score: {summary['avg_score']:+.3f}"
    )

st.divider()

# ── Risk comparison table ─────────────────────────────────────
st.subheader("Risk Comparison — All Stocks")

risk_df = compare_all_tickers(TICKERS)
display_cols = [
    'risk_level', 'volatility', 'var_95',
    'sharpe_ratio', 'max_drawdown', 'total_return'
]
risk_df_display = risk_df[display_cols].copy()
risk_df_display.columns = [
    'Risk Level', 'Volatility %', 'VaR 95%',
    'Sharpe Ratio', 'Max Drawdown %', 'Total Return %'
]

st.dataframe(
    risk_df_display.style.format({
        'Volatility %':    '{:.2f}%',
        'VaR 95%':         '{:.2f}%',
        'Sharpe Ratio':    '{:.3f}',
        'Max Drawdown %':  '{:.2f}%',
        'Total Return %':  '{:.2f}%'
    }),
    use_container_width=True
)