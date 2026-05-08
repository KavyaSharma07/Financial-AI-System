# dashboard/pages/1_Stock_Prediction.py
# Page 2: Stock Prediction

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.config import TICKERS
from src.preprocessing import get_processed_data, get_feature_columns
from src.models.random_forest import predict_direction, load_model
from src.models.lstm import predict_next_price

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Stock Prediction")
st.caption("LSTM price forecasting + Random Forest direction prediction")
st.divider()

# ── Stock selector ────────────────────────────────────────────
ticker = st.selectbox("Select a stock", TICKERS)

# ── Load predictions ──────────────────────────────────────────
with st.spinner("Running models..."):
    lstm_result = predict_next_price(ticker)
    rf_result   = predict_direction(ticker)

# ── Metric cards ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current Price",
    f"₹{lstm_result['current_price']:,.2f}"
    if ticker.endswith('.NS')
    else f"{lstm_result['current_price']:,.2f}"
)
c2.metric(
    "LSTM Predicted Price",
    f"{lstm_result['predicted_price']:,.2f}",
    delta=f"{lstm_result['change_pct']:+.2f}%"
)
c3.metric(
    "RF Direction",
    f"{'⬆️ UP' if rf_result['direction'] == 'UP' else '⬇️ DOWN'}",
    delta=f"Confidence: {rf_result['confidence']}%"
)
c4.metric(
    "Up Probability",
    f"{rf_result['up_prob']}%"
)

st.divider()

# ── Price history chart ───────────────────────────────────────
df = get_processed_data(ticker)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Price History — Last 120 Days")
    df_plot = df.tail(120)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot['close'],
        name='Actual Price',
        line=dict(color='#534AB7', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot['MA20'],
        name='MA20',
        line=dict(color='#F5A623', width=1.5, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot['BB_upper'],
        name='BB Upper',
        line=dict(color='#E24B4A', width=1, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot['BB_lower'],
        name='BB Lower',
        line=dict(color='#1D9E75', width=1, dash='dot')
    ))
    fig.update_layout(
        height=400, template="plotly_white",
        xaxis_title="Date", yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Latest Indicators")
    latest = df.iloc[-1]
    st.metric("RSI", f"{latest['RSI']:.1f}",
              delta="Overbought" if latest['RSI'] > 70
              else "Oversold" if latest['RSI'] < 30
              else "Normal")
    st.metric("Volatility (20d)", f"{latest['volatility']*100:.2f}%")
    st.metric("Volume Ratio",     f"{latest['volume_ratio']:.2f}x")
    st.metric("MA Cross Signal",
              "Bullish 📈" if latest['MA_cross'] == 1 else "Bearish 📉")

st.divider()

# ── Feature importance ────────────────────────────────────────
st.subheader("Feature Importance — What drives this prediction?")

try:
    model    = load_model(ticker)
    features = get_feature_columns()
    imp_df   = pd.DataFrame({
        'Feature':    features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig_imp = px.bar(
        imp_df, x='Importance', y='Feature',
        orientation='h',
        title=f"Feature Importance for {ticker}",
        color='Importance',
        color_continuous_scale='Purples',
        template="plotly_white"
    )
    fig_imp.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load feature importance: {e}")