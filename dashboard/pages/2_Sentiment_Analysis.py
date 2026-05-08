# dashboard/pages/2_Sentiment_Analysis.py
# Page 3: Sentiment Analysis

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.config import TICKERS
from src.sentiment import get_sentiment_summary, fetch_and_score_news
from src.database import load_sentiment

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Sentiment Analysis")
st.caption("Financial news scored using VADER NLP")
st.divider()

# ── Stock selector + refresh ──────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.selectbox("Select a stock", TICKERS)
with col2:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh News"):
        with st.spinner("Fetching latest news..."):
            fetch_and_score_news(ticker)
        st.success("News updated!")

# ── Sentiment summary ─────────────────────────────────────────
summary = get_sentiment_summary(ticker)

c1, c2, c3, c4 = st.columns(4)

sentiment_color = (
    "normal" if summary['label'] == 'Bullish'
    else "inverse" if summary['label'] == 'Bearish'
    else "off"
)

c1.metric(
    "Overall Sentiment",
    summary['label'],
    delta=f"Score: {summary['avg_score']:+.4f}"
)
c2.metric("Positive Articles", f"{summary['positive_pct']}%")
c3.metric("Negative Articles", f"{summary['negative_pct']}%")
c4.metric("Total Articles",    summary['total'])

st.divider()

# ── Charts ────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

# Sentiment over time
df_sent = load_sentiment(ticker)

with col_left:
    st.subheader("Sentiment Score Over Time")
    if not df_sent.empty:
        df_sent['date'] = pd.to_datetime(df_sent['date'])
        daily_avg = (
            df_sent.groupby('date')['score']
            .mean()
            .reset_index()
        )
        fig_time = px.line(
            daily_avg, x='date', y='score',
            title=f"{ticker} — Daily Average Sentiment",
            template="plotly_white",
            color_discrete_sequence=['#534AB7']
        )
        fig_time.add_hline(
            y=0.05, line_dash="dash",
            line_color="#1D9E75",
            annotation_text="Bullish threshold"
        )
        fig_time.add_hline(
            y=-0.05, line_dash="dash",
            line_color="#E24B4A",
            annotation_text="Bearish threshold"
        )
        fig_time.add_hline(
            y=0, line_color="gray",
            line_width=0.5
        )
        fig_time.update_layout(height=350)
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No sentiment data yet. Click Refresh News.")

# Sentiment breakdown pie chart
with col_right:
    st.subheader("Sentiment Breakdown")
    if summary['total'] > 0:
        pie_data = pd.DataFrame({
            'Label':      ['Positive', 'Negative', 'Neutral'],
            'Percentage': [
                summary['positive_pct'],
                summary['negative_pct'],
                summary['neutral_pct']
            ]
        })
        fig_pie = px.pie(
            pie_data, values='Percentage', names='Label',
            color='Label',
            color_discrete_map={
                'Positive': '#1D9E75',
                'Negative': '#E24B4A',
                'Neutral':  '#888780'
            },
            template="plotly_white"
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── Recent headlines table ────────────────────────────────────
st.subheader("Recent Headlines with Scores")

if not df_sent.empty:
    display_df = df_sent[['date', 'headline', 'score', 'label']]\
        .sort_values('date', ascending=False)\
        .head(15)\
        .reset_index(drop=True)
    display_df.columns = ['Date', 'Headline', 'Score', 'Sentiment']

    def color_sentiment(val):
        if val == 'Positive':
            return 'color: #1D9E75; font-weight: bold'
        elif val == 'Negative':
            return 'color: #E24B4A; font-weight: bold'
        return 'color: #888780'

    st.dataframe(
        display_df.style.applymap(
            color_sentiment, subset=['Sentiment']
        ),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No headlines stored yet. Click Refresh News above.")