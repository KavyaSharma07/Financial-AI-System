# src/sentiment.py

import yfinance as yf
import pandas as pd
from datetime import date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.config import TICKERS
from src.database import save_sentiment, load_sentiment

# Initialize VADER once — reusing one instance is more efficient
analyzer = SentimentIntensityAnalyzer()


def score_headline(text: str) -> dict:
    """
    Scores a single headline using VADER.

    Returns compound score and a human-readable label.

    compound > 0.05  → Positive (Bullish)
    compound < -0.05 → Negative (Bearish)
    between          → Neutral

    Example:
    score_headline("Reliance posts record profit")
    → {'score': 0.7906, 'label': 'Positive'}
    """
    scores   = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'

    return {
        'score': round(compound, 4),
        'label': label
    }


def fetch_and_score_news(ticker: str) -> pd.DataFrame:
    """
    Fetches recent news headlines from Yahoo Finance for one ticker,
    scores each headline with VADER, stores results in PostgreSQL,
    and returns the scored DataFrame.
    """
    print(f"Fetching news for {ticker}...")

    try:
        stock = yf.Ticker(ticker)
        news  = stock.news
    except Exception as e:
        print(f"  ERROR fetching news for {ticker}: {e}")
        return pd.DataFrame()

    if not news:
        print(f"  No news found for {ticker}")
        return pd.DataFrame()

    rows = []
    for article in news:
        # Extract headline — try different key names
        # Yahoo Finance sometimes uses 'title', sometimes 'content'
        headline = article.get('title', '')
        if not headline:
            content = article.get('content', {})
            if isinstance(content, dict):
                headline = content.get('title', '')

        if not headline:
            continue

        result = score_headline(headline)

        rows.append({
            'ticker':   ticker,
            'date':     date.today().isoformat(),
            'headline': headline,
            'score':    result['score'],
            'label':    result['label']
        })

    if not rows:
        print(f"  No scoreable headlines found for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    save_sentiment(df)

    pos = (df['label'] == 'Positive').sum()
    neg = (df['label'] == 'Negative').sum()
    neu = (df['label'] == 'Neutral').sum()
    avg = df['score'].mean()

    print(f"  {len(df)} headlines scored | "
          f"Pos: {pos} | Neg: {neg} | Neu: {neu} | "
          f"Avg score: {avg:.4f}")

    return df


def get_sentiment_summary(ticker: str) -> dict:
    """
    Loads all stored sentiment scores for a ticker from PostgreSQL
    and returns a summary dictionary for the dashboard to display.
    """
    df = load_sentiment(ticker)

    if df.empty:
        return {
            'avg_score':    0,
            'label':        'Neutral',
            'positive_pct': 0,
            'negative_pct': 0,
            'neutral_pct':  0,
            'total':        0,
            'recent':       []
        }

    total = len(df)
    avg   = df['score'].mean()

    # Overall sentiment label based on average compound score
    if avg >= 0.05:
        overall = 'Bullish'
    elif avg <= -0.05:
        overall = 'Bearish'
    else:
        overall = 'Neutral'

    pos = round((df['label'] == 'Positive').sum() / total * 100, 1)
    neg = round((df['label'] == 'Negative').sum() / total * 100, 1)
    neu = round((df['label'] == 'Neutral').sum()  / total * 100, 1)

    # Get 5 most recent headlines for dashboard display
    recent = df.sort_values('date', ascending=False).head(5)
    recent = recent[['date', 'headline', 'score', 'label']].to_dict('records')

    return {
        'avg_score':    round(float(avg), 4),
        'label':        overall,
        'positive_pct': pos,
        'negative_pct': neg,
        'neutral_pct':  neu,
        'total':        total,
        'recent':       recent
    }


def fetch_all_tickers():
    """Fetches and scores news for all tickers."""
    print("=" * 50)
    print("Fetching sentiment for all tickers")
    print("=" * 50)
    for ticker in TICKERS:
        fetch_and_score_news(ticker)
    print("\nSentiment collection complete.")


if __name__ == "__main__":
    # Fetch news and score sentiment for all tickers
    fetch_all_tickers()

    # Print summary for each ticker
    print("\n" + "=" * 50)
    print("SENTIMENT SUMMARY")
    print("=" * 50)
    for ticker in TICKERS:
        summary = get_sentiment_summary(ticker)
        print(f"\n{ticker}:")
        print(f"  Overall  : {summary['label']} "
              f"(score: {summary['avg_score']})")
        print(f"  Positive : {summary['positive_pct']}%")
        print(f"  Negative : {summary['negative_pct']}%")
        print(f"  Neutral  : {summary['neutral_pct']}%")
        print(f"  Articles : {summary['total']}")
        if summary['recent']:
            print(f"  Latest headlines:")
            for item in summary['recent'][:3]:
                print(f"    [{item['label']:8s} {item['score']:+.3f}] "
                      f"{item['headline'][:60]}...")
    print("=" * 50)