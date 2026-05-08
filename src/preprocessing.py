import pandas as pd
import numpy as np 
from src.database import load_stock_data

def compute_features (df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw OHLCV data and adds all technical indicators.
    Returns a clean DataFrame ready for model training.

    Input:  raw price DataFrame from database
    Output: same DataFrame with 8 new feature columns added
    """ 
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    #trend indicatores ----
    # Moving average 20: Average of last 20 closing prices
    # Smooths out daily noise to reveal the underlying trend
    df['MA20'] = df['close'].rolling(window=20).mean()

    #Moving average 50: slower average - shows longer term trend 
    df['MA50'] =df['close'].rolling(window= 50).mean()

    #MA crossover signal 
    # When MA20 crosses ABOVE MA50 → bullish signal (1)
    # When MA20 is BELOW MA50 → bearish signal (0)
    # This is called the "Golden Cross" / "Death Cross" in finance
    df['MA_cross'] = (df['MA20']> df['MA50']).astype(int)
    
    #Momentum Indicators 
    #Daily return: percentage change from previous day's close
    # Example: if yesterday was ₹100 and today is ₹103 → return is 3%
    # We use % returns instead of raw prices because:
    # A ₹10 move means very different things for ₹100 vs ₹10,000 stocks
    df['daily_return'] = df['close'].pct_change()

    # RSI — Relative Strength Index (14-day)
    # Measures speed and magnitude of price changes
    # RSI > 70 → overbought (price may fall soon)
    # RSI < 30 → oversold (price may rise soon)
    # RSI between 30-70 → normal range
    delta    = df['close'].diff()
    gain     = delta.clip(lower=0)           # Only positive moves
    loss     = -delta.clip(upper=0)          # Only negative moves (made positive)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs       = avg_gain / (avg_loss + 1e-9)  # 1e-9 prevents division by zero
    df['RSI'] = 100 - (100 / (1 + rs))

    # ── Volatility indicators ─────────────────────────────
    # Rolling 20-day standard deviation of daily returns
    # High volatility = risky stock, wide price swings
    # Low volatility = stable stock, predictable price moves
    df['volatility'] = df['daily_return'].rolling(window=20).std()

    # Bollinger Bands
    # Upper band = MA20 + 2 standard deviations
    # Lower band = MA20 - 2 standard deviations
    # When price touches upper band → potentially overbought
    # When price touches lower band → potentially oversold
    rolling_std      = df['close'].rolling(window=20).std()
    df['BB_upper']   = df['MA20'] + (2 * rolling_std)
    df['BB_lower']   = df['MA20'] - (2 * rolling_std)

    # BB_width: how wide the band is
    # Wider band = more volatile period
    # Narrow band = calm market (often precedes a big move)
    df['BB_width']   = df['BB_upper'] - df['BB_lower']

    # ── Volume indicator ──────────────────────────────────
    # Volume ratio: today's volume vs 20-day average volume
    # Ratio > 1.5 means unusually high trading activity
    # High volume on an up day = strong bullish signal
    # High volume on a down day = strong bearish signal
    df['volume_ratio'] = df['volume'] / (
        df['volume'].rolling(window=20).mean() + 1e-9
    )

    # ── Target variable ───────────────────────────────────
    # This is what our Random Forest model will try to predict
    # 1 = tomorrow's close is HIGHER than today's close (UP)
    # 0 = tomorrow's close is LOWER than today's close (DOWN)
    # shift(-1) looks one row ahead into the future
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop rows where rolling calculations produced NaN
    # First 50 rows will have NaN because MA50 needs 50 days of history
    # Last 1 row will have NaN in target (no "tomorrow" for the last row)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_feature_columns() -> list:
    """
    Returns the exact list of feature columns used for model training.
    Defined once here so models and dashboard always use the same features.
    """
    return [
        'MA20',
        'MA50',
        'MA_cross',
        'daily_return',
        'RSI',
        'volatility',
        'BB_width',
        'volume_ratio'
    ]


def get_processed_data(ticker: str) -> pd.DataFrame:
    """
    Single convenience function used by dashboard pages and models.
    Loads raw data from PostgreSQL and returns processed DataFrame.

    Usage in any other file:
        from src.preprocessing import get_processed_data
        df = get_processed_data('RELIANCE.NS')
    """
    df = load_stock_data(ticker)

    if df.empty:
        print(f"WARNING: No data found for {ticker}")
        return pd.DataFrame()

    return compute_features(df)