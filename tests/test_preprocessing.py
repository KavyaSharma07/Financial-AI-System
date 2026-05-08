# tests/test_preprocessing.py

from src.preprocessing import get_processed_data, get_feature_columns
from src.config import TICKERS

print("Testing preprocessing for all tickers...\n")

features = get_feature_columns()
print(f"Feature columns: {features}\n")

for ticker in TICKERS:
    df = get_processed_data(ticker)

    if df.empty:
        print(f"{ticker}: FAILED — no data")
        continue

    print(f"{ticker}:")
    print(f"  Rows after processing : {len(df)}")
    print(f"  Date range            : {df['date'].min().date()} "
          f"to {df['date'].max().date()}")
    print(f"  Latest close          : {df['close'].iloc[-1]:.2f}")
    print(f"  Latest MA20           : {df['MA20'].iloc[-1]:.2f}")
    print(f"  Latest MA50           : {df['MA50'].iloc[-1]:.2f}")
    print(f"  Latest RSI            : {df['RSI'].iloc[-1]:.2f}")
    print(f"  Latest volatility     : {df['volatility'].iloc[-1]:.4f}")
    print(f"  MA cross signal       : "
          f"{'Bullish' if df['MA_cross'].iloc[-1] == 1 else 'Bearish'}")
    print(f"  Target distribution   : "
          f"{(df['target']==1).sum()} UP days, "
          f"{(df['target']==0).sum()} DOWN days")
    print()

print("Preprocessing test complete.")