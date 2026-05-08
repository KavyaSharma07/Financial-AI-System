import yfinance as yf
import pandas as pd
from src.config import TICKERS, START_DATE, END_DATE
from src.database import save_stock_data


def fetch_single_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Fetching {ticker}...")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        print(f"WARNING: No data returned for {ticker}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)
    df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    print(f"  Got {len(df)} rows for {ticker}")
    return df


def fetch_and_store_all():
    print("=" * 50)
    print("Starting data collection for all tickers")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("=" * 50)

    success = []
    failed  = []

    for ticker in TICKERS:
        try:
            df = fetch_single_ticker(ticker, START_DATE, END_DATE)
            if not df.empty:
                save_stock_data(df, ticker)
                success.append(ticker)
            else:
                failed.append(ticker)
        except Exception as e:
            print(f"ERROR fetching {ticker}: {e}")
            failed.append(ticker)

    print("\n" + "=" * 50)
    print(f"Done. Successful: {success}")
    if failed:
        print(f"Failed: {failed}")
    print("=" * 50)


def verify_data():
    from src.database import load_stock_data
    print("\nVerifying data in database:")
    print("-" * 40)
    for ticker in TICKERS:
        try:
            df = load_stock_data(ticker)
            if df.empty:
                print(f"  {ticker}: NO DATA FOUND")
            else:
                print(f"  {ticker}: {len(df)} rows | "
                      f"{df['date'].min().date()} to "
                      f"{df['date'].max().date()} | "
                      f"Latest close: {df['close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"  {ticker}: ERROR — {e}")
    print("-" * 40)


if __name__ == "__main__":
    fetch_and_store_all()
    verify_data()