from sqlalchemy import create_engine, text
import pandas as pd
from src.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def create_tables():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id        SERIAL PRIMARY KEY,
                ticker    VARCHAR(20)  NOT NULL,
                date      DATE         NOT NULL,
                open      FLOAT,
                high      FLOAT,
                low       FLOAT,
                close     FLOAT,
                volume    BIGINT,
                UNIQUE(ticker, date)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                id        SERIAL PRIMARY KEY,
                ticker    VARCHAR(20)  NOT NULL,
                date      DATE         NOT NULL,
                headline  TEXT,
                score     FLOAT,
                label     VARCHAR(10),
                UNIQUE(ticker, date, headline)
            );
        """))
        conn.commit()
    print("Tables created successfully.")


def save_stock_data(df: pd.DataFrame, ticker: str):
    df = df.copy()
    df['ticker'] = ticker
    df.columns = [c.lower() for c in df.columns]
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO stock_prices
                    (ticker, date, open, high, low, close, volume)
                VALUES
                    (:ticker, :date, :open, :high, :low, :close, :volume)
                ON CONFLICT (ticker, date) DO NOTHING
            """), row.to_dict())
        conn.commit()
    print(f"Saved {len(df)} rows for {ticker}.")


def load_stock_data(ticker: str) -> pd.DataFrame:
    query = text(
        "SELECT * FROM stock_prices WHERE ticker = :ticker ORDER BY date ASC"
    )
    df = pd.read_sql(query, engine, params={"ticker": ticker})
    df['date'] = pd.to_datetime(df['date'])
    return df


def save_sentiment(df: pd.DataFrame):
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO sentiment_scores
                    (ticker, date, headline, score, label)
                VALUES
                    (:ticker, :date, :headline, :score, :label)
                ON CONFLICT (ticker, date, headline) DO NOTHING
            """), row.to_dict())
        conn.commit()
    print(f"Saved {len(df)} sentiment rows.")


def load_sentiment(ticker: str) -> pd.DataFrame:
    query = text(
        "SELECT * FROM sentiment_scores WHERE ticker = :ticker ORDER BY date ASC"
    )
    return pd.read_sql(query, engine, params={"ticker": ticker})