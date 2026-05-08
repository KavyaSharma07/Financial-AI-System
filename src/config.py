from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

TICKERS = [
    "RELIANCE.NS",   # India — Energy
    "TCS.NS",        # India — IT
    "INFY.NS",       # India — IT
    "HDFCBANK.NS",   # India — Banking
    "AAPL",          # USA — Technology (Apple)
    "MSFT",          # USA — Technology (Microsoft)
    "HSBA.L",        # UK — Banking (HSBC)
    "7203.T",        # Japan — Automotive (Toyota)
]

START_DATE     = "2020-01-01"
END_DATE       = "2024-12-31"
SEQ_LEN        = 60
RISK_FREE_RATE = 0.06