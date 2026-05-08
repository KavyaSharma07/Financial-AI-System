# tests/test_risk.py
from src.risk import compute_risk_metrics, compare_all_tickers
from src.config import TICKERS

print("Testing risk analysis...\n")

for ticker in TICKERS:
    m = compute_risk_metrics(ticker)
    print(f"{ticker:15s} | {m['risk_level']:6s} | "
          f"Vol: {m['volatility']:5.1f}% | "
          f"VaR: {m['var_95']:6.2f}% | "
          f"Sharpe: {m['sharpe_ratio']:5.3f} | "
          f"Return: {m['total_return']:6.1f}%")