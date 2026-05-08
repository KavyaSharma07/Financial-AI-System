# src/risk.py

import numpy as np
import pandas as pd
from src.preprocessing import get_processed_data
from src.config import TICKERS, RISK_FREE_RATE


def compute_risk_metrics(ticker: str) -> dict:
    """
    Computes all risk metrics for one ticker.
    Returns a dictionary of metrics ready for the dashboard.
    """
    df      = get_processed_data(ticker)
    returns = df['daily_return'].dropna()

    # ── 1. Annualized Volatility ──────────────────────────
    # Daily std × √252 converts daily risk to annual risk
    # 252 = approximate number of trading days per year
    daily_std  = returns.std()
    volatility = daily_std * np.sqrt(252)

    # ── 2. Value at Risk (95% confidence) ─────────────────
    # The 5th percentile of returns
    # On 95% of days, loss will not exceed this number
    var_95 = np.percentile(returns, 5)

    # ── 3. Sharpe Ratio ───────────────────────────────────
    # Convert annual risk-free rate to daily
    # Then compute: (avg daily return - daily rf) / daily std
    # Multiply by √252 to annualize
    daily_rf = RISK_FREE_RATE / 252
    sharpe   = (
        (returns.mean() - daily_rf) / (daily_std + 1e-9)
    ) * np.sqrt(252)

    # ── 4. Maximum Drawdown ───────────────────────────────
    # The worst peak-to-trough decline during the period
    # Example: if stock fell from ₹1000 to ₹700, drawdown = -30%
    # This is the most important metric for risk-averse investors
    cumulative  = (1 + returns).cumprod()
    # cumprod() builds the growth curve:
    # if returns are [0.01, -0.02, 0.03] then
    # cumulative = [1.01, 0.9898, 1.0195]

    rolling_max = cumulative.cummax()
    # cummax() tracks the highest point reached so far

    drawdown    = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    # min() finds the deepest drop from any peak

    # ── 5. Total Return ───────────────────────────────────
    # How much did the stock grow over the entire period?
    total_return = (
        df['close'].iloc[-1] / df['close'].iloc[0] - 1
    ) * 100

    # ── 6. Risk Classification ────────────────────────────
    # Simple rule-based system
    # Annualized volatility determines risk level
    vol_pct = volatility * 100
    if vol_pct < 20:
        risk_level = 'Low'
        risk_score = 1
    elif vol_pct < 35:
        risk_level = 'Medium'
        risk_score = 2
    else:
        risk_level = 'High'
        risk_score = 3

    return {
        'ticker':       ticker,
        'volatility':   round(vol_pct, 2),
        'var_95':       round(var_95 * 100, 2),
        'sharpe_ratio': round(float(sharpe), 3),
        'max_drawdown': round(float(max_drawdown) * 100, 2),
        'total_return': round(float(total_return), 2),
        'risk_level':   risk_level,
        'risk_score':   risk_score,
        'avg_return':   round(float(returns.mean()) * 100, 4),
        'best_day':     round(float(returns.max()) * 100, 2),
        'worst_day':    round(float(returns.min()) * 100, 2)
    }


def compare_all_tickers(tickers: list) -> pd.DataFrame:
    """
    Computes risk metrics for all tickers and returns
    a comparison DataFrame — used by the dashboard table.
    """
    rows = []
    for ticker in tickers:
        try:
            metrics = compute_risk_metrics(ticker)
            rows.append(metrics)
        except Exception as e:
            print(f"ERROR computing risk for {ticker}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index('ticker')
    return df


def get_risk_level_color(risk_level: str) -> str:
    """Returns a color string for dashboard display."""
    colors = {
        'Low':    'green',
        'Medium': 'orange',
        'High':   'red'
    }
    return colors.get(risk_level, 'gray')


if __name__ == "__main__":
    print("=" * 65)
    print("RISK ANALYSIS — All Tickers")
    print("=" * 65)

    for ticker in TICKERS:
        m = compute_risk_metrics(ticker)
        print(f"\n{ticker}")
        print(f"  Risk Level      : {m['risk_level']} "
              f"(score: {m['risk_score']}/3)")
        print(f"  Volatility      : {m['volatility']}% per year")
        print(f"  VaR (95%)       : {m['var_95']}% max daily loss")
        print(f"  Sharpe Ratio    : {m['sharpe_ratio']}")
        print(f"  Max Drawdown    : {m['max_drawdown']}%")
        print(f"  Total Return    : {m['total_return']}%")
        print(f"  Best Day        : +{m['best_day']}%")
        print(f"  Worst Day       : {m['worst_day']}%")

    print("\n" + "=" * 65)
    print("COMPARISON TABLE")
    print("=" * 65)
    df = compare_all_tickers(TICKERS)
    cols = ['risk_level', 'volatility', 'var_95',
            'sharpe_ratio', 'max_drawdown', 'total_return']
    print(df[cols].to_string())
    print("=" * 65)