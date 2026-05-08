# AI-Driven Financial Market Prediction and Risk Analysis

## Overview
End-to-end financial analytics system using ML, Deep Learning, 
and NLP for 8 global stocks across India, USA, UK, and Japan.

## Features
- LSTM neural network for next-day price forecasting
- Random Forest classifier for direction prediction  
- VADER NLP sentiment analysis on financial news
- Quantitative risk metrics: VaR, Sharpe Ratio, Max Drawdown
- Interactive Streamlit dashboard with 4 pages
- PostgreSQL database with ~10,000 rows of historical data

## Tech Stack
Python | PostgreSQL | PyTorch | scikit-learn | Streamlit | 
Plotly | VADER | yfinance | SQLAlchemy | pandas | NumPy

## Stocks Tracked
| Stock | Exchange | Country |
|-------|----------|---------|
| RELIANCE.NS | NSE | India |
| TCS.NS | NSE | India |
| INFY.NS | NSE | India |
| HDFCBANK.NS | NSE | India |
| AAPL | NASDAQ | USA |
| MSFT | NASDAQ | USA |
| HSBA.L | LSE | UK |
| 7203.T | TSE | Japan |

## How to Run
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Set up PostgreSQL and fill .env file
4. Run: streamlit run dashboard/app.py
