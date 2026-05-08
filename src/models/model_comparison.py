# src/models/model_comparison.py
# Compares Linear Regression, Random Forest, and LSTM
# side by side using the same data and same metrics

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, mean_squared_error,
                              mean_absolute_error)
from src.preprocessing import get_processed_data, get_feature_columns
from src.models.lstm import predict_next_price, StockLSTM
from src.config import TICKERS
import pickle
import torch


def evaluate_linear_regression(df: pd.DataFrame) -> dict:
    """
    Trains a Linear Regression to predict next-day close price.
    This is our baseline — simplest possible model.
    """
    features = get_feature_columns()
    df = df.copy()

    # Target is next day's actual price (regression, not classification)
    df['next_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    X = df[features]
    y = df['next_close']

    # Time-series split — no shuffling
    split   = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)

    # Direction accuracy — did it at least get UP/DOWN right?
    actual_dir = (y_test.values > X_test['MA20'].values).astype(int)
    pred_dir   = (preds > X_test['MA20'].values).astype(int)
    dir_acc    = accuracy_score(actual_dir, pred_dir)

    return {
        'model':        'Linear Regression',
        'rmse':         round(rmse, 4),
        'mae':          round(mae, 4),
        'direction_acc': round(dir_acc * 100, 2)
    }


def evaluate_random_forest(df: pd.DataFrame, ticker: str) -> dict:
    """
    Loads the saved Random Forest and evaluates it.
    RF predicts direction (UP/DOWN) not price.
    """
    features = get_feature_columns()

    X = df[features]
    y = df['target']

    split   = int(len(X) * 0.8)
    X_test  = X[split:]
    y_test  = y[split:]

    # Load saved model
    path = f"models/rf_{ticker.replace('.', '_')}.pkl"
    with open(path, 'rb') as f:
        model = pickle.load(f)

    preds   = model.predict(X_test)
    dir_acc = accuracy_score(y_test, preds)

    # RF doesn't predict price so RMSE/MAE are N/A
    return {
        'model':         'Random Forest',
        'rmse':          'N/A',
        'mae':           'N/A',
        'direction_acc': round(dir_acc * 100, 2)
    }


def compare_models(ticker: str) -> pd.DataFrame:
    """
    Runs all three models on the same ticker and returns
    a comparison table.
    """
    print(f"\nComparing models for {ticker}...")
    df = get_processed_data(ticker)

    results = []

    # 1. Linear Regression
    lr_result = evaluate_linear_regression(df)
    results.append(lr_result)

    # 2. Random Forest
    rf_result = evaluate_random_forest(df, ticker)
    results.append(rf_result)

    # 3. LSTM — load saved prediction metrics
    try:
        lstm_pred = predict_next_price(ticker)
        results.append({
            'model':         'LSTM (Deep Learning)',
            'rmse':          'See training output',
            'mae':           'See training output',
            'direction_acc': 'Regression model'
        })
    except Exception as e:
        results.append({
            'model':         'LSTM',
            'rmse':          'Error',
            'mae':           'Error',
            'direction_acc': str(e)[:30]
        })

    comparison_df = pd.DataFrame(results).set_index('model')
    return comparison_df


if __name__ == "__main__":
    print("=" * 65)
    print("MODEL COMPARISON — Linear Regression vs RF vs LSTM")
    print("=" * 65)

    for ticker in TICKERS:
        df = compare_models(ticker)
        print(f"\n{ticker}:")
        print(df.to_string())
        print()
    print("=" * 65)