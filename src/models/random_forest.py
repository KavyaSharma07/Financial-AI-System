# src/models/random_forest.py

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.preprocessing import get_processed_data, get_feature_columns
from src.config import TICKERS


def train_model(ticker: str):
    """
    Trains a Random Forest on historical data for one ticker.
    Saves the trained model to the models/ folder.
    """
    print(f"\nTraining Random Forest for {ticker}...")

    # Step 1 — Get processed data with all indicators
    df       = get_processed_data(ticker)
    features = get_feature_columns()

    X = df[features]   # The 8 indicators — inputs to the model
    y = df['target']   # UP (1) or DOWN (0) — what model predicts

    # Step 2 — Split into training and testing sets
    # shuffle=False is CRITICAL for time series data
    # If you shuffle, model trains on future data and tests on past
    # That gives falsely high accuracy that won't work in real life
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"  Training rows : {len(X_train)}")
    print(f"  Testing rows  : {len(X_test)}")

    # Step 3 — Create and train the model
    # n_estimators=200 means 200 trees vote together
    # max_depth=10 limits how deep each tree grows
    # Too deep = tree memorizes training data (overfitting)
    # Too shallow = tree misses patterns (underfitting)
    # random_state=42 makes results reproducible
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1   # Use all CPU cores for faster training
    )
    model.fit(X_train, y_train)

    # Step 4 — Evaluate on test data
    predictions = model.predict(X_test)
    accuracy    = accuracy_score(y_test, predictions)

    print(f"  Accuracy: {accuracy:.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=['DOWN', 'UP']
    ))

    # Step 5 — Feature importance
    # Which indicator was most useful for the model?
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print(f"  Feature Importance:")
    for feat, score in importance.items():
        bar = '█' * int(score * 50)
        print(f"    {feat:15s} {score:.4f}  {bar}")

    # Step 6 — Save model to disk
    # We save it so the dashboard can load it instantly
    # without retraining every time
    path = f"models/rf_{ticker.replace('.', '_')}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n  Model saved to {path}")

    return model, accuracy, importance


def load_model(ticker: str):
    """Loads a previously saved Random Forest model."""
    path = f"models/rf_{ticker.replace('.', '_')}.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_direction(ticker: str) -> dict:
    """
    Makes a prediction for tomorrow using today's latest data.
    Returns direction, confidence, and probabilities.
    """
    model    = load_model(ticker)
    df       = get_processed_data(ticker)
    features = get_feature_columns()

    # Take only the most recent row — that is "today"
    latest = df[features].iloc[[-1]]

    direction   = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]

    return {
        'ticker':     ticker,
        'direction':  'UP' if direction == 1 else 'DOWN',
        'confidence': round(max(probability) * 100, 1),
        'up_prob':    round(probability[1] * 100, 1),
        'down_prob':  round(probability[0] * 100, 1)
    }


if __name__ == "__main__":
    results = {}
    for ticker in TICKERS:
        model, accuracy, importance = train_model(ticker)
        results[ticker] = round(accuracy * 100, 2)

    print("\n" + "=" * 50)
    print("SUMMARY — All tickers")
    print("=" * 50)
    for ticker, acc in results.items():
        print(f"  {ticker:15s}  Accuracy: {acc}%")
    print("=" * 50)