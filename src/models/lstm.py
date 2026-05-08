# src/models/lstm.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import get_processed_data
from src.config import TICKERS, SEQ_LEN


# ── Model Architecture ────────────────────────────────────────

class StockLSTM(nn.Module):
    """
    A 2-layer LSTM neural network for stock price prediction.

    input_size=1   : we feed one value per time step (closing price)
    hidden_size=64 : each LSTM layer has 64 memory units
    num_layers=2   : two LSTM layers stacked on top of each other
    dropout=0.2    : randomly switches off 20% of neurons during
                     training to prevent memorizing training data
    batch_first=True: input shape is (batch, sequence, features)
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        # Final layer converts the 64 memory outputs to one price prediction
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, 1)
        out, _ = self.lstm(x)
        # Take only the last time step's output
        # That is the prediction for the next day
        return self.fc(out[:, -1, :])


# ── Sequence Builder ──────────────────────────────────────────

def build_sequences(prices: np.ndarray, seq_len: int):
    """
    Converts a flat array of prices into overlapping sequences.

    Example with seq_len=3 and prices=[100, 101, 102, 103, 104]:
    X = [[100,101,102], [101,102,103]]   ← inputs
    y = [103, 104]                        ← targets (next price)

    The model learns: given this sequence, predict the next value.
    """
    X, y = [], []
    for i in range(seq_len, len(prices)):
        X.append(prices[i - seq_len:i, 0])
        y.append(prices[i, 0])
    return np.array(X), np.array(y)


# ── Training Function ─────────────────────────────────────────

def train_lstm(ticker: str, epochs: int = 100):
    """
    Trains the LSTM model for one ticker and saves it to disk.
    """
    print(f"\nTraining LSTM for {ticker}...")

    # Step 1 — Get processed data and extract closing prices
    df     = get_processed_data(ticker)
    # Use only last 2 years for better recent price prediction
    # 504 trading days ≈ 2 years
    df     = df.tail(504).reset_index(drop=True)
    prices = df['close'].values.reshape(-1, 1)
    # MinMaxScaler expects this shape

    # Step 2 — Scale prices to 0-1 range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)
    # fit_transform learns the min and max, then scales the data
    # We MUST save this scaler — we need it to reverse the scaling later

    scaler_path = f"models/scaler_{ticker.replace('.', '_')}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Step 3 — Build sequences
    X, y = build_sequences(scaled, SEQ_LEN)

    # Step 4 — Train/test split (no shuffling — time order matters)
    split   = int(len(X) * 0.8)
    X_train = X[:split]
    X_test  = X[split:]
    y_train = y[:split]
    y_test  = y[split:]

    print(f"  Training sequences : {len(X_train)}")
    print(f"  Testing sequences  : {len(X_test)}")

    # Step 5 — Convert to PyTorch tensors
    # unsqueeze(-1) adds the feature dimension
    # Shape becomes (samples, sequence_length, 1)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    # Step 6 — Initialize model, optimizer, and loss function
    model     = StockLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam is a smart optimizer that adjusts learning speed automatically
    # lr=0.001 is the learning rate — how big each update step is

    criterion = nn.MSELoss()
    # MSE = Mean Squared Error
    # It measures how far predictions are from actual values
    # Lower MSE = model predictions are closer to real prices

    # Step 7 — Training loop
    print(f"  Training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training mode — enables dropout
        model.train()
        optimizer.zero_grad()        # Clear previous gradients
        output = model(X_train_t).squeeze()
        loss   = criterion(output, y_train_t)
        loss.backward()              # Calculate how to improve
        optimizer.step()             # Update the model weights

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()             # Evaluation mode — disables dropout
            with torch.no_grad():    # No gradient calculation needed
                test_out  = model(X_test_t).squeeze()
                test_loss = criterion(test_out, y_test_t)
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {loss.item():.6f} | "
                  f"Test Loss: {test_loss.item():.6f}")

    # Step 8 — Save the trained model
    model_path = f"models/lstm_{ticker.replace('.', '_')}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to {model_path}")

    return model, scaler

# ── Step 9: Final evaluation metrics ──────────────────
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).squeeze().numpy()
        test_preds  = model(X_test_t).squeeze().numpy()

    # Inverse transform to get actual prices
    train_preds_actual = scaler.inverse_transform(
        train_preds.reshape(-1, 1)
    ).flatten()
    test_preds_actual  = scaler.inverse_transform(
        test_preds.reshape(-1, 1)
    ).flatten()
    train_actual       = scaler.inverse_transform(
        y_train.reshape(-1, 1)
    ).flatten()
    test_actual        = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # RMSE — Root Mean Squared Error
    # Average distance between predicted and actual price
    # Lower is better. In rupees/dollars.
    train_rmse = np.sqrt(np.mean((train_preds_actual - train_actual) ** 2))
    test_rmse  = np.sqrt(np.mean((test_preds_actual  - test_actual)  ** 2))

    # MAE — Mean Absolute Error
    # Average absolute difference between predicted and actual
    # Easier to interpret than RMSE — also in rupees/dollars
    train_mae  = np.mean(np.abs(train_preds_actual - train_actual))
    test_mae   = np.mean(np.abs(test_preds_actual  - test_actual))

    print(f"\n  Evaluation Metrics:")
    print(f"  Train RMSE : {train_rmse:.4f}")
    print(f"  Test  RMSE : {test_rmse:.4f}")
    print(f"  Train MAE  : {train_mae:.4f}")
    print(f"  Test  MAE  : {test_mae:.4f}")

    return model, scaler, {
        'train_rmse': round(float(train_rmse), 4),
        'test_rmse':  round(float(test_rmse),  4),
        'train_mae':  round(float(train_mae),  4),
        'test_mae':   round(float(test_mae),   4)
    }


# ── Prediction Function ───────────────────────────────────────

def predict_next_price(ticker: str) -> dict:
    """
    Loads a saved LSTM model and predicts tomorrow's closing price.
    """
    # Load model architecture and fill it with saved weights
    model = StockLSTM()
    model_path = f"models/lstm_{ticker.replace('.', '_')}.pth"
    model.load_state_dict(
        torch.load(model_path, map_location='cpu', weights_only=True)
    )
    model.eval()

    # Load the scaler that was saved during training
    scaler_path = f"models/scaler_{ticker.replace('.', '_')}.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Get the last SEQ_LEN days of closing prices
    df     = get_processed_data(ticker)
    prices = df['close'].values[-SEQ_LEN:].reshape(-1, 1)
    scaled = scaler.transform(prices)
    # transform (not fit_transform) — use the SAME scaling as training

    # Build input tensor — shape (1, SEQ_LEN, 1)
    X = torch.tensor(
        scaled.reshape(1, SEQ_LEN, 1),
        dtype=torch.float32
    )

    # Predict
    with torch.no_grad():
        pred_scaled = model(X).item()

    # Reverse the scaling to get actual rupee price
    pred_price    = scaler.inverse_transform([[pred_scaled]])[0][0]
    current_price = df['close'].iloc[-1]
    change_pct    = ((pred_price - current_price) / current_price) * 100

    return {
        'ticker':          ticker,
        'current_price':   round(float(current_price), 2),
        'predicted_price': round(float(pred_price), 2),
        'change_pct':      round(float(change_pct), 2)
    }


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    all_metrics = {}
    for ticker in TICKERS:
        model, scaler, metrics = train_lstm(ticker, epochs=100)
        all_metrics[ticker] = metrics

    print("\n" + "=" * 65)
    print("LSTM EVALUATION METRICS SUMMARY")
    print("=" * 65)
    print(f"{'Ticker':15s} {'Train RMSE':>12} {'Test RMSE':>12} "
          f"{'Train MAE':>12} {'Test MAE':>12}")
    print("-" * 65)
    for ticker, m in all_metrics.items():
        print(f"{ticker:15s} {m['train_rmse']:>12.4f} "
              f"{m['test_rmse']:>12.4f} "
              f"{m['train_mae']:>12.4f} "
              f"{m['test_mae']:>12.4f}")
    print("=" * 65)

    print("\n" + "=" * 50)
    print("PREDICTIONS — Next day closing prices")
    print("=" * 50)
    for ticker in TICKERS:
        result = predict_next_price(ticker)
        direction = "▲" if result['change_pct'] > 0 else "▼"
        print(f"  {ticker:15s} "
              f"Current: {result['current_price']:8.2f}  "
              f"Predicted: {result['predicted_price']:8.2f}  "
              f"{direction} {abs(result['change_pct']):.2f}%")
    print("=" * 50)