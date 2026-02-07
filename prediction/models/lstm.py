"""
LSTM-based price prediction model.

Multi-step ahead forecasting using a stacked LSTM with:
- Configurable sequence length and hidden size
- Dropout regularization
- Early stopping on validation loss
- MinMax scaling of features

Designed for capturing temporal dependencies in price sequences.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. LSTMPredictor will be limited.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. LSTMPredictor will be unavailable.")

from prediction.models.base import BasePredictionModel, ModelMetrics


class _LSTMNetwork(nn.Module if HAS_TORCH else object):
    """PyTorch LSTM architecture."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for LSTMPredictor.")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class LSTMPredictor(BasePredictionModel):
    """LSTM model for stock price prediction.

    Wraps a PyTorch LSTM with sklearn-style fit/predict interface.
    Uses walk-forward validation with early stopping.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
    ) -> None:
        super().__init__(name="LSTM")
        self._seq_len = sequence_length
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._patience = patience

        self._model: "_LSTMNetwork | None" = None
        self._scaler_X: "MinMaxScaler | None" = None
        self._scaler_y: "MinMaxScaler | None" = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LSTMPredictor":
        """Train the LSTM model.

        Creates sequences of length seq_len, scales features,
        and trains with Adam optimizer + early stopping.
        """
        if not HAS_TORCH or not HAS_SKLEARN:
            raise RuntimeError("PyTorch and scikit-learn are required.")

        self._feature_names = list(X_train.columns)

        # Scale features and target
        self._scaler_X = MinMaxScaler()
        self._scaler_y = MinMaxScaler()

        X_scaled = self._scaler_X.fit_transform(X_train.values)
        y_scaled = self._scaler_y.fit_transform(
            y_train.values.reshape(-1, 1)
        ).ravel()

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        if len(X_seq) == 0:
            logger.warning("Not enough data for LSTM sequences.")
            self._is_fitted = False
            return self

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _LSTMNetwork(
            input_size=X_train.shape[1],
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(
            torch.FloatTensor(X_seq).to(device),
            torch.FloatTensor(y_seq).to(device),
        )
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scaler_X.transform(X_val.values)
            y_val_scaled = self._scaler_y.transform(
                y_val.values.reshape(-1, 1)
            ).ravel()
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_scaled)
            if len(X_val_seq) > 0:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_seq).to(device),
                    torch.FloatTensor(y_val_seq).to(device),
                )
                val_loader = DataLoader(val_dataset, batch_size=self._batch_size)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self._epochs):
            self._model.train()
            train_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self._model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(loader)

            # Validation
            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        output = self._model(X_batch)
                        val_loss += criterion(output, y_batch).item()
                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone() for k, v in self._model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= self._patience:
                    logger.info(
                        "[LSTM] Early stopping at epoch %d (val_loss=%.6f)",
                        epoch + 1,
                        val_loss,
                    )
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "[LSTM] Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                        epoch + 1,
                        self._epochs,
                        train_loss,
                        val_loss,
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "[LSTM] Epoch %d/%d — train_loss=%.6f",
                        epoch + 1,
                        self._epochs,
                        train_loss,
                    )

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._is_fitted = True
        logger.info("[LSTM] Training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from feature DataFrame."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_scaled = self._scaler_X.transform(X.values)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        if len(X_seq) == 0:
            return np.array([])

        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(device)
            preds_scaled = self._model(X_tensor).cpu().numpy()

        preds = self._scaler_y.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).ravel()

        # Pad predictions to match input length
        pad_length = len(X) - len(preds)
        if pad_length > 0:
            preds = np.concatenate([np.full(pad_length, np.nan), preds])

        return preds

    def save_model(self, path: Path) -> None:
        """Save LSTM model + scalers to disk."""
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None and HAS_TORCH:
            torch.save(self._model.state_dict(), path / "lstm_weights.pt")
        if self._scaler_X is not None:
            with open(path / "scaler_X.pkl", "wb") as f:
                pickle.dump(self._scaler_X, f)
        if self._scaler_y is not None:
            with open(path / "scaler_y.pkl", "wb") as f:
                pickle.dump(self._scaler_y, f)
        # Save architecture params
        params = {
            "seq_len": self._seq_len,
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
            "feature_names": self._feature_names,
        }
        with open(path / "lstm_params.pkl", "wb") as f:
            pickle.dump(params, f)
        logger.info("[LSTM] Model saved to %s", path)

    def load_model(self, path: Path) -> None:
        """Load LSTM model + scalers from disk."""
        if not HAS_TORCH or not HAS_SKLEARN:
            raise RuntimeError("PyTorch and scikit-learn are required.")

        with open(path / "lstm_params.pkl", "rb") as f:
            params = pickle.load(f)

        self._seq_len = params["seq_len"]
        self._hidden_size = params["hidden_size"]
        self._num_layers = params["num_layers"]
        self._dropout = params["dropout"]
        self._feature_names = params["feature_names"]

        self._model = _LSTMNetwork(
            input_size=len(self._feature_names),
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        )
        self._model.load_state_dict(
            torch.load(path / "lstm_weights.pt", map_location="cpu")
        )
        self._model.eval()

        with open(path / "scaler_X.pkl", "rb") as f:
            self._scaler_X = pickle.load(f)
        with open(path / "scaler_y.pkl", "rb") as f:
            self._scaler_y = pickle.load(f)

        self._is_fitted = True
        logger.info("[LSTM] Model loaded from %s", path)

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM input."""
        sequences, targets = [], []
        for i in range(self._seq_len, len(X)):
            sequences.append(X[i - self._seq_len : i])
            targets.append(y[i])
        if sequences:
            return np.array(sequences), np.array(targets)
        return np.array([]), np.array([])
