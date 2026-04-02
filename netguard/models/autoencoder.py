"""PyTorch Autoencoder for unsupervised anomaly detection in network traffic."""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class AutoencoderNet(nn.Module):
    """Symmetric autoencoder network."""

    def __init__(self, input_dim: int, encoding_dim: int = 16):
        super().__init__()
        mid = (input_dim + encoding_dim) // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.BatchNorm1d(mid),
            nn.Dropout(0.2),
            nn.Linear(mid, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, mid),
            nn.ReLU(),
            nn.BatchNorm1d(mid),
            nn.Dropout(0.2),
            nn.Linear(mid, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AEDetector:
    """Autoencoder-based anomaly detector.

    Trained on normal traffic only. High reconstruction error = anomaly (attack).
    """

    def __init__(self, encoding_dim: int = 16, threshold_percentile: float = 95.0):
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_fitted = False

    def train(
        self,
        X_normal: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        """Train autoencoder on normal (non-attack) traffic only."""
        logger.info(
            "Training Autoencoder on %d normal samples (device: %s)...",
            len(X_normal), self.device,
        )

        input_dim = X_normal.shape[1]
        self.model = AutoencoderNet(input_dim, self.encoding_dim).to(self.device)

        tensor = torch.FloatTensor(X_normal.values).to(self.device)
        dataset = TensorDataset(tensor, tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d, Loss: %.6f", epoch + 1, epochs, avg_loss)

        # Calculate threshold from reconstruction errors on training data
        errors = self._reconstruction_error(X_normal)
        self.threshold = np.percentile(errors, self.threshold_percentile)
        self.is_fitted = True
        logger.info("AE threshold (%.1f%%): %.6f", self.threshold_percentile, self.threshold)

    def _reconstruction_error(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X.values).to(self.device)
            output = self.model(tensor)
            errors = torch.mean((tensor - output) ** 2, dim=1).cpu().numpy()
        return errors

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict: 0 = normal, 1 = anomaly (attack)."""
        errors = self._reconstruction_error(X)
        return (errors > self.threshold).astype(int)

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (reconstruction errors)."""
        return self._reconstruction_error(X)

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "threshold": self.threshold,
            "encoding_dim": self.encoding_dim,
            "input_dim": self.model.encoder[0].in_features,
        }, path)
        logger.info("AE model saved to %s", path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.encoding_dim = checkpoint["encoding_dim"]
        self.threshold = checkpoint["threshold"]
        self.model = AutoencoderNet(
            checkpoint["input_dim"], self.encoding_dim
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.is_fitted = True
        logger.info("AE model loaded from %s", path)
