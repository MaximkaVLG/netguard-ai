"""Ensemble model combining RF, XGBoost, and Autoencoder predictions."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """Combines multiple detectors using soft voting.

    For supervised models (RF, XGBoost): uses predict_proba
    For autoencoder: uses anomaly scores normalized to [0, 1]
    """

    def __init__(self, weights: dict = None):
        self.models = {}
        self.weights = weights or {"rf": 0.35, "xgb": 0.35, "ae": 0.30}

    def add_model(self, name: str, model):
        self.models[name] = model
        logger.info("Added %s to ensemble", name)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict using weighted soft voting.

        Returns:
            Binary predictions: 0 = normal, 1 = attack
        """
        scores = self.predict_scores(X)
        return (scores >= threshold).astype(int)

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return weighted average attack probability."""
        all_scores = {}

        for name, model in self.models.items():
            if name == "ae":
                # Autoencoder: normalize anomaly scores to [0, 1]
                raw_scores = model.predict_scores(X)
                # Clip and normalize using min-max
                min_s, max_s = raw_scores.min(), raw_scores.max()
                if max_s > min_s:
                    all_scores[name] = (raw_scores - min_s) / (max_s - min_s)
                else:
                    all_scores[name] = np.zeros(len(X))
            else:
                # Supervised models: use probability of attack class
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    all_scores[name] = proba[:, 1]
                else:
                    # Multi-class: probability of any attack = 1 - P(normal)
                    all_scores[name] = 1 - proba[:, 0]

        # Weighted average
        total_weight = sum(self.weights.get(n, 0) for n in all_scores)
        combined = np.zeros(len(X))
        for name, scores in all_scores.items():
            w = self.weights.get(name, 1.0 / len(all_scores))
            combined += scores * w

        if total_weight > 0:
            combined /= total_weight

        return combined
