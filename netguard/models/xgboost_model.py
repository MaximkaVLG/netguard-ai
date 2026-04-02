"""XGBoost classifier for network intrusion detection."""

import joblib
import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42,
}

TUNING_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
}


class XGBDetector:
    """XGBoost-based network attack detector."""

    def __init__(self, params: dict = None):
        self.params = params or DEFAULT_PARAMS.copy()
        # Fall back to CPU if CUDA not available
        try:
            self.model = XGBClassifier(**self.params)
        except Exception:
            self.params["device"] = "cpu"
            self.model = XGBClassifier(**self.params)
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Training XGBoost on %d samples...", len(X))
        self.model.fit(X, y)
        self.is_fitted = True
        train_acc = self.model.score(X, y)
        logger.info("XGBoost training accuracy: %.4f", train_acc)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def tune(self, X: pd.DataFrame, y: pd.Series, cv: int = 3):
        logger.info("Tuning XGBoost hyperparameters (CV=%d)...", cv)
        base_params = {
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
        }
        grid = GridSearchCV(
            XGBClassifier(**base_params),
            TUNING_GRID,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X, y)
        self.model = grid.best_estimator_
        self.is_fitted = True
        logger.info("Best params: %s", grid.best_params_)
        logger.info("Best F1 (macro): %.4f", grid.best_score_)
        return grid.best_params_

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        importance = pd.Series(
            self.model.feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False)
        return importance

    def save(self, path: str):
        joblib.dump(self.model, path)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info("XGBoost model loaded from %s", path)
