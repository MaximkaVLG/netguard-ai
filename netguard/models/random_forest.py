"""Random Forest classifier for network intrusion detection."""

import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": 42,
}

TUNING_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


class RFDetector:
    """Random Forest-based network attack detector."""

    def __init__(self, params: dict = None):
        self.params = params or DEFAULT_PARAMS
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Training Random Forest on %d samples...", len(X))
        self.model.fit(X, y)
        self.is_fitted = True
        train_acc = self.model.score(X, y)
        logger.info("RF training accuracy: %.4f", train_acc)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def tune(self, X: pd.DataFrame, y: pd.Series, cv: int = 3):
        logger.info("Tuning Random Forest hyperparameters (CV=%d)...", cv)
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
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
        logger.info("RF model saved to %s", path)

    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info("RF model loaded from %s", path)
