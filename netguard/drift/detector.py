"""Data and concept drift detection for network intrusion detection models.

Monitors model performance degradation over time by tracking:
1. Data drift — input feature distributions shift (KS test, PSI)
2. Prediction drift — model confidence/output distribution changes
3. Performance drift — accuracy drops when labels are available

When drift is detected, signals that retraining is needed.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Single drift alert."""
    timestamp: str
    drift_type: str  # "data", "prediction", "performance"
    severity: str  # "low", "medium", "high"
    metric_name: str
    baseline_value: float
    current_value: float
    message: str


@dataclass
class DriftReport:
    """Summary of drift analysis."""
    timestamp: str
    is_drifted: bool
    alerts: list[DriftAlert] = field(default_factory=list)
    feature_drifts: dict = field(default_factory=dict)
    prediction_drift: float = 0.0
    performance_drift: float = 0.0


class DriftDetector:
    """Monitors and detects data/concept drift in IDS models.

    Usage:
        detector = DriftDetector()
        detector.set_baseline(X_train, y_train_pred_proba)
        ...
        report = detector.check(X_new, y_new_pred_proba, y_true=labels_if_available)
    """

    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        confidence_drop_threshold: float = 0.1,
        performance_drop_threshold: float = 0.05,
        window_size: int = 1000,
    ):
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.confidence_drop_threshold = confidence_drop_threshold
        self.performance_drop_threshold = performance_drop_threshold
        self.window_size = window_size

        self.baseline_distributions = {}
        self.baseline_pred_mean = None
        self.baseline_pred_std = None
        self.baseline_accuracy = None

        self.history = deque(maxlen=100)
        self.is_baseline_set = False

    def set_baseline(
        self,
        X: pd.DataFrame,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray = None,
    ):
        """Set baseline distributions from training/validation data."""
        # Store per-feature distributions
        for col in X.columns:
            self.baseline_distributions[col] = X[col].values.copy()

        # Store prediction distribution
        if y_pred_proba.ndim == 2:
            self.baseline_pred_mean = y_pred_proba[:, 1].mean()
            self.baseline_pred_std = y_pred_proba[:, 1].std()
        else:
            self.baseline_pred_mean = y_pred_proba.mean()
            self.baseline_pred_std = y_pred_proba.std()

        # Store baseline accuracy if labels available
        if y_true is not None and y_pred_proba is not None:
            if y_pred_proba.ndim == 2:
                preds = (y_pred_proba[:, 1] >= 0.5).astype(int)
            else:
                preds = (y_pred_proba >= 0.5).astype(int)
            self.baseline_accuracy = (preds == y_true).mean()

        self.is_baseline_set = True
        logger.info(
            "Drift baseline set: %d features, pred_mean=%.4f, accuracy=%.4f",
            len(X.columns), self.baseline_pred_mean,
            self.baseline_accuracy or 0,
        )

    def check(
        self,
        X: pd.DataFrame,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray = None,
    ) -> DriftReport:
        """Check for drift against baseline.

        Args:
            X: New input features
            y_pred_proba: Model predictions (probabilities)
            y_true: True labels (optional, for performance drift)

        Returns:
            DriftReport with alerts and metrics
        """
        if not self.is_baseline_set:
            raise RuntimeError("Call set_baseline() first")

        now = datetime.now().isoformat()
        alerts = []
        feature_drifts = {}

        # 1. Data drift — KS test per feature
        drifted_features = 0
        for col in X.columns:
            if col not in self.baseline_distributions:
                continue
            baseline = self.baseline_distributions[col]
            current = X[col].values

            ks_stat, p_value = stats.ks_2samp(baseline, current)
            psi = self._calculate_psi(baseline, current)

            is_drifted = p_value < self.ks_threshold or psi > self.psi_threshold
            feature_drifts[col] = {
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_value, 4),
                "psi": round(psi, 4),
                "drifted": is_drifted,
            }

            if is_drifted:
                drifted_features += 1
                severity = "high" if psi > self.psi_threshold * 2 else "medium"
                alerts.append(DriftAlert(
                    timestamp=now,
                    drift_type="data",
                    severity=severity,
                    metric_name=f"feature:{col}",
                    baseline_value=float(baseline.mean()),
                    current_value=float(current.mean()),
                    message=f"Feature '{col}' drifted (KS p={p_value:.4f}, PSI={psi:.4f})",
                ))

        # 2. Prediction drift — confidence distribution shift
        if y_pred_proba.ndim == 2:
            current_pred = y_pred_proba[:, 1]
        else:
            current_pred = y_pred_proba

        pred_mean_diff = abs(current_pred.mean() - self.baseline_pred_mean)
        prediction_drift = pred_mean_diff / max(self.baseline_pred_std, 1e-6)

        if pred_mean_diff > self.confidence_drop_threshold:
            alerts.append(DriftAlert(
                timestamp=now,
                drift_type="prediction",
                severity="high" if pred_mean_diff > self.confidence_drop_threshold * 2 else "medium",
                metric_name="prediction_confidence",
                baseline_value=self.baseline_pred_mean,
                current_value=float(current_pred.mean()),
                message=f"Prediction distribution shifted (baseline={self.baseline_pred_mean:.4f}, current={current_pred.mean():.4f})",
            ))

        # 3. Performance drift — accuracy drop (if labels available)
        performance_drift = 0.0
        if y_true is not None and self.baseline_accuracy is not None:
            preds = (current_pred >= 0.5).astype(int)
            current_accuracy = (preds == y_true).mean()
            performance_drift = self.baseline_accuracy - current_accuracy

            if performance_drift > self.performance_drop_threshold:
                alerts.append(DriftAlert(
                    timestamp=now,
                    drift_type="performance",
                    severity="high" if performance_drift > self.performance_drop_threshold * 2 else "medium",
                    metric_name="accuracy",
                    baseline_value=self.baseline_accuracy,
                    current_value=float(current_accuracy),
                    message=f"Accuracy dropped from {self.baseline_accuracy:.4f} to {current_accuracy:.4f} (delta={performance_drift:.4f})",
                ))

        is_drifted = len(alerts) > 0
        report = DriftReport(
            timestamp=now,
            is_drifted=is_drifted,
            alerts=alerts,
            feature_drifts=feature_drifts,
            prediction_drift=prediction_drift,
            performance_drift=performance_drift,
        )

        self.history.append(report)

        if is_drifted:
            logger.warning(
                "DRIFT DETECTED: %d alerts (%d features drifted, pred_drift=%.4f, perf_drift=%.4f)",
                len(alerts), drifted_features, prediction_drift, performance_drift,
            )
        else:
            logger.info("No drift detected")

        return report

    @staticmethod
    def _calculate_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index — measures distribution shift."""
        eps = 1e-6
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())

        if max_val == min_val:
            return 0.0

        bin_edges = np.linspace(min_val, max_val, bins + 1)
        baseline_hist = np.histogram(baseline, bins=bin_edges)[0] / len(baseline) + eps
        current_hist = np.histogram(current, bins=bin_edges)[0] / len(current) + eps

        psi = np.sum((current_hist - baseline_hist) * np.log(current_hist / baseline_hist))
        return float(psi)

    def get_drift_summary(self) -> dict:
        """Get summary of recent drift checks."""
        if not self.history:
            return {"total_checks": 0, "drift_detected": 0}

        recent = list(self.history)
        return {
            "total_checks": len(recent),
            "drift_detected": sum(1 for r in recent if r.is_drifted),
            "last_check": recent[-1].timestamp,
            "last_drifted": recent[-1].is_drifted,
            "total_alerts": sum(len(r.alerts) for r in recent),
        }
