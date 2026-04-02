"""SHAP-based model explainability for network intrusion detection."""

import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Explain model predictions using SHAP values."""

    def __init__(self, model, model_type: str = "tree"):
        """
        Args:
            model: Trained model (sklearn RF, XGBoost, etc.)
            model_type: 'tree' for tree-based models, 'kernel' for others
        """
        self.model_type = model_type
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.KernelExplainer(model.predict, shap.sample(pd.DataFrame(), 100))
        self.shap_values = None

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        """Calculate SHAP values for given samples."""
        self.shap_values = self.explainer(X)
        return self.shap_values

    def explain_single(self, X: pd.DataFrame, index: int = 0) -> dict:
        """Explain a single prediction.

        Returns:
            dict with feature names and their SHAP contributions
        """
        sv = self.explainer(X.iloc[[index]])

        # Handle binary classification
        if isinstance(sv.values, list):
            values = sv.values[1][0]  # class 1 (attack)
        elif sv.values.ndim == 3:
            values = sv.values[0, :, 1]  # class 1
        else:
            values = sv.values[0]

        contributions = pd.Series(values, index=X.columns).sort_values(key=abs, ascending=False)
        return contributions.to_dict()

    def plot_summary(self, X: pd.DataFrame, save_path: str = None, max_display: int = 20):
        """Generate SHAP summary plot (feature importance)."""
        sv = self.explainer(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X, max_display=max_display, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP summary plot saved to %s", save_path)
        plt.close()

    def plot_waterfall(self, X: pd.DataFrame, index: int = 0, save_path: str = None):
        """Generate SHAP waterfall plot for a single prediction."""
        sv = self.explainer(X.iloc[[index]])
        plt.figure(figsize=(10, 6))

        if sv.values.ndim == 3:
            single = shap.Explanation(
                values=sv.values[0, :, 1],
                base_values=sv.base_values[0, 1],
                data=sv.data[0],
                feature_names=X.columns.tolist(),
            )
        else:
            single = sv[0]

        shap.waterfall_plot(single, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP waterfall plot saved to %s", save_path)
        plt.close()

    def get_top_features(self, X: pd.DataFrame, top_k: int = 10) -> list[str]:
        """Get top-k most important features based on SHAP."""
        sv = self.explainer(X)
        if sv.values.ndim == 3:
            vals = np.abs(sv.values[:, :, 1]).mean(axis=0)
        else:
            vals = np.abs(sv.values).mean(axis=0)
        importance = pd.Series(vals, index=X.columns).sort_values(ascending=False)
        return importance.head(top_k).index.tolist()
