"""Evaluation metrics for intrusion detection models."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_binary(y_true, y_pred, y_proba=None, model_name: str = "Model") -> dict:
    """Evaluate binary classification (attack vs normal).

    Returns:
        dict with all metrics
    """
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                results["auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                results["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            results["auc"] = 0.0

    logger.info(
        "%s — Acc: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f | AUC: %.4f",
        model_name, results["accuracy"], results["f1"],
        results["precision"], results["recall"], results.get("auc", 0),
    )
    return results


def evaluate_multiclass(y_true, y_pred, model_name: str = "Model") -> dict:
    """Evaluate multi-class classification (attack type)."""
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    logger.info(
        "%s — Acc: %.4f | F1(macro): %.4f | F1(weighted): %.4f",
        model_name, results["accuracy"], results["f1_macro"], results["f1_weighted"],
    )
    return results


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Create comparison table from list of evaluation results."""
    df = pd.DataFrame(results)
    df.set_index("model", inplace=True)
    return df.round(4)


def plot_confusion_matrix(y_true, y_pred, labels=None, title: str = "Confusion Matrix", save_path: str = None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)
    plt.close()


def plot_roc_curves(models_data: dict, save_path: str = None):
    """Plot ROC curves for multiple models.

    Args:
        models_data: {model_name: (y_true, y_proba)}
    """
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_proba) in models_data.items():
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curves saved to %s", save_path)
    plt.close()


def print_classification_report(y_true, y_pred, model_name: str = "Model"):
    """Print detailed classification report."""
    print(f"\n{'='*60}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, zero_division=0))
