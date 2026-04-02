"""Handle class imbalance in network traffic datasets."""

import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)


def balance_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Balance dataset using SMOTE (Synthetic Minority Over-sampling)."""
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    logger.info(
        "SMOTE: %d -> %d samples. Class distribution: %s",
        len(X), len(X_balanced), dict(pd.Series(y_balanced).value_counts()),
    )
    return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)


def balance_undersample(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Balance dataset by undersampling the majority class."""
    df = X.copy()
    df["_target"] = y.values

    minority_size = df["_target"].value_counts().min()
    frames = []
    for cls in df["_target"].unique():
        cls_df = df[df["_target"] == cls]
        if len(cls_df) > minority_size:
            cls_df = resample(cls_df, n_samples=minority_size, random_state=random_state)
        frames.append(cls_df)

    balanced = pd.concat(frames, ignore_index=True)
    y_balanced = balanced.pop("_target")

    logger.info(
        "Undersample: %d -> %d samples. Class distribution: %s",
        len(X), len(balanced), dict(y_balanced.value_counts()),
    )
    return balanced, y_balanced
