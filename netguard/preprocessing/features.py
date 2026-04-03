"""Feature engineering and selection for network traffic data."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
import logging

logger = logging.getLogger(__name__)


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    # Skip label/target columns
    skip = {"label", "attack_cat", "is_attack"}
    categorical_cols = [c for c in categorical_cols if c not in skip]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.debug("Encoded %s: %d unique values", col, len(le.classes_))

    return df, encoders


def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numerical features to zero mean and unit variance."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_scaled, scaler


def select_features(X: pd.DataFrame, y: pd.Series, top_k: int = 30) -> list[str]:
    """Select top-k features using mutual information."""
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected = mi_series.head(top_k).index.tolist()
    logger.info("Selected %d features by mutual information", len(selected))
    return selected


def prepare_dataset(
    df: pd.DataFrame,
    target: str = "is_attack",
    top_k_features: int = 30,
    label_encoder: LabelEncoder = None,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler, list[str]]:
    """Full preprocessing pipeline: encode, select, scale.

    Args:
        df: Raw dataframe from loader
        target: Target column name ('is_attack' for binary, 'attack_cat' for multiclass)
        top_k_features: Number of features to select (0 = all)
        label_encoder: Pre-fitted LabelEncoder for multiclass (reuse on test set)

    Returns:
        (X_scaled, y, scaler, feature_names)
        For multiclass: y contains encoded integer labels
    """
    df = df.copy()

    # Drop non-feature columns
    drop_cols = ["label", "attack_cat", "is_attack", "id"]
    drop_cols = [c for c in drop_cols if c in df.columns and c != target]

    y = df[target].copy()

    # Encode target if it's categorical (multiclass)
    is_numeric = pd.api.types.is_numeric_dtype(y)
    if not is_numeric:
        if label_encoder is None:
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)
        else:
            y = pd.Series(label_encoder.transform(y.astype(str)), index=y.index)
        logger.info("Encoded %d target classes: %s", len(label_encoder.classes_), list(label_encoder.classes_))

    df.drop(columns=drop_cols + [target], errors="ignore", inplace=True)

    # Encode categoricals
    df, _ = encode_categorical(df)

    # Drop columns with all same values
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        logger.info("Dropped %d constant columns", len(constant_cols))

    # Handle missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Feature selection
    if top_k_features > 0 and len(df.columns) > top_k_features:
        selected = select_features(df, y, top_k=top_k_features)
        df = df[selected]

    feature_names = df.columns.tolist()

    # Scale
    X_scaled, scaler = scale_features(df)

    logger.info("Prepared dataset: %d samples, %d features", len(X_scaled), len(feature_names))
    return X_scaled, y, scaler, feature_names, label_encoder
