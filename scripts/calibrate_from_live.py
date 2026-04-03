"""Calibrate NetGuard model using real network traffic.

Captures live traffic, labels it as normal, combines with attack data,
and retrains the model. This eliminates false positives on your network.

Usage (run as Administrator):
    python scripts/calibrate_from_live.py --duration 300
"""

import sys
import os
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "generated")


def capture_normal_traffic(interface: str, duration: int) -> pd.DataFrame:
    """Capture live traffic and extract flow features."""
    from netguard.preprocessing.flow_extractor import FlowExtractor

    extractor = FlowExtractor(flow_timeout=60)

    logger.info("Capturing %ds of traffic on %s...", duration, interface)
    logger.info("Browse the web normally during capture.")

    all_flows = []

    def on_flow(flow_features):
        all_flows.append(flow_features)

    try:
        extractor.extract_live(interface, duration=duration, callback=on_flow)
    except (PermissionError, OSError) as e:
        logger.error("Permission denied. Run as Administrator!")
        sys.exit(1)

    # Flush remaining
    remaining = extractor.flush_flows(force=True)
    for f in remaining:
        all_flows.append(f)

    if not all_flows:
        logger.error("No flows captured. Check your interface.")
        sys.exit(1)

    df = pd.DataFrame(all_flows)
    # Drop metadata columns
    meta_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=meta_cols, errors="ignore", inplace=True)
    df["label"] = 0  # All captured traffic is normal

    logger.info("Captured %d normal flows", len(df))
    return df


def load_attack_data() -> pd.DataFrame:
    """Load generated attack data."""
    attack_csv = os.path.join(DATA_DIR, "training_data.csv")
    if os.path.exists(attack_csv):
        all_data = pd.read_csv(attack_csv)
        attacks = all_data[all_data["label"] == 1]
        logger.info("Loaded %d attack flows from training data", len(attacks))
        return attacks

    # If no training data, generate attacks
    logger.info("No existing attack data, generating...")
    from netguard.preprocessing.traffic_generator import (
        generate_attack_port_scan, generate_attack_syn_flood,
        generate_attack_brute_force, generate_attack_dns_amplification,
    )
    from netguard.preprocessing.flow_extractor import FlowExtractor
    from scapy.all import wrpcap

    attack_pkts = []
    attack_pkts.extend(generate_attack_port_scan(10, base_time=0))
    attack_pkts.extend(generate_attack_syn_flood(5, base_time=500))
    attack_pkts.extend(generate_attack_brute_force(5, base_time=1000))
    attack_pkts.extend(generate_attack_dns_amplification(base_time=1500))
    attack_pkts.sort(key=lambda p: float(p.time))

    os.makedirs(DATA_DIR, exist_ok=True)
    pcap_path = os.path.join(DATA_DIR, "attack_traffic.pcap")
    wrpcap(pcap_path, attack_pkts)

    ext = FlowExtractor(flow_timeout=120)
    df = ext.extract_from_pcap(pcap_path)
    df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore", inplace=True)
    df["label"] = 1
    logger.info("Generated %d attack flows", len(df))
    return df


def train_model(normal_df: pd.DataFrame, attack_df: pd.DataFrame):
    """Train calibrated XGBoost model."""
    combined = pd.concat([normal_df, attack_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("Training on %d flows (normal=%d, attack=%d)",
                len(combined), (combined["label"] == 0).sum(), (combined["label"] == 1).sum())

    y = combined["label"]
    X = combined.drop(columns=["label"])

    # Encode categoricals
    for col in X.select_dtypes(exclude=["number"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    features = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb = XGBClassifier(
        n_estimators=300, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.9,
        eval_metric="logloss", tree_method="hist",
        device="cuda", random_state=42,
    )
    xgb.fit(X_train, y_train)

    from netguard.evaluation.metrics import evaluate_binary
    pred = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)
    results = evaluate_binary(y_test, pred, proba, "XGBoost (calibrated)")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_model.pkl"))
    joblib.dump({"scaler": scaler, "features": features},
                os.path.join(MODELS_DIR, "preprocessing.pkl"))

    # Also save training data
    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_csv(os.path.join(DATA_DIR, "training_data.csv"), index=False)

    logger.info("Model saved! F1=%.4f AUC=%.4f", results["f1"], results.get("auc", 0))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate NetGuard model using your real network traffic"
    )
    parser.add_argument("--interface", default=None,
                        help="Network interface (default: auto-detect)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Capture duration in seconds (default: 300)")
    parser.add_argument("--normal-csv", default=None,
                        help="Use existing CSV of normal flows instead of capturing")
    args = parser.parse_args()

    # Step 1: Get normal traffic
    if args.normal_csv:
        logger.info("Loading normal traffic from %s", args.normal_csv)
        normal_df = pd.read_csv(args.normal_csv)
        if "label" not in normal_df.columns:
            normal_df["label"] = 0
    else:
        if args.interface is None:
            from scapy.all import conf
            args.interface = str(conf.iface)
        normal_df = capture_normal_traffic(args.interface, args.duration)

    # Step 2: Load attack data
    attack_df = load_attack_data()

    # Step 3: Train
    results = train_model(normal_df, attack_df)

    print("\n" + "=" * 60)
    print("  Calibration complete!")
    print(f"  Normal flows:  {(normal_df['label'] == 0).sum()}")
    print(f"  Attack flows:  {(attack_df['label'] == 1).sum()}")
    print(f"  F1 Score:      {results['f1']:.4f}")
    print(f"  AUC:           {results.get('auc', 0):.4f}")
    print("=" * 60)
    print("\n  Model is now calibrated for your network!")
    print("  Run 'python -m netguard monitor' to test.")


if __name__ == "__main__":
    main()
