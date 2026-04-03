"""NetGuard AI CLI — Analyze network traffic from command line.

Usage:
    python -m netguard scan <pcap_file>          # Scan pcap file
    python -m netguard monitor <interface>        # Live capture
    python -m netguard monitor <interface> --duration 120
"""

import sys
import os
import argparse
import logging
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("netguard")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model_and_scaler():
    """Load trained model, scaler, and feature list."""
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    prep_path = os.path.join(MODELS_DIR, "preprocessing.pkl")

    if not os.path.exists(model_path):
        logger.error("No trained model found at %s", model_path)
        logger.error("Train models first: run notebooks/02_model_training.ipynb")
        sys.exit(1)

    model = joblib.load(model_path)
    logger.info("Loaded model: %s", model_path)

    scaler = None
    features = None
    if os.path.exists(prep_path):
        prep = joblib.load(prep_path)
        scaler = prep.get("scaler")
        features = prep.get("features")
        logger.info("Loaded preprocessing: %d features", len(features) if features else 0)

    return model, scaler, features


def prepare_flow_features(df: pd.DataFrame, scaler, feature_names: list) -> pd.DataFrame:
    """Prepare extracted flow features for model prediction."""
    from netguard.preprocessing.features import encode_categorical

    # Save metadata columns
    meta_cols = [c for c in df.columns if c.startswith("_")]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame()

    # Drop metadata
    df = df.drop(columns=meta_cols, errors="ignore")

    # Clip extreme values to UNSW-NB15 realistic ranges
    # Single-packet flows produce extreme rate values that break the scaler
    clip_ranges = {
        'rate': (0, 1_000_000),
        'sload': (0, 5_000_000_000),
        'dload': (0, 5_000_000_000),
        'sinpkt': (0, 60_000),
        'dinpkt': (0, 60_000),
        'sjit': (0, 60_000),
        'djit': (0, 60_000),
    }
    for col, (lo, hi) in clip_ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # For single-packet flows (no response), set realistic defaults
    single_pkt_mask = (df.get('spkts', 0) <= 1) & (df.get('dpkts', 0) == 0)
    if 'rate' in df.columns:
        df.loc[single_pkt_mask, 'rate'] = df.loc[single_pkt_mask, 'sbytes'] / df.loc[single_pkt_mask, 'dur'].clip(lower=0.001)
        df['rate'] = df['rate'].clip(0, 1_000_000)

    # Encode categoricals
    df, _ = encode_categorical(df)

    # Handle missing/inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Align to training features
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    # Scale
    if scaler:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    return df, meta


def cmd_scan(args):
    """Scan a pcap file for attacks."""
    from netguard.preprocessing.flow_extractor import FlowExtractor

    if not os.path.exists(args.pcap):
        logger.error("File not found: %s", args.pcap)
        sys.exit(1)

    model, scaler, features = load_model_and_scaler()

    # Extract flows from pcap
    extractor = FlowExtractor(flow_timeout=args.timeout)
    flows_df = extractor.extract_from_pcap(args.pcap)

    if flows_df.empty:
        print("No network flows found in pcap file.")
        return

    print(f"\nExtracted {len(flows_df)} network flows")
    print("=" * 70)

    # Prepare and predict
    X, meta = prepare_flow_features(flows_df, scaler, features)
    predictions = model.predict(X)

    try:
        probas = model.predict_proba(X)
        confidence = np.max(probas, axis=1)
    except Exception:
        confidence = np.ones(len(predictions))

    # SHAP explanations for attacks
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        has_shap = True
    except Exception:
        has_shap = False

    # Results
    total = len(predictions)
    attacks = int((predictions == 1).sum())
    normal = total - attacks

    print(f"\n  Total flows:  {total}")
    print(f"  Normal:       {normal}")
    print(f"  ATTACKS:      {attacks}")
    print(f"  Attack rate:  {attacks/total:.1%}")
    print("=" * 70)

    if attacks > 0:
        print("\n  DETECTED ATTACKS:")
        print("-" * 70)

        attack_indices = np.where(predictions == 1)[0]
        for i in attack_indices[:20]:  # Show max 20
            src = f"{meta.iloc[i].get('_src_ip', '?')}:{meta.iloc[i].get('_src_port', '?')}" if not meta.empty else "?"
            dst = f"{meta.iloc[i].get('_dst_ip', '?')}:{meta.iloc[i].get('_dst_port', '?')}" if not meta.empty else "?"
            conf = confidence[i]

            line = f"  [{conf:.0%}] {src} -> {dst}"

            # SHAP explanation
            if has_shap:
                if shap_values.values.ndim == 3:
                    vals = shap_values.values[i, :, 1]
                else:
                    vals = shap_values.values[i]
                top = pd.Series(vals, index=features).abs().nlargest(3)
                reasons = ", ".join([f"{k}={vals[features.index(k)]:+.3f}" for k in top.index])
                line += f"  WHY: {reasons}"

            print(line)

        if attacks > 20:
            print(f"\n  ... and {attacks - 20} more attacks")

    # Save report
    if args.output:
        report_df = flows_df.copy()
        report_df["prediction"] = ["ATTACK" if p == 1 else "normal" for p in predictions]
        report_df["confidence"] = confidence
        report_df.to_csv(args.output, index=False)
        print(f"\n  Report saved: {args.output}")


def cmd_monitor(args):
    """Monitor live network traffic."""
    from netguard.preprocessing.flow_extractor import FlowExtractor

    model, scaler, features = load_model_and_scaler()
    extractor = FlowExtractor(flow_timeout=30)

    print(f"\n  Monitoring interface: {args.interface}")
    print(f"  Duration: {args.duration}s")
    print("=" * 70)

    attack_count = 0
    flow_count = 0

    def on_flow(flow_features):
        nonlocal attack_count, flow_count
        flow_count += 1

        df = pd.DataFrame([flow_features])
        X, meta = prepare_flow_features(df, scaler, features)
        pred = model.predict(X)[0]

        if pred == 1:
            attack_count += 1
            src_ip = flow_features.get("_src_ip", "?")
            dst_ip = flow_features.get("_dst_ip", "?")
            dst_port = flow_features.get("_dst_port", "?")
            print(f"  [ALERT] Attack detected: {src_ip} -> {dst_ip}:{dst_port}")

    try:
        extractor.extract_live(args.interface, duration=args.duration, callback=on_flow)
    except (PermissionError, OSError) as e:
        if "administrator" in str(e).lower() or "10013" in str(e):
            print("\n  ERROR: Live capture requires Administrator privileges.")
            print("  Run this command in an elevated terminal (Run as Administrator).")
            print("\n  Alternative: capture traffic to pcap first, then scan:")
            print('    tcpdump -i eth0 -w capture.pcap -c 10000')
            print('    python -m netguard scan capture.pcap')
        else:
            logger.error("Capture error: %s", e)
        sys.exit(1)

    print("=" * 70)
    print(f"\n  Monitoring complete.")
    print(f"  Total flows: {flow_count}")
    print(f"  Attacks: {attack_count}")


def cmd_interfaces(args):
    """List available network interfaces."""
    try:
        from scapy.all import get_if_list, get_if_addr, conf
    except ImportError:
        print("scapy required: pip install scapy")
        return

    print("\n  Available network interfaces:")
    print("  " + "-" * 60)

    for iface in get_if_list():
        try:
            addr = get_if_addr(iface)
        except Exception:
            addr = "?"
        is_default = " (default)" if iface == str(conf.iface) else ""
        # Shorten long Windows names
        short = iface.split("{")[-1].rstrip("}") if "{" in iface else iface
        print(f"  {short:>38}  {addr:<16}{is_default}")

    print(f"\n  Default interface: {conf.iface}")
    print(f"\n  Usage: python -m netguard monitor \"{conf.iface}\" --duration 30")


def main():
    parser = argparse.ArgumentParser(
        prog="netguard",
        description="NetGuard AI — ML-Powered Network Intrusion Detection",
    )
    subparsers = parser.add_subparsers(dest="command")

    # scan
    scan_p = subparsers.add_parser("scan", help="Scan a pcap file for attacks")
    scan_p.add_argument("pcap", help="Path to .pcap or .pcapng file")
    scan_p.add_argument("-o", "--output", help="Save report to CSV")
    scan_p.add_argument("--timeout", type=float, default=120, help="Flow timeout in seconds")

    # monitor
    mon_p = subparsers.add_parser("monitor", help="Monitor live network traffic")
    mon_p.add_argument("interface", nargs="?", default=None, help="Network interface (run 'interfaces' to see list)")
    mon_p.add_argument("--duration", type=int, default=60, help="Capture duration in seconds")

    # interfaces
    subparsers.add_parser("interfaces", help="List available network interfaces")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "monitor":
        if args.interface is None:
            # Use default interface
            from scapy.all import conf
            args.interface = str(conf.iface)
            logger.info("Using default interface: %s", args.interface)
        cmd_monitor(args)
    elif args.command == "interfaces":
        cmd_interfaces(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
