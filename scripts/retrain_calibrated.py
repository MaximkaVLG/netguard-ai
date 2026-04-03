"""Retrain calibrated model with diverse normal traffic (v3)."""

import sys
import os
import random
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

from scapy.all import IP, TCP, UDP, ICMP, Ether, Raw, wrpcap
from netguard.preprocessing.flow_extractor import FlowExtractor
from netguard.preprocessing.traffic_generator import (
    generate_attack_port_scan, generate_attack_syn_flood,
    generate_attack_brute_force, generate_attack_dns_amplification,
)
from netguard.evaluation.metrics import evaluate_binary

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
os.makedirs(OUT, exist_ok=True)


def gen_normal():
    packets = []
    base = 0.0

    # HTTP: very diverse sessions
    for i in range(500):
        src = f"192.168.{random.randint(1,10)}.{random.randint(10,250)}"
        dst = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        sport = random.randint(49152, 65535)
        dport = random.choice([80, 443, 8080, 8443])
        t = base + i * random.uniform(0.2, 5.0)

        # 3-way handshake
        packets.append(Ether()/IP(src=src, dst=dst)/TCP(sport=sport, dport=dport, flags="S", window=65535))
        packets[-1].time = t
        packets.append(Ether()/IP(src=dst, dst=src)/TCP(sport=dport, dport=sport, flags="SA", window=65535))
        packets[-1].time = t + random.uniform(0.005, 0.3)
        packets.append(Ether()/IP(src=src, dst=dst)/TCP(sport=sport, dport=dport, flags="A", window=65535))
        packets[-1].time = t + random.uniform(0.01, 0.35)

        npkts = random.choice([1, 2, 3, 5, 8, 15, 30, 50, 100])
        for j in range(npkts):
            size = random.randint(40, 1460)
            if random.random() < 0.3:
                packets.append(Ether()/IP(src=src, dst=dst)/TCP(sport=sport, dport=dport, flags="PA")/Raw(b"X" * size))
            else:
                packets.append(Ether()/IP(src=dst, dst=src)/TCP(sport=dport, dport=sport, flags="A")/Raw(b"X" * size))
            packets[-1].time = t + 0.3 + j * random.uniform(0.005, 1.0)

        dur = random.uniform(0.1, 120.0)
        packets.append(Ether()/IP(src=src, dst=dst)/TCP(sport=sport, dport=dport, flags="FA"))
        packets[-1].time = t + dur

    # DNS: very diverse response sizes (30-512 bytes)
    for i in range(600):
        src = f"192.168.{random.randint(1,10)}.{random.randint(10,250)}"
        dns = random.choice(["8.8.8.8", "8.8.4.4", "1.1.1.1", "77.88.8.8"])
        sport = random.randint(49152, 65535)
        t = base + 3000 + i * random.uniform(0.02, 3.0)

        q_size = random.randint(20, 80)
        packets.append(Ether()/IP(src=src, dst=dns)/UDP(sport=sport, dport=53)/Raw(b"Q" * q_size))
        packets[-1].time = t

        # Key: response sizes from 30 to 512 to cover all real DNS
        r_size = random.choice([30, 40, 50, 54, 60, 80, 100, 150, 200, 300, 400, 512])
        packets.append(Ether()/IP(src=dns, dst=src)/UDP(sport=53, dport=sport)/Raw(b"R" * r_size))
        packets[-1].time = t + random.uniform(0.003, 0.15)

    # SSH: bidirectional AND one-sided, varying lengths
    for i in range(100):
        src = f"192.168.{random.randint(1,5)}.{random.randint(10,50)}"
        dst = f"10.0.{random.randint(0,10)}.{random.randint(1,50)}"
        sport = random.randint(49152, 65535)
        t = base + 6000 + i * random.uniform(1, 30)

        # Handshake
        packets.append(Ether()/IP(src=src, dst=dst)/TCP(sport=sport, dport=22, flags="S", window=65535))
        packets[-1].time = t
        packets.append(Ether()/IP(src=dst, dst=src)/TCP(sport=22, dport=sport, flags="SA", window=65535))
        packets[-1].time = t + 0.02
        packets.append(Ether()/IP(src=src, dst=dst)/TCP(sport=sport, dport=22, flags="A", window=65535))
        packets[-1].time = t + 0.03

        session_len = random.choice([5, 10, 20, 50, 100, 200])
        is_bidir = random.random() < 0.7

        for j in range(session_len):
            size = random.randint(50, 500)
            if is_bidir and random.random() < 0.5:
                packets.append(Ether()/IP(src=dst, dst=src)/TCP(sport=22, dport=sport, flags="PA")/Raw(b"X" * size))
            else:
                packets.append(Ether()/IP(src=src, dst=dst)/TCP(sport=sport, dport=22, flags="PA")/Raw(b"X" * size))
            packets[-1].time = t + 0.1 + j * random.uniform(0.05, 2.0)

    # ICMP
    for i in range(80):
        src = f"192.168.1.{random.randint(10,50)}"
        dst = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.1"
        t = base + 10000 + i * random.uniform(0.5, 5)
        packets.append(Ether()/IP(src=src, dst=dst)/ICMP()/Raw(b"X" * 64))
        packets[-1].time = t
        packets.append(Ether()/IP(src=dst, dst=src)/ICMP(type=0)/Raw(b"X" * 64))
        packets[-1].time = t + random.uniform(0.005, 0.2)

    # Single SYN packets (connection attempts, background noise)
    for i in range(200):
        src = f"192.168.{random.randint(1,10)}.{random.randint(10,250)}"
        dst = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        sport = random.randint(49152, 65535)
        dport = random.choice([80, 443, 22, 53, 3389, 8080])
        t = base + 12000 + i * random.uniform(0.1, 2.0)
        packets.append(Ether()/IP(src=src, dst=dst)/TCP(sport=sport, dport=dport, flags="S"))
        packets[-1].time = t

    return packets


def main():
    logger.info("Generating diverse normal traffic...")
    normal_pkts = gen_normal()
    normal_pkts.sort(key=lambda p: float(p.time))
    wrpcap(os.path.join(OUT, "normal_traffic.pcap"), normal_pkts)
    logger.info("Normal: %d packets", len(normal_pkts))

    logger.info("Generating attack traffic...")
    attack_pkts = []
    attack_pkts.extend(generate_attack_port_scan(12, base_time=0))
    attack_pkts.extend(generate_attack_syn_flood(8, base_time=500))
    attack_pkts.extend(generate_attack_dns_amplification(base_time=1500))

    # Custom brute force with clear short-duration pattern
    for target_idx in range(10):
        attacker = f"10.{random.randint(50,99)}.{random.randint(0,255)}.{random.randint(1,254)}"
        target = f"192.168.{random.randint(1,5)}.{random.randint(1,254)}"
        t = 1000 + target_idx * 80
        for i in range(random.randint(80, 200)):
            sport = random.randint(49152, 65535)
            at = t + i * random.uniform(0.2, 0.8)
            # SYN
            attack_pkts.append(Ether()/IP(src=attacker, dst=target)/TCP(sport=sport, dport=22, flags="S", window=1024))
            attack_pkts[-1].time = at
            # SYN-ACK
            attack_pkts.append(Ether()/IP(src=target, dst=attacker)/TCP(sport=22, dport=sport, flags="SA"))
            attack_pkts[-1].time = at + 0.02
            # Tiny payload
            attack_pkts.append(Ether()/IP(src=attacker, dst=target)/TCP(sport=sport, dport=22, flags="PA")/Raw(b"SSH-2.0\r\n"))
            attack_pkts[-1].time = at + 0.05
            # RST — key difference: brute force gets rejected fast
            attack_pkts.append(Ether()/IP(src=target, dst=attacker)/TCP(sport=22, dport=sport, flags="R"))
            attack_pkts[-1].time = at + 0.1

    attack_pkts.sort(key=lambda p: float(p.time))
    wrpcap(os.path.join(OUT, "attack_traffic.pcap"), attack_pkts)
    logger.info("Attack: %d packets", len(attack_pkts))

    # Extract features
    ext1 = FlowExtractor(flow_timeout=120)
    normal_df = ext1.extract_from_pcap(os.path.join(OUT, "normal_traffic.pcap"))
    normal_df["label"] = 0

    ext2 = FlowExtractor(flow_timeout=120)
    attack_df = ext2.extract_from_pcap(os.path.join(OUT, "attack_traffic.pcap"))
    attack_df["label"] = 1

    combined = pd.concat([normal_df, attack_df], ignore_index=True)
    combined.drop(columns=[c for c in combined.columns if c.startswith("_")], inplace=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("Dataset: normal=%d, attack=%d, total=%d", len(normal_df), len(attack_df), len(combined))

    # Train
    y = combined["label"]
    X = combined.drop(columns=["label"])
    for col in X.select_dtypes(exclude=["number"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    features = X.columns.tolist()
    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42, stratify=y)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.9,
        eval_metric="logloss", tree_method="hist", device="cuda", random_state=42,
    )
    xgb.fit(X_train, y_train)

    pred = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)
    results = evaluate_binary(y_test, pred, proba, "XGBoost (calibrated v3)")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(xgb, os.path.join(models_dir, "xgb_model.pkl"))
    joblib.dump({"scaler": scaler, "features": features}, os.path.join(models_dir, "preprocessing.pkl"))
    combined.to_csv(os.path.join(OUT, "training_data.csv"), index=False)
    logger.info("Model saved!")


if __name__ == "__main__":
    main()
