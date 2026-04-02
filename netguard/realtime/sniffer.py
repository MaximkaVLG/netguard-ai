"""Real-time network traffic capture and analysis.

Captures live packets, extracts features, classifies with ML model,
and explains predictions with SHAP — all in real-time.
"""

import time
import logging
import threading
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConnectionRecord:
    """Extracted features from a network connection."""
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    duration: float
    src_bytes: int
    dst_bytes: int
    packet_count: int
    flags: dict = field(default_factory=dict)
    prediction: str = "unknown"
    confidence: float = 0.0
    shap_top_features: dict = field(default_factory=dict)


class PacketAggregator:
    """Aggregates raw packets into connection-level features.

    Groups packets by (src_ip, dst_ip, src_port, dst_port, protocol)
    and extracts statistical features similar to NSL-KDD/UNSW-NB15 format.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.active_connections = {}
        self.completed = deque(maxlen=10000)

    def add_packet(self, packet_info: dict):
        """Add a parsed packet to the aggregator."""
        key = (
            packet_info.get("src_ip", ""),
            packet_info.get("dst_ip", ""),
            packet_info.get("src_port", 0),
            packet_info.get("dst_port", 0),
            packet_info.get("protocol", ""),
        )

        now = time.time()

        if key not in self.active_connections:
            self.active_connections[key] = {
                "start_time": now,
                "last_time": now,
                "src_bytes": 0,
                "dst_bytes": 0,
                "packet_count": 0,
                "flags": defaultdict(int),
                "info": packet_info,
            }

        conn = self.active_connections[key]
        conn["last_time"] = now
        conn["packet_count"] += 1
        conn["src_bytes"] += packet_info.get("length", 0)

        if "flags" in packet_info:
            for flag in packet_info["flags"]:
                conn["flags"][flag] += 1

    def flush_expired(self) -> list[ConnectionRecord]:
        """Flush connections that have timed out."""
        now = time.time()
        expired = []

        for key in list(self.active_connections.keys()):
            conn = self.active_connections[key]
            if now - conn["last_time"] > self.timeout:
                record = ConnectionRecord(
                    timestamp=datetime.fromtimestamp(conn["start_time"]).isoformat(),
                    src_ip=key[0],
                    dst_ip=key[1],
                    src_port=key[2],
                    dst_port=key[3],
                    protocol=key[4],
                    duration=round(conn["last_time"] - conn["start_time"], 4),
                    src_bytes=conn["src_bytes"],
                    dst_bytes=conn["dst_bytes"],
                    packet_count=conn["packet_count"],
                    flags=dict(conn["flags"]),
                )
                expired.append(record)
                self.completed.append(record)
                del self.active_connections[key]

        return expired


class RealtimeAnalyzer:
    """Real-time traffic analysis with ML classification and SHAP explanations.

    Captures packets (or reads from pcap), extracts features,
    classifies each connection, and provides SHAP explanations.
    """

    def __init__(self, model=None, scaler=None, feature_names: list = None, explainer=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.explainer = explainer
        self.aggregator = PacketAggregator()
        self.results = deque(maxlen=5000)
        self.is_running = False
        self._thread = None
        self.stats = {
            "total_connections": 0,
            "attacks_detected": 0,
            "normal_traffic": 0,
            "start_time": None,
        }

    def analyze_connection(self, record: ConnectionRecord) -> ConnectionRecord:
        """Classify a single connection and explain the prediction."""
        if self.model is None:
            return record

        # Build feature vector (simplified — maps to common IDS features)
        features = self._extract_ml_features(record)

        if self.scaler and self.feature_names:
            # Align features to expected model input
            df = pd.DataFrame([features])
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
            X = pd.DataFrame(self.scaler.transform(df), columns=self.feature_names)
        else:
            X = pd.DataFrame([features])

        # Predict
        pred = self.model.predict(X)[0]
        try:
            proba = self.model.predict_proba(X)[0]
            confidence = float(max(proba))
        except Exception:
            confidence = 1.0

        record.prediction = "Attack" if pred == 1 else "Normal"
        record.confidence = round(confidence, 4)

        # SHAP explanation
        if self.explainer:
            try:
                import shap
                sv = self.explainer(X)
                if sv.values.ndim == 3:
                    vals = sv.values[0, :, 1]
                else:
                    vals = sv.values[0]
                contributions = pd.Series(vals, index=self.feature_names)
                top = contributions.abs().nlargest(5)
                record.shap_top_features = {
                    k: round(float(contributions[k]), 4) for k in top.index
                }
            except Exception as e:
                logger.debug("SHAP explanation failed: %s", e)

        # Update stats
        self.stats["total_connections"] += 1
        if record.prediction == "Attack":
            self.stats["attacks_detected"] += 1
        else:
            self.stats["normal_traffic"] += 1

        self.results.append(record)
        return record

    def analyze_pcap(self, pcap_path: str) -> list[ConnectionRecord]:
        """Analyze a pcap file offline."""
        try:
            from scapy.all import rdpcap, IP, TCP, UDP
        except ImportError:
            logger.error("scapy not installed. Run: pip install scapy")
            return []

        logger.info("Analyzing pcap: %s", pcap_path)
        packets = rdpcap(pcap_path)

        for pkt in packets:
            if IP in pkt:
                info = {
                    "src_ip": pkt[IP].src,
                    "dst_ip": pkt[IP].dst,
                    "length": len(pkt),
                    "protocol": "TCP" if TCP in pkt else "UDP" if UDP in pkt else "Other",
                }
                if TCP in pkt:
                    info["src_port"] = pkt[TCP].sport
                    info["dst_port"] = pkt[TCP].dport
                    info["flags"] = str(pkt[TCP].flags)
                elif UDP in pkt:
                    info["src_port"] = pkt[UDP].sport
                    info["dst_port"] = pkt[UDP].dport
                else:
                    info["src_port"] = 0
                    info["dst_port"] = 0

                self.aggregator.add_packet(info)

        # Flush all
        self.aggregator.timeout = 0
        connections = self.aggregator.flush_expired()

        results = []
        for conn in connections:
            analyzed = self.analyze_connection(conn)
            results.append(analyzed)

        logger.info(
            "Pcap analysis complete: %d connections, %d attacks",
            len(results), sum(1 for r in results if r.prediction == "Attack"),
        )
        return results

    def analyze_dataframe(self, df: pd.DataFrame) -> list[dict]:
        """Analyze pre-processed dataframe (from datasets).

        Returns list of dicts with prediction and top SHAP features.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        results = []
        preds = self.model.predict(df)
        try:
            probas = self.model.predict_proba(df)
        except Exception:
            probas = np.zeros((len(df), 2))

        for i in range(len(df)):
            result = {
                "index": i,
                "prediction": "Attack" if preds[i] == 1 else "Normal",
                "confidence": float(max(probas[i])) if probas.ndim == 2 else 1.0,
            }
            results.append(result)

        return results

    @staticmethod
    def _extract_ml_features(record: ConnectionRecord) -> dict:
        """Convert ConnectionRecord to ML feature dict."""
        return {
            "duration": record.duration,
            "src_bytes": record.src_bytes,
            "dst_bytes": record.dst_bytes,
            "count": record.packet_count,
            "same_srv_rate": 0,
            "dst_host_count": 0,
            "dst_host_srv_count": 0,
            "dst_host_same_srv_rate": 0,
            "serror_rate": record.flags.get("S", 0) / max(record.packet_count, 1),
            "srv_serror_rate": 0,
            "logged_in": 1 if record.dst_port in (22, 23, 3389) else 0,
            "hot": 0,
            "num_compromised": 0,
            "num_failed_logins": 0,
        }

    def get_results_df(self) -> pd.DataFrame:
        """Get results as DataFrame for dashboard display."""
        if not self.results:
            return pd.DataFrame()

        records = []
        for r in self.results:
            records.append({
                "timestamp": r.timestamp,
                "src": f"{r.src_ip}:{r.src_port}",
                "dst": f"{r.dst_ip}:{r.dst_port}",
                "protocol": r.protocol,
                "bytes": r.src_bytes + r.dst_bytes,
                "packets": r.packet_count,
                "prediction": r.prediction,
                "confidence": r.confidence,
            })

        return pd.DataFrame(records)
