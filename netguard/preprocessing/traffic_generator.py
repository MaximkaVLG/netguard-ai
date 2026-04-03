"""Generate labeled training data by creating realistic network traffic.

Creates pcap files with known normal and attack traffic, runs them through
our FlowExtractor, and produces labeled training data.

This ensures the ML model is trained on the EXACT feature distributions
that it will see in production — solving the domain shift problem.
"""

import os
import random
import logging
import pandas as pd
from scapy.all import (
    IP, TCP, UDP, ICMP, DNS, DNSQR, Ether, Raw,
    wrpcap, RandIP, RandShort,
)

logger = logging.getLogger(__name__)


def generate_normal_http(n_sessions: int = 200, base_time: float = 0.0) -> list:
    """Generate realistic HTTP browsing sessions."""
    packets = []
    for i in range(n_sessions):
        src_ip = f"192.168.{random.randint(1,5)}.{random.randint(10,250)}"
        dst_ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        sport = random.randint(49152, 65535)
        dport = random.choice([80, 443, 8080, 8443])
        t = base_time + i * random.uniform(0.5, 5.0)

        # SYN
        packets.append(Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=dport, flags="S", seq=1000, window=65535))
        packets[-1].time = t

        # SYN-ACK
        packets.append(Ether()/IP(src=dst_ip, dst=src_ip)/TCP(sport=dport, dport=sport, flags="SA", seq=2000, ack=1001, window=65535))
        packets[-1].time = t + random.uniform(0.01, 0.1)

        # ACK
        packets.append(Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=dport, flags="A", seq=1001, ack=2001, window=65535))
        packets[-1].time = t + random.uniform(0.02, 0.12)

        # Data exchange (request + response)
        req_size = random.randint(100, 800)
        packets.append(Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=dport, flags="PA", seq=1001, ack=2001)/Raw(b"GET / HTTP/1.1\r\n" + b"X" * req_size))
        packets[-1].time = t + random.uniform(0.05, 0.15)

        # Response packets (multiple)
        resp_pkts = random.randint(3, 20)
        for j in range(resp_pkts):
            resp_size = random.randint(500, 1460)
            packets.append(Ether()/IP(src=dst_ip, dst=src_ip)/TCP(sport=dport, dport=sport, flags="A", seq=2001 + j * 1000, ack=1001 + req_size)/Raw(b"X" * resp_size))
            packets[-1].time = t + 0.2 + j * random.uniform(0.01, 0.1)

        # ACKs from client
        for j in range(resp_pkts // 2):
            packets.append(Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=dport, flags="A", seq=1001 + req_size, ack=2001 + (j + 1) * 2000))
            packets[-1].time = t + 0.3 + j * 0.05

        # FIN
        duration = random.uniform(1.0, 30.0)
        packets.append(Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=dport, flags="FA"))
        packets[-1].time = t + duration
        packets.append(Ether()/IP(src=dst_ip, dst=src_ip)/TCP(sport=dport, dport=sport, flags="FA"))
        packets[-1].time = t + duration + 0.05

    return packets


def generate_normal_dns(n_queries: int = 300, base_time: float = 0.0) -> list:
    """Generate realistic DNS queries and responses."""
    packets = []
    dns_servers = ["8.8.8.8", "8.8.4.4", "1.1.1.1", "77.88.8.8"]
    domains = ["google.com", "yandex.ru", "vk.com", "github.com", "stackoverflow.com",
               "youtube.com", "mail.ru", "telegram.org", "cloudflare.com", "amazon.com"]

    for i in range(n_queries):
        src_ip = f"192.168.{random.randint(1,5)}.{random.randint(10,250)}"
        dns_srv = random.choice(dns_servers)
        sport = random.randint(49152, 65535)
        domain = random.choice(domains)
        t = base_time + i * random.uniform(0.1, 3.0)

        # Query
        packets.append(Ether()/IP(src=src_ip, dst=dns_srv)/UDP(sport=sport, dport=53)/DNS(rd=1, qd=DNSQR(qname=domain)))
        packets[-1].time = t

        # Response
        packets.append(Ether()/IP(src=dns_srv, dst=src_ip)/UDP(sport=53, dport=sport)/DNS(rd=1, qd=DNSQR(qname=domain))/Raw(b"X" * random.randint(50, 200)))
        packets[-1].time = t + random.uniform(0.005, 0.05)

    return packets


def generate_normal_ssh(n_sessions: int = 30, base_time: float = 0.0) -> list:
    """Generate realistic SSH sessions (long-lived, encrypted)."""
    packets = []
    for i in range(n_sessions):
        src_ip = f"192.168.{random.randint(1,3)}.{random.randint(10,50)}"
        dst_ip = f"10.0.{random.randint(0,5)}.{random.randint(1,20)}"
        sport = random.randint(49152, 65535)
        t = base_time + i * random.uniform(5, 30)

        # Handshake
        packets.append(Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=22, flags="S", window=65535))
        packets[-1].time = t
        packets.append(Ether()/IP(src=dst_ip, dst=src_ip)/TCP(sport=22, dport=sport, flags="SA", window=65535))
        packets[-1].time = t + 0.02
        packets.append(Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=22, flags="A", window=65535))
        packets[-1].time = t + 0.03

        # Data exchange (encrypted SSH traffic — variable sizes)
        n_exchanges = random.randint(20, 200)
        for j in range(n_exchanges):
            size = random.randint(50, 500)
            if random.random() < 0.6:
                packets.append(Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=22, flags="PA")/Raw(b"X" * size))
            else:
                packets.append(Ether()/IP(src=dst_ip, dst=src_ip)/TCP(sport=22, dport=sport, flags="PA")/Raw(b"X" * size))
            packets[-1].time = t + 0.1 + j * random.uniform(0.1, 2.0)

    return packets


def generate_attack_port_scan(n_targets: int = 5, base_time: float = 0.0) -> list:
    """Generate port scan attacks (SYN scan)."""
    packets = []
    for target_idx in range(n_targets):
        attacker = f"10.{random.randint(50,99)}.{random.randint(0,255)}.{random.randint(1,254)}"
        target = f"192.168.{random.randint(1,5)}.{random.randint(1,254)}"
        t = base_time + target_idx * 10

        # Scan 100-1024 ports rapidly
        ports = list(range(1, 1025))
        random.shuffle(ports)
        for i, port in enumerate(ports[:random.randint(100, 500)]):
            packets.append(Ether()/IP(src=attacker, dst=target)/TCP(sport=random.randint(40000, 60000), dport=port, flags="S"))
            packets[-1].time = t + i * random.uniform(0.0005, 0.005)

    return packets


def generate_attack_syn_flood(n_waves: int = 3, base_time: float = 0.0) -> list:
    """Generate SYN flood (DDoS) attacks."""
    packets = []
    for wave in range(n_waves):
        target = f"192.168.{random.randint(1,5)}.{random.randint(1,254)}"
        target_port = random.choice([80, 443, 8080])
        t = base_time + wave * 30

        # 200-500 SYNs from random IPs in short time
        for i in range(random.randint(200, 500)):
            src = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            packets.append(Ether()/IP(src=src, dst=target)/TCP(sport=random.randint(1024, 65535), dport=target_port, flags="S"))
            packets[-1].time = t + i * random.uniform(0.001, 0.01)

    return packets


def generate_attack_brute_force(n_targets: int = 3, base_time: float = 0.0) -> list:
    """Generate SSH/RDP brute force attacks.

    Key characteristics that distinguish from normal SSH:
    - Very short connection duration (< 2s vs minutes/hours)
    - Very small packets (< 100 bytes vs 200-500)
    - Many connections from same IP in short time (high ct_src_dport_ltm)
    - RST/FIN after 1-3 packets (failed login)
    - All connections to same port from same source
    """
    packets = []
    for target_idx in range(n_targets):
        attacker = f"10.{random.randint(50,99)}.{random.randint(0,255)}.{random.randint(1,254)}"
        target = f"192.168.{random.randint(1,5)}.{random.randint(1,254)}"
        target_port = random.choice([22, 3389])
        t = base_time + target_idx * 60

        # 50-200 rapid connection attempts from SAME attacker IP
        n_attempts = random.randint(50, 200)
        for i in range(n_attempts):
            sport = random.randint(49152, 65535)
            attempt_t = t + i * random.uniform(0.2, 0.8)  # Fast: 1-5 attempts/sec

            # Pattern 1: SYN only (port closed or filtered) — 30%
            # Pattern 2: SYN-SYN/ACK-RST (connection refused) — 40%
            # Pattern 3: SYN-SYN/ACK-small data-RST (failed auth) — 30%
            pattern = random.choices([1, 2, 3], weights=[30, 40, 30])[0]

            # SYN
            packets.append(Ether()/IP(src=attacker, dst=target)/TCP(
                sport=sport, dport=target_port, flags="S", window=1024))
            packets[-1].time = attempt_t

            if pattern >= 2:
                # SYN-ACK
                packets.append(Ether()/IP(src=target, dst=attacker)/TCP(
                    sport=target_port, dport=sport, flags="SA", window=65535))
                packets[-1].time = attempt_t + random.uniform(0.01, 0.05)

            if pattern == 3:
                # Small data (login attempt — very small payload)
                payload = random.choice([
                    b"SSH-2.0-libssh\r\n",
                    b"SSH-2.0-paramiko\r\n",
                    b"\x00\x00\x00\x1c",
                ])
                packets.append(Ether()/IP(src=attacker, dst=target)/TCP(
                    sport=sport, dport=target_port, flags="PA")/Raw(payload))
                packets[-1].time = attempt_t + random.uniform(0.05, 0.15)

            # RST or FIN (quick disconnect)
            if pattern >= 2:
                rst_flag = random.choice(["R", "R", "FA"])
                packets.append(Ether()/IP(src=target, dst=attacker)/TCP(
                    sport=target_port, dport=sport, flags=rst_flag))
                packets[-1].time = attempt_t + random.uniform(0.1, 0.3)

    return packets


def generate_attack_dns_amplification(base_time: float = 0.0) -> list:
    """Generate DNS amplification attack (large responses to victim)."""
    packets = []
    victim = f"192.168.{random.randint(1,5)}.{random.randint(1,254)}"
    dns_servers = [f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(10)]

    for i in range(200):
        dns_srv = random.choice(dns_servers)
        # Large DNS response (amplified) to victim
        packets.append(Ether()/IP(src=dns_srv, dst=victim)/UDP(sport=53, dport=random.randint(1024, 65535))/Raw(b"X" * random.randint(800, 4000)))
        packets[-1].time = base_time + i * random.uniform(0.005, 0.02)

    return packets


def generate_training_dataset(output_dir: str) -> pd.DataFrame:
    """Generate complete labeled training dataset.

    Creates pcap files with known traffic, extracts features using our
    FlowExtractor, and returns labeled DataFrame.

    Returns:
        DataFrame with flow features + 'label' column (0=normal, 1=attack)
    """
    from netguard.preprocessing.flow_extractor import FlowExtractor

    os.makedirs(output_dir, exist_ok=True)
    all_features = []

    # === NORMAL TRAFFIC ===
    logger.info("Generating normal traffic...")
    normal_packets = []
    normal_packets.extend(generate_normal_http(300, base_time=0))
    normal_packets.extend(generate_normal_dns(500, base_time=2000))
    normal_packets.extend(generate_normal_ssh(50, base_time=4000))
    normal_packets.sort(key=lambda p: float(p.time))

    normal_pcap = os.path.join(output_dir, "normal_traffic.pcap")
    wrpcap(normal_pcap, normal_packets)
    logger.info("Normal: %d packets -> %s", len(normal_packets), normal_pcap)

    extractor = FlowExtractor(flow_timeout=120)
    normal_df = extractor.extract_from_pcap(normal_pcap)
    normal_df["label"] = 0
    all_features.append(normal_df)
    logger.info("Normal flows: %d", len(normal_df))

    # === ATTACK TRAFFIC ===
    logger.info("Generating attack traffic...")
    attack_packets = []
    attack_packets.extend(generate_attack_port_scan(8, base_time=0))
    attack_packets.extend(generate_attack_syn_flood(5, base_time=500))
    attack_packets.extend(generate_attack_brute_force(5, base_time=1000))
    attack_packets.extend(generate_attack_dns_amplification(base_time=1500))
    attack_packets.sort(key=lambda p: float(p.time))

    attack_pcap = os.path.join(output_dir, "attack_traffic.pcap")
    wrpcap(attack_pcap, attack_packets)
    logger.info("Attack: %d packets -> %s", len(attack_packets), attack_pcap)

    extractor2 = FlowExtractor(flow_timeout=120)
    attack_df = extractor2.extract_from_pcap(attack_pcap)
    attack_df["label"] = 1
    all_features.append(attack_df)
    logger.info("Attack flows: %d", len(attack_df))

    # === COMBINE ===
    combined = pd.concat(all_features, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop metadata columns
    meta_cols = [c for c in combined.columns if c.startswith("_")]
    combined.drop(columns=meta_cols, errors="ignore", inplace=True)

    csv_path = os.path.join(output_dir, "training_data.csv")
    combined.to_csv(csv_path, index=False)

    logger.info(
        "Training dataset: %d flows (normal=%d, attack=%d) -> %s",
        len(combined),
        (combined["label"] == 0).sum(),
        (combined["label"] == 1).sum(),
        csv_path,
    )

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated")
    df = generate_training_dataset(output)
    print(f"\nDataset shape: {df.shape}")
    print(f"Normal: {(df['label'] == 0).sum()}")
    print(f"Attack: {(df['label'] == 1).sum()}")
    print(f"Features: {[c for c in df.columns if c != 'label']}")
