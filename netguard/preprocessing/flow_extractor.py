"""Extract network flow features from raw packets (pcap files or live capture).

Converts raw network traffic into feature vectors compatible with UNSW-NB15
trained models. This is the bridge between real traffic and ML detection.

Supports:
- pcap/pcapng files
- Live network interface capture
- Individual packet processing

Extracted features match UNSW-NB15 format:
dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, sload, dload,
sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt,
synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_src_dport_ltm,
ct_dst_sport_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, is_sm_ips_ports
"""

import time
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Common port -> service mapping (matching UNSW-NB15 encoding)
PORT_SERVICE = {
    20: 'ftp-data', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
    53: 'dns', 80: 'http', 110: 'pop3', 143: 'imap', 443: 'ssl',
    993: 'imap-ssl', 995: 'pop3-ssl', 3306: 'mysql', 3389: 'rdp',
    5432: 'postgres', 8080: 'http-proxy',
}

# TCP state mapping (matching UNSW-NB15)
TCP_STATES = {
    'SYN': 'INT', 'SYN-ACK': 'INT', 'FIN': 'FIN',
    'RST': 'RST', 'established': 'CON',
}


@dataclass
class Flow:
    """Represents a bidirectional network flow (connection)."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP, 1=ICMP

    start_time: float = 0.0
    end_time: float = 0.0

    # Packet counts
    src_pkts: int = 0
    dst_pkts: int = 0

    # Byte counts
    src_bytes: int = 0
    dst_bytes: int = 0

    # Packet sizes for mean/jitter calculation
    src_pkt_sizes: list = field(default_factory=list)
    dst_pkt_sizes: list = field(default_factory=list)

    # Timestamps for inter-packet time and jitter
    src_timestamps: list = field(default_factory=list)
    dst_timestamps: list = field(default_factory=list)

    # TCP specific
    tcp_flags: list = field(default_factory=list)
    syn_time: float = 0.0
    synack_time: float = 0.0
    ack_time: float = 0.0
    src_win: int = 0
    dst_win: int = 0
    src_tcp_base_seq: int = 0
    dst_tcp_base_seq: int = 0

    # HTTP
    http_methods: int = 0
    http_trans_depth: int = 0
    response_body_len: int = 0

    # State tracking
    state: str = 'INT'
    is_finished: bool = False


class FlowExtractor:
    """Extract UNSW-NB15-compatible features from raw network packets.

    Usage:
        extractor = FlowExtractor()

        # From pcap file
        features_df = extractor.extract_from_pcap('traffic.pcap')

        # From live capture
        extractor.start_capture('eth0')
        features_df = extractor.get_features()
    """

    def __init__(self, flow_timeout: float = 120.0):
        self.flow_timeout = flow_timeout
        self.flows: dict[tuple, Flow] = {}
        self.completed_flows: list[Flow] = []

        # Connection tracking for ct_* features
        self._src_dport_history = defaultdict(list)
        self._dst_sport_history = defaultdict(list)

    def process_packet(self, pkt_data: dict):
        """Process a single parsed packet and update flow state.

        Args:
            pkt_data: dict with keys: src_ip, dst_ip, src_port, dst_port,
                      protocol, length, timestamp, tcp_flags, tcp_win, tcp_seq
        """
        src = pkt_data['src_ip']
        dst = pkt_data['dst_ip']
        sport = pkt_data.get('src_port', 0)
        dport = pkt_data.get('dst_port', 0)
        proto = pkt_data.get('protocol', 0)
        ts = pkt_data.get('timestamp', time.time())
        length = pkt_data.get('length', 0)

        # Bidirectional flow key (sorted so A->B and B->A are same flow)
        fwd_key = (src, dst, sport, dport, proto)
        rev_key = (dst, src, dport, sport, proto)

        is_reverse = rev_key in self.flows
        key = rev_key if is_reverse else fwd_key

        if key not in self.flows:
            self.flows[key] = Flow(
                src_ip=src, dst_ip=dst,
                src_port=sport, dst_port=dport,
                protocol=proto, start_time=ts,
            )

        flow = self.flows[key]
        flow.end_time = ts

        if not is_reverse:
            # Forward direction (src -> dst)
            flow.src_pkts += 1
            flow.src_bytes += length
            flow.src_pkt_sizes.append(length)
            flow.src_timestamps.append(ts)

            if 'tcp_win' in pkt_data and flow.src_win == 0:
                flow.src_win = pkt_data['tcp_win']
            if 'tcp_seq' in pkt_data and flow.src_tcp_base_seq == 0:
                flow.src_tcp_base_seq = pkt_data['tcp_seq']
        else:
            # Reverse direction (dst -> src)
            flow.dst_pkts += 1
            flow.dst_bytes += length
            flow.dst_pkt_sizes.append(length)
            flow.dst_timestamps.append(ts)

            if 'tcp_win' in pkt_data and flow.dst_win == 0:
                flow.dst_win = pkt_data['tcp_win']
            if 'tcp_seq' in pkt_data and flow.dst_tcp_base_seq == 0:
                flow.dst_tcp_base_seq = pkt_data['tcp_seq']

        # TCP flags tracking
        flags = pkt_data.get('tcp_flags', 0)
        if flags:
            flow.tcp_flags.append(flags)
            if flags & 0x02 and flow.syn_time == 0:  # SYN
                flow.syn_time = ts
            if flags & 0x12 and flow.synack_time == 0:  # SYN-ACK
                flow.synack_time = ts
            if flags & 0x10 and flow.ack_time == 0 and flow.synack_time > 0:  # ACK
                flow.ack_time = ts
            if flags & 0x04:  # RST
                flow.state = 'RST'
                flow.is_finished = True
            if flags & 0x01:  # FIN
                flow.state = 'FIN'
                flow.is_finished = True
            if flow.syn_time > 0 and flow.synack_time > 0 and flow.ack_time > 0:
                flow.state = 'CON'

        # Connection tracking
        self._src_dport_history[(src, dport)].append(ts)
        self._dst_sport_history[(dst, sport)].append(ts)

    def flush_flows(self, force: bool = False) -> list[dict]:
        """Convert completed/timed-out flows to feature dicts.

        Args:
            force: If True, flush ALL flows (for end of pcap)

        Returns:
            List of feature dicts compatible with UNSW-NB15 format
        """
        now = time.time()
        features_list = []
        to_remove = []

        for key, flow in self.flows.items():
            is_expired = (now - flow.end_time) > self.flow_timeout
            if force or flow.is_finished or is_expired:
                features = self._flow_to_features(flow)
                features_list.append(features)
                self.completed_flows.append(flow)
                to_remove.append(key)

        for key in to_remove:
            del self.flows[key]

        return features_list

    def _flow_to_features(self, flow: Flow) -> dict:
        """Convert a Flow object to UNSW-NB15-compatible feature dict."""
        dur = max(flow.end_time - flow.start_time, 1e-6)
        total_pkts = flow.src_pkts + flow.dst_pkts

        # Inter-packet times and jitter
        sinpkt = self._mean_inter_time(flow.src_timestamps)
        dinpkt = self._mean_inter_time(flow.dst_timestamps)
        sjit = self._jitter(flow.src_timestamps)
        djit = self._jitter(flow.dst_timestamps)

        # TCP RTT
        tcprtt = 0.0
        synack = 0.0
        ackdat = 0.0
        if flow.syn_time > 0 and flow.synack_time > 0:
            synack = flow.synack_time - flow.syn_time
            if flow.ack_time > 0:
                ackdat = flow.ack_time - flow.synack_time
                tcprtt = synack + ackdat

        # Connection tracking (last 100 seconds)
        ct_src_dport = len([t for t in self._src_dport_history[(flow.src_ip, flow.dst_port)]
                           if flow.end_time - t < 100])
        ct_dst_sport = len([t for t in self._dst_sport_history[(flow.dst_ip, flow.src_port)]
                           if flow.end_time - t < 100])

        # Protocol mapping
        proto_map = {6: 'tcp', 17: 'udp', 1: 'icmp'}
        proto_str = proto_map.get(flow.protocol, 'other')

        # Service detection
        service = PORT_SERVICE.get(flow.dst_port, PORT_SERVICE.get(flow.src_port, '-'))

        # Loss estimation (retransmissions approximation)
        sloss = max(0, flow.src_pkts - flow.dst_pkts) if flow.dst_pkts > 0 else 0
        dloss = max(0, flow.dst_pkts - flow.src_pkts) if flow.src_pkts > 0 else 0

        return {
            'dur': dur,
            'proto': proto_str,
            'service': service,
            'state': flow.state,
            'spkts': flow.src_pkts,
            'dpkts': flow.dst_pkts,
            'sbytes': flow.src_bytes,
            'dbytes': flow.dst_bytes,
            'rate': total_pkts / dur if dur > 0 else 0,
            'sload': (flow.src_bytes * 8) / dur if dur > 0 else 0,
            'dload': (flow.dst_bytes * 8) / dur if dur > 0 else 0,
            'sloss': sloss,
            'dloss': dloss,
            'sinpkt': sinpkt,
            'dinpkt': dinpkt,
            'sjit': sjit,
            'djit': djit,
            'swin': flow.src_win,
            'stcpb': flow.src_tcp_base_seq,
            'dtcpb': flow.dst_tcp_base_seq,
            'dwin': flow.dst_win,
            'tcprtt': tcprtt,
            'synack': synack,
            'ackdat': ackdat,
            'smean': int(np.mean(flow.src_pkt_sizes)) if flow.src_pkt_sizes else 0,
            'dmean': int(np.mean(flow.dst_pkt_sizes)) if flow.dst_pkt_sizes else 0,
            'trans_depth': flow.http_trans_depth,
            'response_body_len': flow.response_body_len,
            'ct_src_dport_ltm': ct_src_dport,
            'ct_dst_sport_ltm': ct_dst_sport,
            'is_ftp_login': 1 if flow.dst_port == 21 and flow.dst_pkts > 3 else 0,
            'ct_ftp_cmd': 0,
            'ct_flw_http_mthd': flow.http_methods,
            'is_sm_ips_ports': 1 if (flow.src_ip == flow.dst_ip and flow.src_port == flow.dst_port) else 0,
            # Metadata (not features, for display)
            '_src_ip': flow.src_ip,
            '_dst_ip': flow.dst_ip,
            '_src_port': flow.src_port,
            '_dst_port': flow.dst_port,
            '_timestamp': flow.start_time,
        }

    def extract_from_pcap(self, pcap_path: str) -> pd.DataFrame:
        """Extract features from a pcap file.

        Args:
            pcap_path: Path to .pcap or .pcapng file

        Returns:
            DataFrame with UNSW-NB15-compatible features
        """
        try:
            from scapy.all import rdpcap, IP, TCP, UDP, ICMP
        except ImportError:
            raise ImportError("scapy required: pip install scapy")

        logger.info("Reading pcap: %s", pcap_path)
        packets = rdpcap(pcap_path)
        logger.info("Loaded %d packets", len(packets))

        for pkt in packets:
            if IP not in pkt:
                continue

            pkt_data = {
                'src_ip': pkt[IP].src,
                'dst_ip': pkt[IP].dst,
                'protocol': pkt[IP].proto,
                'length': len(pkt),
                'timestamp': float(pkt.time),
            }

            if TCP in pkt:
                pkt_data['src_port'] = pkt[TCP].sport
                pkt_data['dst_port'] = pkt[TCP].dport
                pkt_data['tcp_flags'] = int(pkt[TCP].flags)
                pkt_data['tcp_win'] = pkt[TCP].window
                pkt_data['tcp_seq'] = pkt[TCP].seq
            elif UDP in pkt:
                pkt_data['src_port'] = pkt[UDP].sport
                pkt_data['dst_port'] = pkt[UDP].dport
            elif ICMP in pkt:
                pkt_data['src_port'] = 0
                pkt_data['dst_port'] = 0

            self.process_packet(pkt_data)

        # Flush all remaining flows
        features = self.flush_flows(force=True)

        if not features:
            logger.warning("No flows extracted from pcap")
            return pd.DataFrame()

        df = pd.DataFrame(features)
        logger.info("Extracted %d flows with %d features", len(df), len(df.columns))
        return df

    def extract_live(self, interface: str, duration: int = 60, callback=None):
        """Capture and extract features from live network interface.

        Args:
            interface: Network interface name (e.g., 'eth0', 'Wi-Fi')
            duration: Capture duration in seconds
            callback: Function called for each analyzed flow: callback(flow_features_dict)
        """
        try:
            from scapy.all import sniff, IP, TCP, UDP
        except ImportError:
            raise ImportError("scapy required: pip install scapy")

        logger.info("Starting live capture on %s for %ds", interface, duration)

        def packet_handler(pkt):
            if IP not in pkt:
                return

            pkt_data = {
                'src_ip': pkt[IP].src,
                'dst_ip': pkt[IP].dst,
                'protocol': pkt[IP].proto,
                'length': len(pkt),
                'timestamp': float(pkt.time),
            }

            if TCP in pkt:
                pkt_data['src_port'] = pkt[TCP].sport
                pkt_data['dst_port'] = pkt[TCP].dport
                pkt_data['tcp_flags'] = int(pkt[TCP].flags)
                pkt_data['tcp_win'] = pkt[TCP].window
                pkt_data['tcp_seq'] = pkt[TCP].seq
            elif UDP in pkt:
                pkt_data['src_port'] = pkt[UDP].sport
                pkt_data['dst_port'] = pkt[UDP].dport

            self.process_packet(pkt_data)

            # Check for completed flows
            completed = self.flush_flows()
            if callback and completed:
                for flow_feat in completed:
                    callback(flow_feat)

        import sys
        if sys.platform == "win32":
            # Windows: use L3 socket to avoid WinPcap/Npcap L2 requirement
            from scapy.all import conf as scapy_conf
            sock = scapy_conf.L3socket(iface=interface)
            sniff(opened_socket=sock, prn=packet_handler, timeout=duration, store=False)
            sock.close()
        else:
            sniff(iface=interface, prn=packet_handler, timeout=duration, store=False)

        # Flush remaining
        remaining = self.flush_flows(force=True)
        if callback:
            for flow_feat in remaining:
                callback(flow_feat)

        logger.info("Capture complete. Total flows: %d", len(self.completed_flows))

    @staticmethod
    def _mean_inter_time(timestamps: list) -> float:
        if len(timestamps) < 2:
            return 0.0
        diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        return float(np.mean(diffs)) if diffs else 0.0

    @staticmethod
    def _jitter(timestamps: list) -> float:
        if len(timestamps) < 3:
            return 0.0
        diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        return float(np.std(diffs)) if len(diffs) > 1 else 0.0
