"""Dataset loaders for UNSW-NB15, CIC-IDS2017, and NSL-KDD."""

import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# NSL-KDD column names (no header in raw files)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

NSL_KDD_ATTACK_MAP = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}


def load_nsl_kdd(split: str = "train") -> pd.DataFrame:
    """Load NSL-KDD dataset.

    Args:
        split: 'train' or 'test'

    Returns:
        DataFrame with features and 'label' column (Normal/DoS/Probe/R2L/U2R)
    """
    filename = "KDDTrain+.txt" if split == "train" else "KDDTest+.txt"
    path = os.path.join(DATA_DIR, "nsl-kdd", filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NSL-KDD not found at {path}. Run: python data/download_datasets.py --dataset nsl-kdd"
        )

    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    df.drop(columns=["difficulty_level"], inplace=True)

    # Map specific attacks to categories
    df["attack_cat"] = df["label"].str.strip().str.lower().map(NSL_KDD_ATTACK_MAP)
    df["attack_cat"] = df["attack_cat"].fillna("Unknown")

    # Binary label
    df["is_attack"] = (df["attack_cat"] != "Normal").astype(int)

    logger.info("Loaded NSL-KDD %s: %d rows, %d features", split, len(df), len(df.columns))
    return df


def load_unsw_nb15(split: str = "train") -> pd.DataFrame:
    """Load UNSW-NB15 dataset.

    Args:
        split: 'train', 'test', or 'all' (concatenated)

    Returns:
        DataFrame with features, 'attack_cat' and 'is_attack' columns
    """
    raw_dir = os.path.join(DATA_DIR, "unsw-nb15")

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"UNSW-NB15 not found at {raw_dir}. "
            "Run: python data/download_datasets.py --dataset unsw-nb15"
        )

    # Try parquet first (from Hugging Face), then CSV
    parquet_files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]

    if parquet_files:
        if split == "all":
            frames = [pd.read_parquet(os.path.join(raw_dir, f)) for f in parquet_files]
            df = pd.concat(frames, ignore_index=True)
        else:
            fname = f"{split}.parquet"
            path = os.path.join(raw_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            df = pd.read_parquet(path)
    else:
        csv_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv") and "feature" not in f.lower()])
        if not csv_files:
            raise FileNotFoundError(f"No data files found in {raw_dir}")

        features_file = os.path.join(raw_dir, "UNSW-NB15_features.csv")
        column_names = None
        if os.path.exists(features_file):
            features_df = pd.read_csv(features_file, encoding="latin-1")
            column_names = features_df["Name"].str.strip().tolist()

        frames = []
        for f in csv_files:
            path = os.path.join(raw_dir, f)
            chunk = pd.read_csv(path, header=0 if column_names is None else None, low_memory=False)
            if column_names and len(chunk.columns) == len(column_names):
                chunk.columns = column_names
            frames.append(chunk)
        df = pd.concat(frames, ignore_index=True)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Ensure attack_cat and label exist
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).fillna("Normal").str.strip()
    if "label" in df.columns:
        df["is_attack"] = df["label"].astype(int)
    elif "attack_cat" in df.columns:
        df["is_attack"] = (df["attack_cat"] != "Normal").astype(int)

    logger.info("Loaded UNSW-NB15 %s: %d rows, %d features", split, len(df), len(df.columns))
    return df


def load_cic_ids2017() -> pd.DataFrame:
    """Load CIC-IDS2017 dataset.

    Returns:
        DataFrame with features, 'attack_cat' and 'is_attack' columns
    """
    raw_dir = os.path.join(DATA_DIR, "cic-ids2017")

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"CIC-IDS2017 not found at {raw_dir}. "
            "Download from https://www.unb.ca/cic/datasets/ids-2017.html "
            "and place CSV files in data/raw/cic-ids2017/"
        )

    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for f in csv_files:
        path = os.path.join(raw_dir, f)
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()

    # Rename label column
    label_col = [c for c in df.columns if "label" in c.lower()]
    if label_col:
        df.rename(columns={label_col[0]: "label"}, inplace=True)
        df["attack_cat"] = df["label"].str.strip()
        df["is_attack"] = (df["attack_cat"] != "BENIGN").astype(int)

    # Clean infinities and NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    logger.info("Loaded CIC-IDS2017: %d rows, %d features", len(df), len(df.columns))
    return df


def load_dataset(name: str, **kwargs) -> pd.DataFrame:
    """Universal dataset loader.

    Args:
        name: 'unsw-nb15', 'cic-ids2017', or 'nsl-kdd'

    Returns:
        DataFrame with 'attack_cat' and 'is_attack' columns
    """
    loaders = {
        "unsw-nb15": load_unsw_nb15,
        "cic-ids2017": load_cic_ids2017,
        "nsl-kdd": load_nsl_kdd,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(loaders.keys())}")
    return loaders[name](**kwargs)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    name = sys.argv[1] if len(sys.argv) > 1 else "nsl-kdd"
    df = load_dataset(name)
    print(f"\nDataset: {name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}...")
    if "attack_cat" in df.columns:
        print(f"\nAttack distribution:\n{df['attack_cat'].value_counts()}")
    if "is_attack" in df.columns:
        print(f"\nBinary: Normal={len(df[df['is_attack']==0])}, Attack={len(df[df['is_attack']==1])}")
