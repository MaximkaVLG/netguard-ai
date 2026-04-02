"""Download and prepare datasets for NetGuard AI.

Datasets:
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html

Note: Due to size, datasets are not included in the repository.
Run this script to download them automatically.
"""

import os
import sys
import urllib.request
import zipfile
import argparse

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")

DATASETS = {
    "nsl-kdd": {
        "urls": [
            ("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt", "KDDTrain+.txt"),
            ("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt", "KDDTest+.txt"),
        ],
        "description": "NSL-KDD - Baseline dataset (41 features, 4 attack categories)",
    },
}

MANUAL_DATASETS = {
    "unsw-nb15": {
        "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "description": "UNSW-NB15 - Primary dataset (49 features, 9 attack types, 2.5M records)",
        "instructions": (
            "1. Go to https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
            "2. Download UNSW-NB15 CSV files\n"
            "3. Place them in: data/raw/unsw-nb15/\n"
            "   Expected files: UNSW-NB15_1.csv, UNSW-NB15_2.csv, ..., UNSW-NB15_4.csv\n"
            "   Plus: UNSW-NB15_features.csv, UNSW-NB15_LIST_EVENTS.csv"
        ),
    },
    "cic-ids2017": {
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "description": "CIC-IDS2017 - Cross-validation dataset (78 features, 14 attack types)",
        "instructions": (
            "1. Go to https://www.unb.ca/cic/datasets/ids-2017.html\n"
            "2. Download the MachineLearningCSV.zip\n"
            "3. Extract to: data/raw/cic-ids2017/\n"
            "   Expected files: Friday-WorkingHours-*.csv, etc."
        ),
    },
}


def download_file(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved: {dest}")


def download_nsl_kdd():
    print("\n=== Downloading NSL-KDD ===")
    ds = DATASETS["nsl-kdd"]
    print(ds["description"])
    dest_dir = os.path.join(RAW_DIR, "nsl-kdd")
    os.makedirs(dest_dir, exist_ok=True)
    for url, filename in ds["urls"]:
        download_file(url, os.path.join(dest_dir, filename))
    print("NSL-KDD download complete.")


def show_manual_instructions(name: str):
    ds = MANUAL_DATASETS[name]
    print(f"\n=== {name.upper()} (Manual Download Required) ===")
    print(ds["description"])
    print(f"\nInstructions:\n{ds['instructions']}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for NetGuard AI")
    parser.add_argument(
        "--dataset",
        choices=["nsl-kdd", "unsw-nb15", "cic-ids2017", "all"],
        default="all",
        help="Dataset to download",
    )
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    if args.dataset in ("nsl-kdd", "all"):
        download_nsl_kdd()

    if args.dataset in ("unsw-nb15", "all"):
        show_manual_instructions("unsw-nb15")

    if args.dataset in ("cic-ids2017", "all"):
        show_manual_instructions("cic-ids2017")

    print("\n=== Done ===")
    print(f"Raw data directory: {RAW_DIR}")


if __name__ == "__main__":
    main()
