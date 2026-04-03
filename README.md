<p align="center">
  <h1 align="center">NetGuard AI</h1>
  <p align="center">
    <strong>ML-Powered Network Intrusion Detection System with Explainable AI</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> &bull;
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#architecture">Architecture</a> &bull;
    <a href="#datasets">Datasets</a> &bull;
    <a href="#results">Results</a> &bull;
    <a href="#dashboard">Dashboard</a>
  </p>
</p>

---

An intelligent network intrusion detection system that combines multiple ML models (Random Forest, XGBoost, Autoencoder) with SHAP explainability to detect and explain network attacks in real-time.

## Features

- **Multi-Model Detection** --- Random Forest, XGBoost (supervised) + Autoencoder (unsupervised anomaly detection)
- **Explainable AI** --- SHAP-based explanations for every prediction (why was this traffic flagged?)
- **Drift Detection** --- KS-test, PSI, and performance monitoring to detect model degradation over time
- **Real-Time Analysis** --- Live traffic classification with per-connection SHAP explanations
- **Cross-Dataset Validation** --- Evaluated on UNSW-NB15, CIC-IDS2017, and NSL-KDD
- **Ensemble Voting** --- Combines models for higher accuracy and lower false positives
- **Web Dashboard** --- Streamlit-based UI with 6 interactive pages
- **Attack Types** --- DoS/DDoS, Port Scanning, Brute Force, Exploits, Backdoors, Fuzzers, Anomalies

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/netguard-ai.git
cd netguard-ai
pip install -r requirements.txt
```

### Download Dataset

```bash
python data/download_datasets.py --dataset unsw-nb15
```

### Train Models

```bash
python -m netguard.models.random_forest --train
python -m netguard.models.xgboost_model --train
python -m netguard.models.autoencoder --train
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Architecture

```
Raw Traffic / Dataset
        |
        v
+-------------------+
| Data Preprocessor |  <- Normalization, feature engineering, balancing
+--------+----------+
         |
         v
+-----------------------------+
|      Detection Engine       |
|  +---------+  +----------+  |
|  |   RF    |  | XGBoost  |  |  <- Supervised classification
|  +---------+  +----------+  |
|  +------------------------+ |
|  |     Autoencoder        | |  <- Unsupervised anomaly detection
|  +------------------------+ |
|  +------------------------+ |
|  |   Ensemble Voting      | |  <- Model combination
|  +------------------------+ |
+--------+--------------------+
         |
         v
+-------------------+
|  SHAP Explainer   |  <- Why did the model flag this traffic?
+--------+----------+
         |
         v
+-------------------+
|   Streamlit UI    |  <- Dashboard, alerts, charts, explanations
+-------------------+
```

## Datasets

| Dataset | Features | Attack Types | Records | Role |
|---------|----------|-------------|---------|------|
| **UNSW-NB15** | 49 | 9 | 2.5M | Primary evaluation |
| **CIC-IDS2017** | 78 | 14 | 2.8M | Cross-validation |
| **NSL-KDD** | 41 | 4 | 150K | Baseline comparison |

## Results (UNSW-NB15)

| Model | Accuracy | F1 | Precision | Recall | AUC |
|-------|----------|-----|-----------|--------|-----|
| **XGBoost** | **79.4%** | **86.0%** | 80.1% | **92.7%** | **89.0%** |
| Random Forest | 75.4% | 83.9% | 75.5% | 94.5% | 83.4% |
| Ensemble | 77.9% | 82.7% | 88.7% | 77.4% | 88.5% |
| Autoencoder | 60.4% | 60.4% | 94.6% | 44.3% | 66.1% |

XGBoost achieves the best balance of accuracy and recall. Autoencoder provides highest precision (94.6%) with low false positives --- useful for zero-day anomaly detection.

## Dashboard

*Screenshots will be added after implementation.*

## Project Structure

```
netguard-ai/
├── netguard/              # Core package
│   ├── preprocessing/     # Data loading, feature engineering, balancing
│   ├── models/            # RF, XGBoost, Autoencoder, Ensemble
│   ├── explainability/    # SHAP explanations
│   ├── evaluation/        # Metrics and cross-dataset evaluation
│   └── realtime/          # Packet capture demo
├── notebooks/             # Reproducible Jupyter notebooks
├── dashboard/             # Streamlit web UI
├── data/                  # Dataset download scripts
├── tests/                 # Unit tests
└── docs/                  # Documentation and figures
```

## Tech Stack

- Python 3.12+
- scikit-learn, XGBoost --- classical ML models
- PyTorch --- Autoencoder
- SHAP --- model explainability
- Streamlit --- web dashboard
- scapy --- packet capture
- pandas, numpy, matplotlib, plotly

## License

MIT License --- see [LICENSE](LICENSE) for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{netguard_ai_2026,
  title={NetGuard AI: ML-Powered Network Intrusion Detection with Explainable AI},
  author={Melnikov, Maxim},
  year={2026},
  url={https://github.com/yourusername/netguard-ai}
}
```
