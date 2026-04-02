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
- **Cross-Dataset Validation** --- Evaluated on UNSW-NB15, CIC-IDS2017, and NSL-KDD
- **Ensemble Voting** --- Combines models for higher accuracy and lower false positives
- **Web Dashboard** --- Streamlit-based UI for real-time monitoring and analysis
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

## Results

*Results will be added after training.*

| Model | Dataset | Accuracy | F1 (macro) | Precision | Recall | AUC |
|-------|---------|----------|-----------|-----------|--------|-----|
| Random Forest | UNSW-NB15 | - | - | - | - | - |
| XGBoost | UNSW-NB15 | - | - | - | - | - |
| Autoencoder | UNSW-NB15 | - | - | - | - | - |
| Ensemble | UNSW-NB15 | - | - | - | - | - |

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
