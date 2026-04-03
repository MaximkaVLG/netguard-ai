"""NetGuard AI — Streamlit Dashboard for Network Intrusion Detection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("🛡️ NetGuard AI")
st.sidebar.markdown("ML-Powered Network Intrusion Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "📊 Analyze Traffic",
    "🔬 Model Comparison",
    "🔍 Explain Prediction",
    "⚡ Real-Time Monitor",
    "📉 Drift Detection",
])

# --- State ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_trained_models():
    """Load trained models if available."""
    models = {}
    for name, filename in [("rf", "rf_model.pkl"), ("xgb", "xgb_model.pkl")]:
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


# === OVERVIEW ===
if page == "🏠 Overview":
    st.title("🛡️ NetGuard AI")
    st.markdown("### ML-Powered Network Intrusion Detection System with Explainable AI")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", "4", "RF, XGBoost, AE, Ensemble")
    col2.metric("Attack Types", "9+", "DoS, Probe, Exploits...")
    col3.metric("Explainability", "SHAP", "Per-prediction")
    col4.metric("Datasets", "3", "UNSW-NB15, CIC-IDS, NSL-KDD")

    st.markdown("---")
    st.markdown("""
    #### How It Works
    1. **Upload** network traffic data (CSV) or use built-in datasets
    2. **Detect** attacks using ensemble of ML models
    3. **Explain** why each connection was flagged using SHAP
    4. **Visualize** results with interactive charts
    """)

    st.markdown("---")
    st.markdown("""
    #### Architecture
    ```
    Traffic Data → Preprocessing → [RF | XGBoost | Autoencoder] → Ensemble → SHAP → Dashboard
    ```
    """)


# === ANALYZE TRAFFIC ===
elif page == "📊 Analyze Traffic":
    st.title("📊 Analyze Network Traffic")

    source = st.radio("Data Source", ["Upload CSV", "Use NSL-KDD Test Set"])

    df = None

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload network traffic CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded, low_memory=False)
            st.success(f"Loaded {len(df)} records, {len(df.columns)} features")

    elif source == "Use NSL-KDD Test Set":
        try:
            from netguard.preprocessing.loader import load_nsl_kdd
            with st.spinner("Loading NSL-KDD test set..."):
                df = load_nsl_kdd(split="test")
            st.success(f"Loaded {len(df)} records")
        except FileNotFoundError:
            st.error("NSL-KDD not found. Run: `python data/download_datasets.py --dataset nsl-kdd`")

    if df is not None:
        st.markdown("### Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

        if "attack_cat" in df.columns:
            st.markdown("### Attack Distribution")
            counts = df["attack_cat"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, title="Attack Categories")
            st.plotly_chart(fig, use_container_width=True)

        if "is_attack" in df.columns:
            col1, col2 = st.columns(2)
            normal = len(df[df["is_attack"] == 0])
            attacks = len(df[df["is_attack"] == 1])
            col1.metric("Normal Traffic", f"{normal:,}")
            col2.metric("Attacks Detected", f"{attacks:,}")

            fig = px.bar(
                x=["Normal", "Attack"], y=[normal, attacks],
                color=["Normal", "Attack"],
                color_discrete_map={"Normal": "#2ecc71", "Attack": "#e74c3c"},
                title="Traffic Classification",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Run detection if models available
        models = load_trained_models()
        if models and st.button("🚀 Run Detection"):
            from netguard.preprocessing.features import prepare_dataset
            with st.spinner("Running ML detection..."):
                try:
                    X, y, scaler, features, _ = prepare_dataset(df)
                    results = []
                    for name, model in models.items():
                        preds = model.predict(X)
                        from netguard.evaluation.metrics import evaluate_binary
                        res = evaluate_binary(y, preds, model_name=name.upper())
                        results.append(res)

                    st.markdown("### Detection Results")
                    st.dataframe(pd.DataFrame(results).set_index("model"), use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")


# === MODEL COMPARISON ===
elif page == "🔬 Model Comparison":
    st.title("🔬 Model Comparison")

    results_path = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

    st.markdown("### How to use")
    st.markdown("""
    1. Train models using notebooks or CLI
    2. Results and plots will appear here automatically
    3. Compare Accuracy, F1, Precision, Recall, AUC across models
    """)

    # Check for saved results
    comparison_file = os.path.join(results_path, "model_comparison.csv")
    if os.path.exists(comparison_file):
        results_df = pd.read_csv(comparison_file, index_col=0)
        st.dataframe(results_df, use_container_width=True)

        # Bar chart
        fig = go.Figure()
        for metric in ["accuracy", "f1", "precision", "recall"]:
            if metric in results_df.columns:
                fig.add_trace(go.Bar(name=metric, x=results_df.index, y=results_df[metric]))
        fig.update_layout(barmode="group", title="Model Metrics Comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No results yet. Train models first using the notebooks.")

    # Show saved plots
    for plot_name in ["roc_curves.png", "confusion_matrix_rf.png", "confusion_matrix_xgb.png"]:
        plot_path = os.path.join(results_path, plot_name)
        if os.path.exists(plot_path):
            st.image(plot_path, caption=plot_name.replace("_", " ").replace(".png", "").title())


# === EXPLAIN PREDICTION ===
elif page == "🔍 Explain Prediction":
    st.title("🔍 Explain Prediction (SHAP)")

    st.markdown("""
    Upload a single network connection or select from the test set.
    The system will explain **why** the model classified it as normal or attack.
    """)

    models = load_trained_models()
    if not models:
        st.warning("No trained models found. Train models first.")
    else:
        model_name = st.selectbox("Select Model", list(models.keys()))

        try:
            from netguard.preprocessing.loader import load_nsl_kdd
            from netguard.preprocessing.features import prepare_dataset

            with st.spinner("Loading data..."):
                df = load_nsl_kdd(split="test")
                X, y, scaler, features, _ = prepare_dataset(df)

            idx = st.slider("Select sample index", 0, min(len(X) - 1, 1000), 0)

            sample = X.iloc[[idx]]
            actual = "Attack" if y.iloc[idx] == 1 else "Normal"
            predicted = "Attack" if models[model_name].predict(sample)[0] == 1 else "Normal"

            col1, col2 = st.columns(2)
            col1.metric("Actual", actual)
            col2.metric("Predicted", predicted)

            if st.button("🔍 Explain with SHAP"):
                import shap
                with st.spinner("Computing SHAP values..."):
                    explainer = shap.TreeExplainer(models[model_name])
                    sv = explainer(sample)

                    # Show top contributing features
                    if sv.values.ndim == 3:
                        vals = sv.values[0, :, 1]
                    else:
                        vals = sv.values[0]

                    contributions = pd.Series(vals, index=features).sort_values(key=abs, ascending=False)
                    top = contributions.head(15)

                    fig = px.bar(
                        x=top.values, y=top.index,
                        orientation="h",
                        color=top.values,
                        color_continuous_scale=["#2ecc71", "#95a5a6", "#e74c3c"],
                        title=f"SHAP Feature Contributions (Predicted: {predicted})",
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### Feature Values")
                    feat_df = pd.DataFrame({
                        "Feature": features,
                        "Value": sample.values[0],
                        "SHAP Contribution": vals,
                    }).sort_values("SHAP Contribution", key=abs, ascending=False)
                    st.dataframe(feat_df.head(20), use_container_width=True)

        except FileNotFoundError:
            st.error("NSL-KDD not found. Run: `python data/download_datasets.py --dataset nsl-kdd`")
        except Exception as e:
            st.error(f"Error: {e}")


# === REAL-TIME MONITOR ===
elif page == "⚡ Real-Time Monitor":
    st.title("⚡ Real-Time Traffic Monitor")

    st.markdown("""
    Analyze network traffic in real-time with ML classification and SHAP explanations.
    Upload a PCAP file or use simulated traffic from the test dataset.
    """)

    models = load_trained_models()

    mode = st.radio("Mode", ["Simulate from Test Data", "Upload PCAP File"])

    if mode == "Simulate from Test Data" and models:
        model_name = st.selectbox("Model", list(models.keys()), key="rt_model")
        batch_size = st.slider("Connections per batch", 10, 200, 50)

        if st.button("▶ Start Simulation"):
            try:
                from netguard.preprocessing.loader import load_nsl_kdd
                from netguard.preprocessing.features import prepare_dataset
                import shap

                with st.spinner("Loading data and model..."):
                    df = load_nsl_kdd(split="test")
                    X, y, scaler, features, _ = prepare_dataset(df)
                    model = models[model_name]
                    explainer = shap.TreeExplainer(model)

                # Simulate real-time batches
                total = min(len(X), batch_size * 5)
                progress = st.progress(0)
                stats_container = st.empty()
                chart_container = st.empty()
                table_container = st.empty()

                all_results = []
                total_attacks = 0
                total_normal = 0

                for batch_start in range(0, total, batch_size):
                    batch_end = min(batch_start + batch_size, total)
                    X_batch = X.iloc[batch_start:batch_end]
                    y_batch = y.iloc[batch_start:batch_end]

                    preds = model.predict(X_batch)
                    probas = model.predict_proba(X_batch)

                    # SHAP for top suspicious
                    sv = explainer(X_batch)

                    for i in range(len(X_batch)):
                        pred_label = "Attack" if preds[i] == 1 else "Normal"
                        actual_label = "Attack" if y_batch.iloc[i] == 1 else "Normal"
                        conf = float(max(probas[i]))

                        if sv.values.ndim == 3:
                            vals = sv.values[i, :, 1]
                        else:
                            vals = sv.values[i]
                        top_feat = pd.Series(vals, index=features).abs().nlargest(3)
                        top_str = ", ".join([f"{k}={v:.3f}" for k, v in top_feat.items()])

                        all_results.append({
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Actual": actual_label,
                            "Predicted": pred_label,
                            "Confidence": f"{conf:.2%}",
                            "Top SHAP Features": top_str,
                            "Correct": "Yes" if pred_label == actual_label else "No",
                        })

                        if preds[i] == 1:
                            total_attacks += 1
                        else:
                            total_normal += 1

                    # Update UI
                    progress.progress(batch_end / total)

                    col1, col2, col3, col4 = stats_container.columns(4)
                    col1.metric("Processed", batch_end)
                    col2.metric("Attacks", total_attacks)
                    col3.metric("Normal", total_normal)
                    acc = sum(1 for r in all_results if r["Correct"] == "Yes") / len(all_results)
                    col4.metric("Accuracy", f"{acc:.1%}")

                    # Attack timeline chart
                    results_df = pd.DataFrame(all_results)
                    attack_counts = results_df["Predicted"].value_counts()
                    fig = px.pie(
                        values=attack_counts.values,
                        names=attack_counts.index,
                        color=attack_counts.index,
                        color_discrete_map={"Normal": "#2ecc71", "Attack": "#e74c3c"},
                        title="Real-Time Classification",
                    )
                    chart_container.plotly_chart(fig, use_container_width=True)

                    # Recent detections table
                    table_container.dataframe(
                        pd.DataFrame(all_results[-20:]),
                        use_container_width=True,
                    )

                st.success(f"Simulation complete! Processed {total} connections.")

                # Final attack details
                from datetime import datetime
                attacks_only = [r for r in all_results if r["Predicted"] == "Attack"]
                if attacks_only:
                    st.markdown("### Detected Attacks (with SHAP explanations)")
                    st.dataframe(pd.DataFrame(attacks_only[:50]), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

    elif mode == "Upload PCAP File":
        uploaded = st.file_uploader("Upload .pcap file", type=["pcap", "pcapng"])
        if uploaded:
            st.info("PCAP analysis requires scapy. Feature coming soon.")
            st.markdown("For now, use 'Simulate from Test Data' to see real-time analysis.")

    elif not models:
        st.warning("No trained models. Train models first.")


# === DRIFT DETECTION ===
elif page == "📉 Drift Detection":
    st.title("📉 Model Drift Detection")

    st.markdown("""
    Monitor if your model's performance is degrading over time.
    Drift detection compares current data distributions against the training baseline.
    """)

    models = load_trained_models()
    if not models:
        st.warning("No trained models. Train models first.")
    else:
        model_name = st.selectbox("Model", list(models.keys()), key="drift_model")

        st.markdown("### Simulate Drift")
        st.markdown("""
        We split the test set in half: first half as "recent baseline", second half as "new data".
        Then add synthetic noise to simulate feature drift.
        """)

        noise_level = st.slider("Noise level (simulated drift)", 0.0, 2.0, 0.5, 0.1)

        if st.button("🔍 Run Drift Analysis"):
            try:
                from netguard.preprocessing.loader import load_nsl_kdd
                from netguard.preprocessing.features import prepare_dataset
                from netguard.drift.detector import DriftDetector

                with st.spinner("Running drift detection..."):
                    df = load_nsl_kdd(split="test")
                    X, y, scaler, features, _ = prepare_dataset(df)
                    model = models[model_name]

                    # Split: first half = baseline, second half = new data
                    mid = len(X) // 2
                    X_base, y_base = X.iloc[:mid], y.iloc[:mid]
                    X_new, y_new = X.iloc[mid:], y.iloc[mid:]

                    # Add noise to simulate drift
                    if noise_level > 0:
                        noise = np.random.normal(0, noise_level, X_new.shape)
                        X_new = pd.DataFrame(
                            X_new.values + noise,
                            columns=X_new.columns,
                            index=X_new.index,
                        )

                    # Get predictions
                    base_proba = model.predict_proba(X_base)
                    new_proba = model.predict_proba(X_new)

                    # Run drift detection
                    detector = DriftDetector()
                    detector.set_baseline(X_base, base_proba, y_base.values)
                    report = detector.check(X_new, new_proba, y_new.values)

                # Display results
                if report.is_drifted:
                    st.error(f"DRIFT DETECTED — {len(report.alerts)} alerts")
                else:
                    st.success("No significant drift detected")

                col1, col2, col3 = st.columns(3)
                col1.metric("Alerts", len(report.alerts))
                drifted_count = sum(1 for v in report.feature_drifts.values() if v["drifted"])
                col2.metric("Features Drifted", f"{drifted_count}/{len(report.feature_drifts)}")
                col3.metric("Performance Drop", f"{report.performance_drift:.4f}")

                # Alerts table
                if report.alerts:
                    st.markdown("### Alerts")
                    alerts_data = []
                    for a in report.alerts:
                        alerts_data.append({
                            "Type": a.drift_type,
                            "Severity": a.severity,
                            "Metric": a.metric_name,
                            "Baseline": f"{a.baseline_value:.4f}",
                            "Current": f"{a.current_value:.4f}",
                            "Message": a.message,
                        })
                    st.dataframe(pd.DataFrame(alerts_data), use_container_width=True)

                # Feature drift heatmap
                if report.feature_drifts:
                    st.markdown("### Feature Drift (PSI)")
                    drift_df = pd.DataFrame(report.feature_drifts).T
                    drift_df = drift_df.sort_values("psi", ascending=False)

                    fig = px.bar(
                        drift_df.head(20),
                        x=drift_df.head(20).index,
                        y="psi",
                        color="drifted",
                        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                        title="Population Stability Index by Feature (top 20)",
                    )
                    fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Drift Threshold")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### All Features")
                    st.dataframe(drift_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
