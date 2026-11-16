
import streamlit as st
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

# Page configuration
st.set_page_config(
    page_title="Moderna mRNA Safety Intelligence",
    page_icon="🧬",
    layout="wide"
)

# Dashboard title
st.title("🧬 Moderna mRNA Vaccine Safety Intelligence")
st.markdown(
    "### AI/ML Research Prototype Dashboard"
)

st.info(
    "Educational research prototype only. "
    "This dashboard is not intended for clinical decisions "
    "or vaccine dosage recommendations."
)

st.divider()

# Model performance
st.header("📊 1. Model Performance")

metrics_file = REPORTS_DIR / "model_comparison.csv"

if metrics_file.exists():
    metrics_df = pd.read_csv(metrics_file)

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    if not metrics_df.empty:
        best_model = metrics_df.loc[
            metrics_df["F1 Score"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Model",
            best_model["Model"]
        )

        col2.metric(
            "Accuracy",
            f'{best_model["Accuracy"]:.2%}'
        )

        col3.metric(
            "F1 Score",
            f'{best_model["F1 Score"]:.2%}'
        )
else:
    st.warning("Model comparison file not found.")

st.divider()

# Three-panel dashboard
st.header("📈 2. Research Visualizations")

col1, col2, col3 = st.columns(3)

# Panel 1: ROC Curve
with col1:
    st.subheader("ROC Curve")

    roc_image = FIGURES_DIR / "roc_curve.png"

    if roc_image.exists():
        st.image(
            str(roc_image),
            use_container_width=True
        )
    else:
        st.warning("ROC curve image not found.")

    auc_file = REPORTS_DIR / "roc_auc_scores.csv"

    if auc_file.exists():
        auc_df = pd.read_csv(auc_file)
        st.dataframe(
            auc_df,
            use_container_width=True,
            hide_index=True
        )

# Panel 2: Attention Heatmap
with col2:
    st.subheader("Self-Attention Heatmap")

    attention_image = FIGURES_DIR / "attention_heatmap.png"

    if attention_image.exists():
        st.image(
            str(attention_image),
            use_container_width=True
        )
        st.caption("Illustrative nucleotide attention weights.")
    else:
        st.warning("Attention heatmap image not found.")

# Panel 3: Latency Comparison
with col3:
    st.subheader("RAG vs LLM Latency")

    latency_image = FIGURES_DIR / "latency_comparison.png"

    if latency_image.exists():
        st.image(
            str(latency_image),
            use_container_width=True
        )
    else:
        st.warning("Latency comparison image not found.")

    latency_file = REPORTS_DIR / "latency_comparison.csv"

    if latency_file.exists():
        latency_df = pd.read_csv(latency_file)

        st.dataframe(
            latency_df,
            use_container_width=True,
            hide_index=True
        )

st.divider()

# Project summary
st.header("📁 3. Project Information")

st.markdown("""
- **Project:** Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform
- **Machine Learning:** Logistic Regression and Random Forest
- **NLP:** Retrieval-Augmented Generation (RAG)
- **Vector Database:** ChromaDB
- **LLM:** Llama 3.2
- **Validation:** Pydantic
- **Dashboard:** Streamlit
""")

st.success("Dashboard application loaded.")

st.caption(
    "Academic project | AI & ML Research Prototype | 2026"
)
