"""
Streamlit interface for RedRock Email Classification.

Simple, business-friendly interface. No technical jargon.
"""

import logging
import shutil
import sys
import tempfile
import zipfile
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
import main as pipeline
from src.explainability import (
    initialize_shap_explainer,
    plot_shap_bar_by_group,
)

log = logging.getLogger("app")

st.set_page_config(
    page_title="RedRock Email Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

css = """
<style>
:root {
    --color-primary: #1a9c8f;
    --color-primary-light: #20b2a3;
    --color-dark: #2d2d3d;
    --color-gray: #5a5a6f;
    --color-light: #f5f5f7;
}

.main-header {
    background: linear-gradient(90deg, #2d2d3d 0%, #1a9c8f 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(26, 156, 143, 0.2);
}

.success-box {
    background: linear-gradient(135deg, #e8f5f3 0%, #d4ede8 100%);
    border-left: 4px solid #1a9c8f;
    padding: 1rem;
    border-radius: 6px;
    color: #1a5a52;
}

.stButton>button {
    background: linear-gradient(90deg, #1a9c8f 0%, #20b2a3 100%);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #157a72 0%, #1a8b7d 100%);
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>RedRock Email Classifier</h1>
    <p style="font-size: 1.1rem; margin: 0;">Predict, then a person double-checks the unclear cases.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Upload client emails. Each one is sorted into a category and either "
    "routed automatically or sent to someone for a quick check."
)

DEFAULT_MODELS_DIR = Path("outputs/models")

CSV_DTYPES = {"email_id": str}


def _extract_zip(uploaded_zip) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(uploaded_zip) as zf:
        zf.extractall(tmp_dir)
    return tmp_dir


st.header("Classify Emails")
st.write("Upload a `.zip` containing `.html` email files.")

if not (DEFAULT_MODELS_DIR / "best_model_calibrated.pkl").exists():
    st.error("No model is currently set up. Please contact your administrator.")
else:
    uploaded_zip = st.file_uploader("Upload .zip of .html email files", type=["zip"])

    if uploaded_zip and st.button("Classify", use_container_width=True):
        try:
            with st.spinner("Processing..."):
                tmp_root = _extract_zip(uploaded_zip)
                html_files = list(tmp_root.rglob("*.html"))

                if not html_files:
                    st.warning("No .html files found in the uploaded zip.")
                    st.stop()

                test_dir = tmp_root / "_emails"
                test_dir.mkdir(exist_ok=True)
                for f in html_files:
                    shutil.copy(f, test_dir / f.name)

                out_path = tmp_root / "results.csv"
                args = SimpleNamespace(
                    test_dir=str(test_dir),
                    models_dir=str(DEFAULT_MODELS_DIR),
                    out=str(out_path),
                )
                pipeline.cmd_predict(args)

                results = pd.read_csv(out_path, dtype=CSV_DTYPES)
                detailed_path = out_path.parent / "results_detailed.csv"
                detailed = (
                    pd.read_csv(detailed_path, dtype=CSV_DTYPES)
                    if detailed_path.exists() else results
                )

            st.markdown("""
            <div class="success-box"><strong>Done!</strong> All emails have been processed.</div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Emails", len(results))
            if "decision" in detailed.columns:
                n_auto = (detailed["decision"] == "AUTO_FORWARD").sum()
                n_defer = (detailed["decision"] == "DEFER_TO_HUMAN").sum()
                with col2:
                    st.metric("Routed Automatically", f"{n_auto}")
                with col3:
                    st.metric("Sent for Review", f"{n_defer}")

            st.subheader("Results")
            display_cols = [c for c in ["email_id", "predicted_category", "confidence_score", "decision"]
                             if c in detailed.columns]
            st.dataframe(detailed[display_cols], use_container_width=True, height=400)

            # --- What influenced these decisions: one SHAP bar chart per group ---
            if "decision" in detailed.columns and detailed["decision"].nunique() > 0:
                st.subheader("What Influenced These Decisions")

                with st.spinner("Analyzing..."):
                    with open(DEFAULT_MODELS_DIR / "best_model_calibrated.pkl", "rb") as f:
                        model = pickle.load(f)

                    from src.ingestion import build_dataframe
                    from src.preprocessing import add_clean_text_column
                    from src.features import add_engineered_features
                    from src.modelling import STRUCTURED_BOOL_COLS, STRUCTURED_NUM_COLS

                    X_emails = build_dataframe(test_dir)  # email_id already str here
                    X_emails = add_clean_text_column(X_emails)
                    X_emails = add_engineered_features(X_emails)
                    X_input = X_emails[["subject", "message"] + STRUCTURED_BOOL_COLS + STRUCTURED_NUM_COLS]

                    explainer, shap_values, feature_names = initialize_shap_explainer(model, X_input)
                    class_names = list(model.classes_)

                    # Both sides are str email_id now (CSV_DTYPES on read,
                    # build_dataframe's own .astype(str) on the ingestion side)
                    decision_by_id = detailed.set_index("email_id")["decision"]
                    decisions_aligned = X_emails["email_id"].map(decision_by_id)

                    auto_mask = (decisions_aligned == "AUTO_FORWARD").values
                    defer_mask = (decisions_aligned == "DEFER_TO_HUMAN").values

                    fig_auto = plot_shap_bar_by_group(
                        shap_values, feature_names, class_names,
                        mask=auto_mask, top_n=15, title="Routed automatically"
                    )
                    fig_defer = plot_shap_bar_by_group(
                        shap_values, feature_names, class_names,
                        mask=defer_mask, top_n=15, title="Sent for review"
                    )

                st.markdown("**Routed automatically**")
                if fig_auto is not None:
                    st.pyplot(fig_auto)
                    plt.close(fig_auto)
                else:
                    st.info("No emails were routed automatically in this batch.")

                st.markdown("**Sent for review**")
                if fig_defer is not None:
                    st.pyplot(fig_defer)
                    plt.close(fig_defer)
                else:
                    st.info("No emails were sent for review in this batch.")

                st.caption(
                    "What the system noticed in emails it routed automatically, "
                    "compared to the ones sent for a quick human check."
                )

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="results.csv",
                    mime="text/csv",
                )
            with col2:
                if detailed_path.exists():
                    st.download_button(
                        "Download detailed report",
                        data=detailed.to_csv(index=False).encode("utf-8"),
                        file_name="results_detailed.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            log.exception("Failed to process uploaded emails")
            st.error("Something went wrong while processing this file. Please try again or contact your administrator.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### How It Works

1. Upload client emails
2. Each one is sorted into a category
3. Clear cases are routed automatically
4. Uncertain cases are sent to a person to check
""")