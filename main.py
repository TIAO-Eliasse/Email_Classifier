"""
RedRock Email Classification -- main.py
=========================================

End-to-end orchestrator for the email routing pipeline. Wraps the modules
in src/ (ingestion, preprocessing, features, modelling, calibration,
uncertainty, explainability) into two CLI subcommands.

    python main.py train    ingest train/ + train_labels.csv -> fit the
                             classifier -> calibrate it -> pick an
                             operating (auto-forward vs defer-to-human)
                             threshold -> persist everything to
                             outputs/models/

    python main.py predict  load the model persisted by `train` (never
                             retrains) -> ingest test/ -> classify each
                             email -> apply the routing threshold ->
                             generate a SHAP-based audit trail -> write
                             outputs/results.csv (spec-compliant) and
                             outputs/results_detailed.csv (+ routing
                             decision, top evidence words)

Design decisions (see README for the full write-up):
  - train/predict are deliberately separate commands, not one script that
    always retrains. `predict` must be fast, deterministic, and reusable
    as-is from the future Streamlit app.
  - The routing threshold is a property of the TRAINED MODEL, computed
    once in `train` and stored in metadata.json -- `predict` never
    recomputes it from whatever batch of emails it happens to be given.
  - Human-in-the-loop by design: emails the model isn't confident about
    are routed to DEFER_TO_HUMAN rather than force-classified, mirroring
    the AI+human-review pattern Isazi already ships in production
    (e.g. their Sophia document-transcription product).
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a window from a CLI run
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC

from src.ingestion import build_dataframe, load_labels
from src.preprocessing import add_clean_text_column
from src.features import add_engineered_features
from src.modelling import (
    STRUCTURED_BOOL_COLS,
    STRUCTURED_NUM_COLS,
    train_and_save_best_model,
)
from src.calibration import (
    build_calibrated_pipeline,
    evaluate_calibration_performance,
    save_calibrated_model,
)
from src.uncertainty import select_operating_threshold, save_routing_results
from src.explainability import generate_compliance_audit_trail

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

FEATURE_COLS = ["subject", "message"] + STRUCTURED_BOOL_COLS + STRUCTURED_NUM_COLS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ingest_and_featurize(directory: str) -> pd.DataFrame:
    """Ingestion + preprocessing + feature engineering, shared by both commands."""
    df = build_dataframe(directory)
    if df.empty:
        raise ValueError(f"No emails found in '{directory}'. Check the path.")
    df = add_clean_text_column(df)
    df = add_engineered_features(df)
    return df


def _load_model(models_dir: Path):
    """Load the calibrated pipeline + metadata persisted by `train`."""
    model_path = models_dir / "best_model_calibrated.pkl"
    metadata_path = models_dir / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        log.error(
            "No trained model found in '%s'. Run `python main.py train` first.",
            models_dir,
        )
        sys.exit(1)

    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        log.error(
            "Failed to load the model from '%s' (%s). This usually means the "
            "pickled model was created with a different scikit-learn version "
            "than the one currently installed. Re-run `python main.py train` "
            "to regenerate it in this environment.",
            models_dir, e,
        )
        sys.exit(1)

    if "routing_threshold" not in metadata:
        log.error(
            "metadata.json has no routing_threshold -- it looks like training "
            "did not complete. Re-run `python main.py train`."
        )
        sys.exit(1)

    return pipeline, metadata


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("Ingesting training emails from '%s'...", args.train_dir)
    train_df = _ingest_and_featurize(args.train_dir)

    labels_df = load_labels(args.labels_csv)
    train_labeled = train_df.merge(
        labels_df, left_on="source_filename", right_on="filename", how="left"
    )
    n_missing = train_labeled["true_category"].isna().sum()
    if n_missing:
        raise ValueError(
            f"{n_missing} training email(s) have no matching label in "
            f"'{args.labels_csv}' -- check that 'filename' values match the "
            "email .html filenames exactly."
        )

    y = train_labeled["true_category"]
    log.info("Loaded %d labeled emails across %d categories.", len(y), y.nunique())

    # --- Fit + persist the best (uncalibrated) model, with one final CV score ---
    log.info("Training LinearSVC on separated subject/message TF-IDF + structured features...")
    raw_pipeline, metadata = train_and_save_best_model(train_labeled, y, models_dir)
    log.info(
        "Raw model F1-macro (5-fold CV): %.3f (+/- %.3f)",
        metadata["f1_macro_cv_mean"], metadata["f1_macro_cv_std"],
    )

    # --- Calibrate (Platt/sigmoid) so predict_proba is a trustworthy confidence score ---
    X = train_labeled[FEATURE_COLS]
    log.info("Calibrating with Platt scaling (sigmoid, cv=3)...")
    calibrated_pipeline = build_calibrated_pipeline(
        LinearSVC(class_weight="balanced", random_state=42),
        ("subject", "message"),
        cv=3,
    )
    calibrated_pipeline.fit(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    brier_improvements, class_names = evaluate_calibration_performance(
        X, y, raw_pipeline, calibrated_pipeline, skf
    )
    # evaluate_calibration_performance() plots calibration curves as a side
    # effect (headless-safe here via the Agg backend set above); persist
    # the figure instead of losing it to plt.show() doing nothing.
    plt.savefig(models_dir / "calibration_curves.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    metadata = save_calibrated_model(
        calibrated_pipeline, models_dir, metadata, brier_improvements, class_names
    )
    log.info("Average Brier score improvement from calibration: %.1f%%",
              metadata["average_brier_improvement"])

    # --- Pick the auto-forward / defer-to-human operating threshold ---
    log.info("Selecting the auto-forward confidence threshold (zero observed errors)...")
    calib_proba = cross_val_predict(calibrated_pipeline, X, y, cv=skf, method="predict_proba")
    classes = calibrated_pipeline.classes_
    threshold, threshold_metrics, routing_df = select_operating_threshold(
        calib_proba, y.values, classes
    )
    auto, defer = save_routing_results(routing_df, threshold, models_dir, metadata)

    pct_auto = auto / (auto + defer) * 100
    log.info(
        "Routing threshold = %.3f -> on training data: %d/%d emails (%.1f%%) would be "
        "auto-forwarded with zero observed misrouting; %d deferred to human "
        "compliance review.",
        threshold, auto, auto + defer, pct_auto, defer,
    )
    log.info("Model, metadata and routing threshold saved to '%s'.", models_dir)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def cmd_predict(args):
    models_dir = Path(args.models_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline, metadata = _load_model(models_dir)
    threshold = metadata["routing_threshold"]
    log.info(
        "Loaded %s (%s text, calibrated via %s). Routing threshold = %.3f.",
        metadata["model_type"], metadata["text_representation"],
        metadata["calibration_method"], threshold,
    )

    log.info("Ingesting emails to classify from '%s'...", args.test_dir)
    test_df = _ingest_and_featurize(args.test_dir)
    X_test = test_df[FEATURE_COLS].reset_index(drop=True)
    log.info("Classifying %d emails...", len(X_test))

    proba = pipeline.predict_proba(X_test)
    classes = pipeline.classes_
    pred_idx = proba.argmax(axis=1)
    predicted_category = classes[pred_idx]
    confidence_score = proba.max(axis=1)
    decision = ["AUTO_FORWARD" if c >= threshold else "DEFER_TO_HUMAN" for c in confidence_score]

    # --- Spec-compliant output: exactly the 3 columns the PDF asks for ---
    results = pd.DataFrame({
        "email_id": test_df["email_id"].values,
        "predicted_category": predicted_category,
        "confidence_score": confidence_score.round(4),
    })
    results.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows).", out_path, len(results))

    # --- Extended output: routing decision + SHAP audit trail for compliance review ---
    try:
        audit = generate_compliance_audit_trail(pipeline, X_test, top_n=5)
        top_driving_features = audit["top_driving_features"].values
    except Exception as e:
        log.warning("Could not generate the SHAP audit trail (%s); "
                    "results_detailed.csv will omit top_driving_features.", e)
        top_driving_features = [""] * len(results)

    detailed = results.copy()
    detailed["decision"] = decision
    detailed["top_driving_features"] = top_driving_features
    detailed_path = out_path.parent / "results_detailed.csv"
    detailed.to_csv(detailed_path, index=False)

    n_auto = sum(d == "AUTO_FORWARD" for d in decision)
    n_defer = len(decision) - n_auto
    log.info(
        "Routing summary: %d/%d (%.1f%%) auto-forwarded, %d deferred to human "
        "compliance review.",
        n_auto, len(decision), n_auto / len(decision) * 100, n_defer,
    )
    log.info("Wrote %s (with routing decision + audit trail).", detailed_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="RedRock email classification -- train or predict."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fit, calibrate, and persist the model.")
    p_train.add_argument("--train_dir", default="train")
    p_train.add_argument("--labels_csv", default="train_labels.csv")
    p_train.add_argument("--models_dir", default="outputs/models")
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="Classify new emails with the trained model.")
    p_predict.add_argument("--test_dir", default="test")
    p_predict.add_argument("--models_dir", default="outputs/models")
    p_predict.add_argument("--out", default="outputs/results.csv")
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()