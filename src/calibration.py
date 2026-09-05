"""
Calibration Module.

Applies Platt Scaling (Sigmoid) to convert a base classifier's raw decision 
functions into reliable, statistically sound probabilities. This is critical 
for the downstream Uncertainty Quantification and Conformal Prediction tasks.
"""

import pickle
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.metrics import brier_score_loss
from src.modelling import build_text_pipeline, STRUCTURED_BOOL_COLS, STRUCTURED_NUM_COLS


def build_calibrated_pipeline(base_clf, text_columns, cv=5):
    """
    Wraps the baseline classifier in a CalibratedClassifierCV.
    
    Design Decision: 
    The 'sigmoid' method (Platt Scaling) is explicitly chosen over 'isotonic' 
    regression because isotonic regression tends to overfit on very small 
    datasets (N=44).
    """
    calibrated_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=cv)
    
    pipeline = build_text_pipeline(
        text_columns, 
        STRUCTURED_BOOL_COLS, 
        STRUCTURED_NUM_COLS, 
        calibrated_clf
    )
    return pipeline


def evaluate_calibration_performance(X, y, raw_pipeline, calibrated_pipeline, skf):
    """
    Evaluates the calibration visually (Calibration Curves) and quantitatively (Brier Score).
    This function demonstrates the reduction in probabilistic error post-calibration.

    Returns:
        tuple: (brier_improvements, class_names) where brier_improvements is a
        list of per-class percentage improvements (raw Brier -> calibrated
        Brier), in the same order as class_names. Returned so callers (e.g.
        save_calibrated_model) can persist these results without
        recomputing them.
    """
    class_names = sorted(y.unique())
    y_bin = label_binarize(y, classes=class_names)
    n_classes = len(class_names)

    # 1. Extract raw scores (handles both predict_proba and decision_function)
    base_estimator = raw_pipeline.named_steps.get('clf', raw_pipeline)
    if hasattr(base_estimator, "predict_proba"):
        raw_scores = cross_val_predict(raw_pipeline, X, y, cv=skf, method="predict_proba")
        needs_scaling = False
    else:
        raw_scores = cross_val_predict(raw_pipeline, X, y, cv=skf, method="decision_function")
        needs_scaling = True

    # 2. Extract calibrated probabilities
    calibrated_proba = cross_val_predict(calibrated_pipeline, X, y, cv=skf, method="predict_proba")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    print("--- Brier Score Comparison (Lower is better) ---")
    improvements = []

    for i, cname in enumerate(class_names):
        if needs_scaling:
            raw_class_scores = MinMaxScaler().fit_transform(raw_scores[:, [i]]).ravel()
            
            raw_class_scores = np.clip(raw_class_scores, 0.0, 1.0)
        else:
            raw_class_scores = raw_scores[:, i]

        # Calculate Brier Score
        brier_raw = brier_score_loss(y_bin[:, i], raw_class_scores)
        brier_calib = brier_score_loss(y_bin[:, i], calibrated_proba[:, i])
        improvement = ((brier_raw - brier_calib) / brier_raw * 100) if brier_raw > 0 else 0
        improvements.append(improvement)
        print(f"{cname: <25} | Raw Brier: {brier_raw:.3f} -> Calibrated Brier: {brier_calib:.3f} ({improvement:+.1f}%)")

        CalibrationDisplay.from_predictions(
            y_bin[:, i], raw_class_scores, n_bins=5, name="Before (raw)", ax=axes[i]
        )
        CalibrationDisplay.from_predictions(
            y_bin[:, i], calibrated_proba[:, i], n_bins=5, name="After (Platt)", ax=axes[i]
        )
        axes[i].set_title(f"{cname}", fontsize=11)
        axes[i].legend(loc="lower right", fontsize=8)

    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Calibration Curves: Raw vs Calibrated Probabilities", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    print(f"\nAverage improvement: {np.mean(improvements):.1f}%")
    return improvements, class_names


def save_calibrated_model(calibrated_pipeline, models_dir, metadata, brier_improvements, class_names):
    """
    Persist the calibrated pipeline and update metadata with calibration
    results (Brier improvement, per-class breakdown, calibration settings).

    Design Decision:
    Metadata is stored as JSON alongside the pickled pipeline so that any
    downstream notebook or script (e.g. main.py, 04_uncertainty.ipynb) can
    inspect model provenance without unpickling the model itself.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "best_model_calibrated.pkl", "wb") as f:
        pickle.dump(calibrated_pipeline, f)

    metadata["calibration_method"] = "Platt Scaling (sigmoid)"
    metadata["calibration_cv_folds"] = 3
    metadata["calibration_notes"] = "cv=3 used instead of cv=5 due to small class sizes (N=44)"
    metadata["average_brier_improvement"] = float(np.mean(brier_improvements))
    metadata["brier_improvement_by_class"] = {
        class_names[i]: float(brier_improvements[i]) for i in range(len(class_names))
    }

    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata