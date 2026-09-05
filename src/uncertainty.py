"""
Uncertainty Quantification & Routing Module.

Implements the "Deferral to Human" logic utilizing Conformal Prediction (MAPIE) 
and heuristic uncertainty measures (Entropy/Margin). 

Design Trade-off Note:
Strict Conformal Prediction requires a held-out calibration set. With a dataset 
of N=44, splitting the data further reduces the training signal. This implementation 
is provided as a robust, production-ready architectural pattern for regulatory 
routing, acknowledging that marginal coverage guarantees are statistically loose 
at this extreme low-sample regime.

Note on MAPIE API: this uses the v1.x API (mapie>=1.0), where the former
MapieClassifier(cv="prefit") workflow was replaced by SplitConformalClassifier
with an explicit fit -> conformalize -> predict_set sequence.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from mapie.classification import SplitConformalClassifier
from sklearn.model_selection import train_test_split


def calculate_heuristic_uncertainty(proba_matrix):
    """
    Calculates heuristic uncertainty metrics for a given probability matrix.
    Useful as a secondary fallback metric for extremely small datasets.
    """
    entropies = entropy(proba_matrix, axis=1, base=2) 
    
    sorted_proba = np.sort(proba_matrix, axis=1)
    margins = sorted_proba[:, -1] - sorted_proba[:, -2]
    
    return entropies, margins


def build_conformal_router(base_pipeline, X, y, test_size=0.25, confidence_level=0.9, random_state=42):
    """
    Splits the data, fits the base classifier, then conformalizes it on
    the held-out calibration split.

    Design Trade-off Note:
    The calibration split here (~25% of 44 examples, i.e. ~11 emails) is
    small enough that MAPIE's marginal coverage guarantee should be read
    as demonstrative rather than statistically robust at this sample size.
    """
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    mapie_clf = SplitConformalClassifier(
        estimator=base_pipeline, confidence_level=confidence_level,
        prefit=False, random_state=random_state
    )
    mapie_clf.fit(X_train, y_train)
    mapie_clf.conformalize(X_calib, y_calib)

    return mapie_clf

def execute_routing_decision(mapie_clf, X_test):
    """
    Agentic Workflow Router leveraging Conformal Prediction sets.
    
    Routing Logic:
    - Set size == 1: Absolute certainty -> AUTO_FORWARD
    - Set size != 1: Epistemic uncertainty or ambiguity -> DEFER_TO_HUMAN

    Note: mapie>=1.0's SplitConformalClassifier does not publicly expose
    `classes_` (unlike the old MapieClassifier). It is accessed here via
    the internal `_mapie_classifier` attribute -- functional in mapie
    1.5.0, but relies on a private API that could change again in a
    future release without notice. Flagged explicitly as a
    maintainability risk tied to pinning mapie's version in
    requirements.txt.
    """
    classes = mapie_clf._mapie_classifier.classes_
    _, y_pred_set = mapie_clf.predict_set(X_test)

    sets_squeezed = y_pred_set[:, :, 0]

    routing_actions = []
    for p_set in sets_squeezed:
        set_size = p_set.sum()

        if set_size == 1:
            predicted_class = classes[np.argmax(p_set)]
            routing_actions.append(f"AUTO_FORWARD: {predicted_class}")
        else:
            routing_actions.append("DEFER_TO_HUMAN: Compliance Review")

    return routing_actions

def select_operating_threshold(proba_matrix, y_true, classes, thresholds=None):
    """
    Grid-search a confidence threshold that maximises auto-forward
    coverage subject to zero observed errors among auto-forwarded emails.

    Design Trade-off Note:
    The threshold is selected and evaluated on the same cross-validated
    predictions passed in. This is optimistic by construction -- it
    should be re-validated on a genuinely held-out set before production
    use.
    """
    if thresholds is None:
        thresholds = np.linspace(0.35, 0.85, 21)

    y_pred = classes[np.argmax(proba_matrix, axis=1)]
    max_proba = np.max(proba_matrix, axis=1)
    is_correct = (y_pred == y_true).astype(int)

    rows = []
    for t in thresholds:
        routed = (max_proba >= t)
        n_auto = routed.sum()
        rows.append({
            "threshold": t,
            "n_auto": n_auto,
            "pct_auto": n_auto / len(routed) * 100,
            "accuracy_auto": is_correct[routed].mean() if n_auto > 0 else 0,
            "cost_auto_errors": ((routed) & (is_correct == 0)).sum(),
        })

    metrics_df = pd.DataFrame(rows)
    zero_error = metrics_df[metrics_df["cost_auto_errors"] == 0]
    if len(zero_error) == 0:
        raise ValueError("No threshold achieves zero auto-forward errors on this sample.")
    best_idx = zero_error["n_auto"].idxmax()
    selected_threshold = metrics_df.loc[best_idx, "threshold"]

    routing_df = pd.DataFrame({
        "predicted_class": y_pred,
        "confidence": max_proba,
        "decision": np.where(max_proba >= selected_threshold, "AUTO_FORWARD", "DEFER_TO_HUMAN"),
    })
    return selected_threshold, metrics_df, routing_df


def save_routing_results(routing_df, selected_threshold, models_dir, metadata):
    """Persist routing decisions to CSV and update metadata with the chosen threshold."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    routing_df.to_csv(models_dir / "routing_decisions_final.csv", index=False)

    auto = (routing_df["decision"] == "AUTO_FORWARD").sum()
    defer = (routing_df["decision"] == "DEFER_TO_HUMAN").sum()
    metadata["routing_threshold"] = float(selected_threshold)
    metadata["routing_auto"] = int(auto)
    metadata["routing_defer"] = int(defer)

    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return auto, defer