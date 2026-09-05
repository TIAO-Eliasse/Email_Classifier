"""
Explainability & Audit Trail Module.

Provides Local Feature Attribution using SHAP for email classification.
For regulatory compliance, generates human-readable rationales explaining 
why an email was classified into a specific department by extracting the 
exact text tokens driving the prediction.

Optimized for Linear models over TF-IDF vectors for high computational efficiency.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional


def _get_base_estimator(fitted_pipeline):
    """
    Extract the base estimator from a potentially wrapped classifier.
    
    Handles CalibratedClassifierCV and other wrappers by extracting
    the underlying fitted linear models and averaging their coefficients.
    """
    clf_step = fitted_pipeline.named_steps["clf"]
    
    if hasattr(clf_step, 'calibrated_classifiers_'):
        fold_estimators = [cc.estimator for cc in clf_step.calibrated_classifiers_]
        avg_coef = np.mean([est.coef_ for est in fold_estimators], axis=0)
        avg_intercept = np.mean([est.intercept_ for est in fold_estimators], axis=0)
        return (avg_coef, avg_intercept)
    
    elif hasattr(clf_step, 'coef_'):
        return (clf_step.coef_, clf_step.intercept_)
    
    else:
        raise ValueError(
            "Pipeline clf step must be a linear estimator or CalibratedClassifierCV"
        )


def _convert_to_dense(X_transformed):
    """Convert sparse matrix to dense if needed."""
    if hasattr(X_transformed, 'toarray'):
        return X_transformed.toarray()
    return X_transformed


def initialize_shap_explainer(
    fitted_pipeline,
    X_df: pd.DataFrame
) -> Tuple[shap.LinearExplainer, shap.Explanation, np.ndarray]:
    """
    Initialize SHAP LinearExplainer for model interpretation.
    
    Computes SHAP values once to be reused across audit trail generation
    and visualization functions, avoiding redundant computation.
    
    Returns:
        explainer: SHAP LinearExplainer object
        shap_values: SHAP Explanation object for all samples
        feature_names: Array of feature names
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    base_estimator = _get_base_estimator(fitted_pipeline)
    
    X_transformed = preprocessor.transform(X_df)
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.LinearExplainer(
        base_estimator,
        X_transformed,
        feature_names=feature_names
    )
    
    shap_values = explainer(X_transformed)
    
    return explainer, shap_values, feature_names


def generate_compliance_audit_trail(
    fitted_pipeline,
    X_df: pd.DataFrame,
    top_n: int = 5,
    email_id_col: Optional[str] = None,
    shap_values=None,
    feature_names=None
) -> pd.DataFrame:
    """
    Generate SHAP-based explanation report for a batch of emails.
    
    Each feature is labelled with direction: positive SHAP means 
    feature pushed toward predicted class, negative means against 
    (but outweighed by other evidence).
    
    Supports optional pre-computed shap_values and feature_names for
    optimization when processing multiple batches.
    """
    if shap_values is None:
        explainer, shap_values, feature_names = initialize_shap_explainer(
            fitted_pipeline, X_df
        )
    
    audit_records = []
    
    for idx in range(len(X_df)):
        if email_id_col and email_id_col in X_df.columns:
            email_id = X_df.iloc[idx][email_id_col]
        else:
            email_id = idx
        
        pred_class_label = fitted_pipeline.predict(X_df.iloc[[idx]])[0]
        pred_class_idx = list(fitted_pipeline.classes_).index(pred_class_label)
        
        shap_vals_for_class = shap_values.values[idx, :, pred_class_idx]
        top_contributors_idx = np.argsort(-np.abs(shap_vals_for_class))[:top_n]
        
        explanations = []
        for feat_idx in top_contributors_idx:
            feat_name = feature_names[feat_idx]
            feat_name = feat_name.replace("body_tfidf__", "").replace("subject_tfidf__", "")
            
            impact_score = shap_vals_for_class[feat_idx]
            direction = "supports" if impact_score > 0 else "against"
            explanations.append(f"{feat_name} ({impact_score:+.2f}, {direction})")
        
        audit_records.append({
            "email_id": email_id,
            "predicted_category": pred_class_label,
            "top_driving_features": " | ".join(explanations)
        })
    
    return pd.DataFrame(audit_records)


def compute_global_feature_importance(
    shap_values,
    feature_names: np.ndarray
) -> pd.DataFrame:
    """Compute global feature importance across all predictions."""
    feature_importance = np.mean(np.abs(shap_values.values), axis=(0, 2))
    
    feature_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)
    
    feature_imp_df["feature"] = (
        feature_imp_df["feature"]
        .str.replace("subject_tfidf__", "")
        .str.replace("body_tfidf__", "")
    )
    
    return feature_imp_df


def compute_class_feature_importance(
    shap_values,
    feature_names: np.ndarray,
    class_names: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """Compute per-class feature importance."""
    class_importance_dict = {}
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_importance = np.mean(np.abs(shap_values.values[:, :, cls_idx]), axis=0)
        
        cls_df = pd.DataFrame({
            "feature": feature_names,
            "importance": cls_importance
        }).sort_values("importance", ascending=False)
        
        cls_df["feature"] = (
            cls_df["feature"]
            .str.replace("subject_tfidf__", "")
            .str.replace("body_tfidf__", "")
        )
        
        class_importance_dict[cls_name] = cls_df
    
    return class_importance_dict


def plot_group_comparison_figure(
    shap_values,
    feature_names: np.ndarray,
    auto_mask: np.ndarray,
    defer_mask: np.ndarray,
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Side-by-side comparison: AUTO_FORWARD vs DEFER_TO_HUMAN.
    
    Both panels show the SAME set of features (selected from combined
    importance) with SHARED x-axis scale, enabling direct feature-by-feature
    comparison. This avoids the misleading case where each panel 
    independently picks different top features.
    """
    auto_mask = np.asarray(auto_mask)
    defer_mask = np.asarray(defer_mask)
    n_features = len(feature_names)
    top_n = min(top_n, n_features)
    
    def _clean(name):
        return name.replace("subject_tfidf__", "").replace("body_tfidf__", "")
    
    def _group_importance(mask):
        if mask.sum() == 0:
            return np.zeros(n_features)
        return np.mean(np.abs(shap_values.values[mask, :, :]), axis=(0, 2))
    
    auto_importance = _group_importance(auto_mask)
    defer_importance = _group_importance(defer_mask)
    
    combined_importance = auto_importance + defer_importance
    top_idx = np.argsort(-combined_importance)[:top_n]
    clean_names = [_clean(feature_names[i]) for i in top_idx]
    
    max_value = max(
        auto_importance[top_idx].max() if auto_mask.sum() > 0 else 0,
        defer_importance[top_idx].max() if defer_mask.sum() > 0 else 0
    )
    max_value = max_value * 1.1 if max_value > 0 else 1.0
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    
    for ax, mask, importance, title, color in [
        (axes[0], auto_mask, auto_importance, "AUTO_FORWARD", "steelblue"),
        (axes[1], defer_mask, defer_importance, "DEFER_TO_HUMAN", "darkorange"),
    ]:
        n_emails = int(mask.sum())
        if n_emails == 0:
            ax.text(0.5, 0.5, f"No emails in this group", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{title} (0 emails)", fontweight="bold")
            ax.set_xlim(0, max_value)
            continue
        
        values = importance[top_idx]
        ax.barh(clean_names, values, color=color, edgecolor="black", alpha=0.8)
        
        for y_pos, v in enumerate(values):
            ax.text(v, y_pos, f" {v:.3f}", va="center", fontsize=8)
        
        ax.set_title(f"{title} ({n_emails} emails)", fontweight="bold", fontsize=11)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_xlim(0, max_value)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle("Top contributing features: same features, shared scale", 
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_global_feature_importance(
    feature_imp_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """Plot global feature importance as horizontal bar chart."""
    top_features = feature_imp_df.head(top_n).copy()
    
    plt.figure(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    plt.barh(
        y_pos,
        top_features["importance"].values,
        color='steelblue',
        edgecolor='black',
        height=0.7
    )
    
    plt.yticks(y_pos, top_features["feature"].values, fontsize=10)
    plt.xlabel("Mean |SHAP|", fontsize=12, fontweight='bold')
    plt.title(f"Top {top_n} Global Features by Importance", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance'], i, f"  {row['importance']:.4f}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_class_feature_importance(
    class_importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 10,
    figsize: Tuple[int, int] = (18, 6)
) -> None:
    """Plot per-class feature importance as subplots."""
    class_names = list(class_importance_dict.keys())
    fig, axes = plt.subplots(1, len(class_names), figsize=figsize)
    
    if len(class_names) == 1:
        axes = [axes]
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_df = class_importance_dict[cls_name].head(top_n).copy()
        
        y_pos = np.arange(len(cls_df))
        axes[cls_idx].barh(
            y_pos,
            cls_df["importance"].values,
            color='steelblue',
            edgecolor='black',
            height=0.7
        )
        axes[cls_idx].set_yticks(y_pos)
        axes[cls_idx].set_yticklabels(cls_df["feature"].values, fontsize=9)
        axes[cls_idx].set_title(f"{cls_name}", fontweight='bold', fontsize=11)
        axes[cls_idx].invert_yaxis()
        axes[cls_idx].grid(axis='x', alpha=0.3, linestyle='--')
        axes[cls_idx].set_xlabel("Importance", fontsize=9)
    
    plt.suptitle(f"Top {top_n} Features by Class", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_shap_summary(
    shap_values,
    X_transformed,
    feature_names: np.ndarray,
    plot_type: str = "bar",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Generate SHAP summary plot."""
    if hasattr(X_transformed, 'toarray'):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed
    
    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_dense,
        feature_names=feature_names,
        plot_type=plot_type,
        show=False
    )
    plt.title(
        f"SHAP Summary Plot - {plot_type.capitalize()}",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def plot_waterfall(
    fitted_pipeline,
    X_df: pd.DataFrame,
    explainer: shap.LinearExplainer,
    shap_values,
    feature_names: np.ndarray,
    sample_idx: int = 0,
    max_display: int = 15,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Generate SHAP waterfall plot for individual prediction explanation.
    
    Shows step-by-step how each feature contributes to the prediction.
    """
    if sample_idx >= len(X_df):
        raise ValueError(f"sample_idx {sample_idx} out of range (max: {len(X_df)-1})")
    
    pred_proba = fitted_pipeline.predict_proba(X_df.iloc[[sample_idx]])[0]
    pred_idx = np.argmax(pred_proba)
    pred_class = fitted_pipeline.classes_[pred_idx]
    confidence = pred_proba[pred_idx]
    
    if isinstance(explainer.expected_value, np.ndarray):
        expected_val = explainer.expected_value[pred_idx]
    else:
        expected_val = explainer.expected_value
    
    shap_vals = shap_values.values[sample_idx, :, pred_idx]
    
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_df.iloc[[sample_idx]])
    X_dense = _convert_to_dense(X_transformed)
    
    shap_explanation = shap.Explanation(
        values=shap_vals,
        base_values=expected_val,
        data=X_dense[0],
        feature_names=feature_names
    )
    
    plt.figure(figsize=figsize)
    shap.waterfall_plot(shap_explanation, max_display=max_display, show=False)
    plt.title(
        f"SHAP Waterfall - Email {sample_idx} "
        f"(Predicted: {pred_class} | Confidence: {confidence:.2%})",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def plot_dependence(
    shap_values,
    X_transformed,
    feature_names: np.ndarray,
    top_n: int = 4,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """Generate SHAP dependence plots for top features."""
    if hasattr(X_transformed, 'toarray'):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed
    
    feature_importance = np.mean(np.abs(shap_values.values), axis=(0, 2))
    top_feature_indices = np.argsort(-feature_importance)[:top_n]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, feat_idx in enumerate(top_feature_indices):
        shap.dependence_plot(
            feat_idx,
            shap_values.values[:, :, 0],
            X_dense,
            feature_names=feature_names,
            ax=axes[i],
            show=False
        )
        axes[i].set_title(
            f"Dependence: {feature_names[feat_idx]}",
            fontweight='bold',
            fontsize=10
        )
    
    plt.suptitle(
        f"SHAP Dependence Plots - Top {top_n} Features",
        fontsize=14,
        fontweight='bold',
        y=1.00
    )
    plt.tight_layout()
    plt.show()


def generate_explainability_summary(
    fitted_pipeline,
    X_df: pd.DataFrame,
    shap_values,
    feature_names: np.ndarray
) -> Dict:
    """Generate comprehensive explainability summary."""
    global_importance = compute_global_feature_importance(shap_values, feature_names)
    
    summary = {
        "total_samples": len(X_df),
        "total_features": len(feature_names),
        "num_classes": len(fitted_pipeline.classes_),
        "classes": list(fitted_pipeline.classes_),
        "top_5_global_features": global_importance.head(5)[
            ["feature", "importance"]
        ].to_dict('records'),
        "explainability_method": "SHAP (Linear Explainer)",
    }
    
    return summary




def _clean_feature_name(name: str) -> str:
    """Strip internal column-transformer prefixes for a business-readable label."""
    return (
        name.replace("subject_tfidf__", "")
        .replace("body_tfidf__", "")
        .replace("bool_features__", "")
        .replace("num_features__", "")
        .replace("_", " ")
    )


def plot_group_comparison_signed(
    fitted_pipeline,
    X_df: pd.DataFrame,
    shap_values,
    feature_names: np.ndarray,
    auto_mask: np.ndarray,
    defer_mask: np.ndarray,
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 6),
    group_labels: Tuple[str, str] = ("Routed automatically", "Sent for review"),
) -> plt.Figure:
    """
    Side-by-side SIGNED comparison: for each email, uses the SHAP values
    for ITS OWN predicted class (not averaged across all 5 classes), so
    the bars answer "what pushed toward / against the category this
    specific email was actually assigned to" -- positive = supports the
    prediction, negative = worked against it but was outweighed.
    """
    auto_mask = np.asarray(auto_mask)
    defer_mask = np.asarray(defer_mask)
    n_samples, n_features = shap_values.values.shape[0], shap_values.values.shape[1]

    pred_idx_per_class = list(fitted_pipeline.classes_)
    pred_labels = fitted_pipeline.predict(X_df)
    pred_class_indices = [pred_idx_per_class.index(p) for p in pred_labels]

    # Each email's SHAP vector for its OWN predicted class only
    per_sample_signed = np.array([
        shap_values.values[i, :, pred_class_indices[i]] for i in range(n_samples)
    ])

    def _group_mean(mask):
        if mask.sum() == 0:
            return np.zeros(n_features)
        return per_sample_signed[mask].mean(axis=0)

    auto_signed = _group_mean(auto_mask)
    defer_signed = _group_mean(defer_mask)

    combined_abs = np.abs(auto_signed) + np.abs(defer_signed)
    top_n = min(top_n, n_features)
    top_idx = np.argsort(-combined_abs)[:top_n]
    clean_names = [_clean_feature_name(feature_names[i]) for i in top_idx]

    max_abs = max(
        np.abs(auto_signed[top_idx]).max() if auto_mask.sum() > 0 else 0,
        np.abs(defer_signed[top_idx]).max() if defer_mask.sum() > 0 else 0,
    )
    max_abs = max_abs * 1.15 if max_abs > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

    for ax, mask, signed_vals, title in [
        (axes[0], auto_mask, auto_signed, group_labels[0]),
        (axes[1], defer_mask, defer_signed, group_labels[1]),
    ]:
        n_emails = int(mask.sum())
        if n_emails == 0:
            ax.text(0.5, 0.5, "No emails in this group", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{title} (0 emails)", fontweight="bold")
            ax.set_xlim(-max_abs, max_abs)
            continue

        values = signed_vals[top_idx]
        colors = ["#1a9c8f" if v > 0 else "#d1495b" for v in values]
        ax.barh(clean_names, values, color=colors, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)

        for y_pos, v in enumerate(values):
            ax.text(v, y_pos, f" {v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)

        ax.set_title(f"{title} ({n_emails} emails)", fontweight="bold", fontsize=11)
        ax.set_xlabel("Average influence")
        ax.set_xlim(-max_abs, max_abs)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle("What pushed toward (green, right) or against (red, left) the outcome",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_shap_bar_by_group(
    shap_values,
    feature_names: np.ndarray,
    class_names: List[str],
    mask: Optional[np.ndarray] = None,
    top_n: int = 15,
    title: str = "SHAP Summary Plot - Bar",
    figsize: Tuple[int, int] = (8, 4),
) -> Optional[plt.Figure]:
    """
    Stacked SHAP bar plot (same format as shap.summary_plot with
    plot_type='bar'), restricted to a subset of samples via `mask`
    (e.g. only AUTO_FORWARD or only DEFER_TO_HUMAN emails), with
    human-readable feature and class names instead of raw column
    names and "Class 0"/"Class 1"/etc.

    Returns None if the mask selects zero samples (nothing to plot).
    """
    values = shap_values.values
    if mask is not None:
        mask = np.asarray(mask)
        if mask.sum() == 0:
            return None
        values = values[mask]

    n_classes = values.shape[2]
    per_class_values = [values[:, :, c] for c in range(n_classes)]
    display_names = [_clean_feature_name(n) for n in feature_names]

    plt.figure(figsize=figsize)
    shap.summary_plot(
        per_class_values,
        feature_names=display_names,
        class_names=list(class_names),
        plot_type="bar",
        max_display=top_n,
        show=False,
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return plt.gcf()