"""
Modelling module: builds, compares, and calibrates classification
pipelines for RedRock email routing.

Design rationale:
- Given the small training set (44 labelled emails across 5 classes),
  a simple, well-regularised linear model on TF-IDF features is used
  as the primary approach, following evidence that bag-of-words
  baselines remain competitive with more complex architectures on
  small/medium text classification tasks (Galke & Scherp, 2022/2024).
- Structured features (is_internal_sender, is_internal_signature,
  is_department_greeting, question_mark_count) were selected from a
  larger candidate set based on empirical validation against
  true_category (see notebooks/01_data_exploration.ipynb, Section
  5.1). has_dollar_amount was tested and excluded: its signal was too
  sparse (at most 1-2 positive examples per class) to generalise
  reliably on this dataset size.
- Stratified K-Fold cross-validation is used instead of a single
  train/validation split, since a one-off split with this few
  examples per class would be too noisy to trust.
- Confidence scores are calibrated via Platt scaling (sigmoid
  method), recommended by scikit-learn's documentation specifically
  for small sample sizes, rather than isotonic regression.
"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Finalised after empirical validation (see EDA notebook, Section 5.1):
# - is_internal_signature: near-perfect separator for "Other" (100% vs 0%)
# - is_department_greeting: strong separator for "Other" (0% vs 57-83%)
# - is_internal_sender: weak but non-zero signal, kept
# - question_mark_count: meaningful gradient across request-type categories
# - has_dollar_amount: EXCLUDED, too sparse to generalise
STRUCTURED_BOOL_COLS = ["is_internal_sender", "is_internal_signature", "is_department_greeting"]
STRUCTURED_NUM_COLS = ["question_mark_count"]


def build_text_pipeline(text_columns, structured_bool_cols=None, structured_num_cols=None,
                         classifier=None, max_features=2000, ngram_range=(1, 2),
                         subject_max_features=300):
    """
    Build a scikit-learn Pipeline combining TF-IDF text feature(s) with
    optional structured (boolean/numeric) features, feeding into a
    classifier of choice.

    text_columns: either a single column name (str) for the fused-text
    approach, or a tuple ("subject", "message") for the separated-text
    approach (each vectorised independently, subject given a smaller
    vocabulary budget since it is short and its signal is
    disproportionately strong relative to its length).
    """
    structured_bool_cols = structured_bool_cols or []
    structured_num_cols = structured_num_cols or []
    classifier = classifier or LogisticRegression(max_iter=1000, class_weight="balanced")

    transformers = []
    if isinstance(text_columns, str):
        transformers.append(
            ("text", TfidfVectorizer(stop_words="english", ngram_range=ngram_range,
                                      max_features=max_features), text_columns)
        )
    else:
        subject_col, body_col = text_columns
        transformers.append(
            ("subject_tfidf", TfidfVectorizer(stop_words="english",
                                               max_features=subject_max_features), subject_col)
        )
        transformers.append(
            ("body_tfidf", TfidfVectorizer(stop_words="english", ngram_range=ngram_range,
                                            max_features=max_features), body_col)
        )

    if structured_bool_cols:
        transformers.append(("bool_features", "passthrough", structured_bool_cols))
    if structured_num_cols:
        transformers.append(("num_features", "passthrough", structured_num_cols))

    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("preprocessor", preprocessor), ("clf", classifier)])


def compare_text_representations(df, y, n_splits=5, random_state=42):
    """
    Compare fused text (subject+message in one TF-IDF) vs separated
    text (subject and message vectorised independently), holding the
    classifier and structured features constant, to decide which
    representation to use going forward -- empirically, not by
    assumption.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipeline_fused = build_text_pipeline(
        "clean_text", STRUCTURED_BOOL_COLS, STRUCTURED_NUM_COLS, clf
    )
    X_fused = df[["clean_text"] + STRUCTURED_BOOL_COLS + STRUCTURED_NUM_COLS]
    scores_fused = cross_val_score(pipeline_fused, X_fused, y, cv=skf, scoring="f1_macro")

    pipeline_separated = build_text_pipeline(
        ("subject", "message"), STRUCTURED_BOOL_COLS, STRUCTURED_NUM_COLS, clf
    )
    X_separated = df[["subject", "message"] + STRUCTURED_BOOL_COLS + STRUCTURED_NUM_COLS]
    scores_separated = cross_val_score(pipeline_separated, X_separated, y, cv=skf, scoring="f1_macro")

    return {
        "fused": (scores_fused.mean(), scores_fused.std()),
        "separated": (scores_separated.mean(), scores_separated.std()),
    }


def compare_models(X, y, text_columns, n_splits=5, random_state=42):
    """
    Compare several simple, well-tuned baseline classifiers via
    stratified cross-validation, scored on F1-macro (chosen over raw
    accuracy so minority classes are not overshadowed by the majority
    class -- important in a regulatory-routing context where every
    category matters equally).

    Returns a dict of {model_name: (mean_f1, std_f1)}.
    """
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
        "naive_bayes": MultinomialNB(),
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for name, clf in candidates.items():
        pipeline = build_text_pipeline(
            text_columns, STRUCTURED_BOOL_COLS, STRUCTURED_NUM_COLS, clf
        )
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro")
        results[name] = (scores.mean(), scores.std())

    return results



def train_and_save_best_model(train_labeled, y, models_dir, cv_splits=5, random_state=42):
    """
    Fit the winning pipeline (LinearSVC, separated subject/message text)
    on all available training data, cross-validate it one final time for
    a recorded F1-macro, and persist both the pipeline and its metadata.

    This consolidates what was previously inline notebook code (fitting,
    cross-validating, pickling, writing metadata.json) into a single,
    reusable function -- keeping 02_modelling.ipynb's cells short.
    """
    import pickle
    import json
    from pathlib import Path

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    text_columns = ("subject", "message")
    X_best = train_labeled[["subject", "message"] + STRUCTURED_BOOL_COLS + STRUCTURED_NUM_COLS]

    best_clf = LinearSVC(class_weight="balanced", random_state=random_state)
    best_pipeline = build_text_pipeline(text_columns, STRUCTURED_BOOL_COLS, STRUCTURED_NUM_COLS, best_clf)

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(best_pipeline, X_best, y, cv=skf, scoring="f1_macro")

    best_pipeline.fit(X_best, y)

    with open(models_dir / "best_model_raw.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    metadata = {
        "model_type": "LinearSVC",
        "text_representation": "separated",
        "text_columns": list(text_columns),
        "structured_features": STRUCTURED_BOOL_COLS + STRUCTURED_NUM_COLS,
        "classes": sorted(y.unique()),
        "n_samples": len(y),
        "f1_macro_cv_mean": float(cv_scores.mean()),
        "f1_macro_cv_std": float(cv_scores.std()),
        "cv_scores_all": cv_scores.tolist(),
        "notes": "Best performer from 02_modelling comparison. Use for calibration.",
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return best_pipeline, metadata