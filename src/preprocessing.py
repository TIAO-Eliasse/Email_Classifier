"""
Preprocessing module: lightweight text cleaning and construction of
the final text field used for model input (TF-IDF vectorization
and/or Sentence Transformer embeddings, depending on the model used
in modelling.py).

Design note: the raw 'message' field (extracted by ingestion.py from
the email's middle paragraph(s)) is already clean — no HTML residue,
no non-ASCII characters, no excessive whitespace (verified during
EDA). Signature and raw greeting are deliberately excluded from the
text fed to the model: signatures are near-unique per email (client
names) and add no generalisable signal for either TF-IDF or
embedding-based models, while greeting content is instead captured
as a structured feature (see features.py) rather than injected as
raw text.

Note on model-specific preprocessing: unlike TF-IDF, Sentence
Transformer-based models (e.g. SetFit) do not require additional
steps such as lowercasing or stopword removal — their tokenizers and
pretrained embeddings already handle casing and function words
effectively. No extra preprocessing branch is needed here for that
case; the same 'clean_text' output is used for both approaches.
"""

import re


def clean_text(text):
    """Normalize whitespace in a text field. Returns '' for missing values."""
    if text is None or isinstance(text, float):  # handles pandas NaN
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_full_text(row):
    """
    Construct the text field used for TF-IDF: subject + message only.
    Signature and greeting are excluded here (see module docstring).
    """
    subject = clean_text(row.get("subject"))
    message = clean_text(row.get("message"))
    return f"{subject}. {message}".strip()


def add_clean_text_column(df):
    """Add a 'clean_text' column to a DataFrame with subject/message fields."""
    df = df.copy()
    df["clean_text"] = df.apply(build_full_text, axis=1)
    return df