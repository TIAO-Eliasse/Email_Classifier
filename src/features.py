"""
Feature engineering module: extracts structured signals from emails
beyond the raw TF-IDF text representation.

Each feature here is motivated by a pattern observed during EDA, not
added speculatively. Their individual contribution should be
validated empirically (see notebooks/01_data_exploration.ipynb)
before being kept in the final modelling pipeline; features that do
not show a clear signal are documented as "tested, not retained"
rather than silently dropped.
"""

import re
import pandas as pd

INTERNAL_DOMAIN = "redrock.com"
DEPARTMENT_KEYWORDS = ["department", "team", "services", "advisor", "planner", "compliance", "redrock"]


def _is_missing(value):
    """True for None, NaN, or empty/whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def extract_sender_domain(sender):
    """Extract the domain part of an email address. Returns '' if malformed or missing."""
    if _is_missing(sender) or "@" not in sender:
        return ""
    return sender.split("@")[-1].strip().lower()


def is_internal_sender(sender):
    """True if the sender's domain matches RedRock's internal domain."""
    return extract_sender_domain(sender) == INTERNAL_DOMAIN


def is_internal_signature(signature):
    """True if the signature looks like an internal RedRock department, not a client name."""
    if _is_missing(signature):
        return False
    return any(kw in signature.lower() for kw in DEPARTMENT_KEYWORDS)


def extract_greeting_target(greeting):
    """
    Extracts the addressee from a greeting like 'Dear Student Loan
    Department,' -> 'Student Loan Department'. Returns None if no
    greeting or no clear pattern.
    """
    if _is_missing(greeting):
        return None
    match = re.match(r"^(?:Dear|Hello|Hi)\s+([^,]+),?", greeting.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def is_department_greeting(greeting_target):
    """True if the greeting target looks like a department/team, not a generic salutation."""
    if _is_missing(greeting_target):
        return False
    return any(kw in greeting_target.lower() for kw in DEPARTMENT_KEYWORDS)


def has_dollar_amount(text):
    """True if the text mentions a dollar amount (e.g. $3,200)."""
    if _is_missing(text):
        return False
    return bool(re.search(r"\$[\d,]+(\.\d+)?", text))


def count_question_marks(text):
    """Number of '?' characters — proxy for request/inquiry-style emails."""
    if _is_missing(text):
        return 0
    return text.count("?")


def add_engineered_features(df):
    """Add all structured feature columns to a DataFrame."""
    df = df.copy()
    df["sender_domain"] = df["sender"].apply(extract_sender_domain)
    df["is_internal_sender"] = df["sender"].apply(is_internal_sender)
    df["is_internal_signature"] = df["signature"].apply(is_internal_signature)
    df["greeting_target"] = df["greeting"].apply(extract_greeting_target)
    df["is_department_greeting"] = df["greeting_target"].apply(is_department_greeting)
    df["has_dollar_amount"] = df["message"].apply(has_dollar_amount)
    df["question_mark_count"] = df["message"].apply(count_question_marks)
    return df