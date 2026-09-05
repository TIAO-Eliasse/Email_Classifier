"""
Ingestion module: parses RedRock's HTML email files into a
structured pandas DataFrame.
"""

import os
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup


def parse_email_html(filepath):
    """Parse a single .html email file and return a dict of structured fields."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    soup = BeautifulSoup(raw, "html.parser")

    def get_field(name):
        tag = soup.find("div", {"data-field": name})
        return tag.get_text(strip=True) if tag else None

    email_id = get_field("email_id")
    subject = get_field("subject")
    sender = get_field("sender")
    date_received = get_field("date_received")

    # The email body itself contains a nested HTML document, so we
    # re-parse it and pull text only from its <body> tag to avoid
    # leaking the nested <title> into the extracted text.
    #
    # That nested body is structured as separate <p> tags:
    # typically [greeting, message, signature]. We split them out
    # individually so the signature (near-unique per email, mostly
    # noise for classification) can be excluded from the text used
    # for TF-IDF, while still being available as a standalone
    # feature (e.g. to detect internal RedRock department signatures).
    body_div = soup.find("div", {"class": "email-body"})
    greeting, message, signature, body_text = None, None, None, None

    if body_div:
        inner_soup = BeautifulSoup(body_div.decode_contents(), "html.parser")
        inner_body_tag = inner_soup.find("body")
        target = inner_body_tag if inner_body_tag else inner_soup

        paragraphs = [p.get_text(strip=True) for p in target.find_all("p")]
        paragraphs = [p for p in paragraphs if p]  # drop empty <p> tags

        if len(paragraphs) >= 3:
            greeting = paragraphs[0]
            signature = paragraphs[-1]
            message = " ".join(paragraphs[1:-1])
        elif len(paragraphs) == 2:
            # edge case: only 2 paragraphs, assume [message, signature]
            greeting, message, signature = None, paragraphs[0], paragraphs[1]
        elif len(paragraphs) == 1:
            message = paragraphs[0]
        else:
            # fallback if there are no <p> tags at all
            message = target.get_text(separator=" ", strip=True)

        body_text = target.get_text(separator=" ", strip=True)  # kept for backward compatibility / debugging

    return {
        "email_id": email_id,
        "subject": subject,
        "sender": sender,
        "date_received": date_received,
        "greeting": greeting,
        "message": message,
        "signature": signature,
        "body": body_text,
        "source_filename": Path(filepath).name,
    }


def build_dataframe(directory):
    """Parse every .html file in a directory and return a DataFrame."""
    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if f.endswith(".html"))

    records = []
    for fname in files:
        filepath = directory / fname
        try:
            records.append(parse_email_html(filepath))
        except Exception as e:
            print(f"Error parsing {fname}: {e}")

    df = pd.DataFrame(records)
    df["email_id"] = df["email_id"].astype(str).str.strip()
    return df


def load_labels(labels_path):
    """Load the ground-truth labels CSV."""
    labels_df = pd.read_csv(labels_path)
    return labels_df