
## Key design decisions

- **Filename ≠ identifier**: `email_1.html` numbering is per-folder and
  arbitrary. The true identifier is the internal `email_id` field.
  Verified: 0 overlap between train/test on `email_id` and exact body
  content.
- **Signature excluded from text features**: near-unique per email
  (client names), adds noise rather than signal on a 44-example
  training set. Kept as a separate structured feature
  (`is_internal_signature`) instead.
- **TF-IDF + linear models over LLM/Transformer fine-tuning**: with
  only 44 labelled examples across 5 classes, fine-tuning a
  Transformer risks severe overfitting. This is supported by Galke &
  Scherp (2022/2024), who show bag-of-words baselines remain
  competitive with more complex architectures on constrained text
  classification tasks, and by Borg et al. (2020, *Neural Computing
  and Applications*), an email-routing study which explicitly
  recommends SVM over LSTM when data is insufficient for the latter.
  The brief explicitly permits either "LLMs or other NLP techniques"
  — TF-IDF is a valid NLP technique, chosen here as the *appropriate*
  one for this data volume.
- **Subject and message vectorised separately** (not concatenated):
  empirically confirmed better and more stable (F1-macro 0.894 vs
  0.786 for fused text, std 0.107 vs 0.190), consistent with Borg et
  al.'s observation that subject and body often carry different
  signal.

## Feature engineering — empirically validated

Each structured feature was tested against `true_category` before
being retained (see notebook, Section 5.1):

| Feature | Signal | Retained |
|---|---|---|
| `is_internal_signature` | Perfect separator for "Other" (100% vs 0%) | ✅ |
| `is_department_greeting` | Strong separator for "Other" (0% vs 57-83%) | ✅ |
| `question_mark_count` | Gradient across request-type categories | ✅ |
| `is_internal_sender` | Weak but non-zero | ✅ |
| `has_dollar_amount` | Too sparse (1-2 positive examples/class) | ❌ Tested, not retained |

## Model comparison (5-fold stratified CV, F1-macro)

| Representation | Model | F1-macro |
|---|---|---|
| Fused (subject+message) | Logistic Regression | 0.786 (±0.190) |
| **Separated** | Logistic Regression | 0.894 (±0.107) |
| **Separated** | **Linear SVM** | **0.947 (±0.076)** |
| Separated | Naive Bayes | 0.655 (±0.070) |

**Final model**: Linear SVM on separated subject/message TF-IDF +
structured features, calibrated via Platt scaling (`sigmoid` method)
for reliable `confidence_score` output.

## Measuring success

- **F1-macro** (not raw accuracy) — chosen because classes are
  imbalanced (6-13 examples/class); macro-averaging ensures minority
  categories aren't overshadowed by majority ones, important since
  every category matters equally for correct routing.
- **Confusion matrix per category** — surfaces which categories get
  confused with which, more actionable than a single aggregate score.
- **Calibration quality** (reliability diagram, before/after Platt
  scaling) — a `confidence_score` is only useful for compliance if it
  reflects true correctness likelihood, not raw model overconfidence.
- **Abstention rate at a chosen confidence threshold** — proportion
  of emails where the model should defer to a human reviewer rather
  than auto-route, directly tied to regulatory risk mitigation.

## Additional features / data sources for improved compliance

- **PII detection & anonymisation** (not implemented in this MVP —
  see limitations below).
- **Historical routing corrections** (if support staff re-route
  misclassified emails, that signal could be fed back for retraining).
- **Attachment metadata** (e.g. presence of a claim form or policy
  document could strengthen Insurance Claims signal).
- **Thread/conversation context** (currently only the latest message
  is classified; prior messages in a thread could disambiguate).

## Limitations

- Training set is small (44 examples, 5 classes) — cross-validation
  scores carry meaningful variance; minority classes (6 examples)
  are evaluated on as few as 1-2 examples per fold.
- No PII detection/masking implemented (see above).
- Conformal prediction (MAPIE) was considered for formal uncertainty
  quantification but not implemented: reliably requires a held-out
  calibration set, impractical to carve out of only 44 examples.