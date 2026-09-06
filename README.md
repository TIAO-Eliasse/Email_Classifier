# RedRock Email Classification

Automated classification of client emails into 5 business categories
(Account Management, Investment Advisory, Loan Processing, Insurance
Claims, Other), for departmental routing at a financial services
company. Built entirely with open-source Python tools.

Live demo: https://emailclassifier-app.streamlit.app/

## Setup & Run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Command line (this is the main way to run it — no browser needed):
```bash
python main.py predict    # classify test/, write outputs/results.csv
```

There's also a small web UI if you'd rather click than type:
```bash
streamlit run app.py
```
It just wraps `main.py`'s train/predict calls — I didn't want two
copies of the pipeline logic floating around, so the app and the CLI
share the exact same code.

## Architecture

```
src/
├── ingestion.py       # HTML parsing -> structured DataFrame
├── preprocessing.py   # Text cleaning, subject/message extraction
├── features.py        # Structured feature engineering
├── modelling.py        # Pipeline, model comparison, final training
├── calibration.py      # Platt scaling, Brier score evaluation
├── uncertainty.py      # Entropy/margin, MAPIE, routing threshold
└── explainability.py   # SHAP audit trail, comparison plots
main.py                 # CLI: train / predict
streamlit_app.py        # optional web UI, wraps main.py
notebooks/               # 01 exploration, 02 modelling, 03 calibration,
                         # 04 uncertainty, 05 explainability
```

## Key decisions and why

**Filenames aren't real identifiers.** `email_1.html` is just
alphabetical numbering inside each folder — it's arbitrary and
different between `train/` and `test/`. The actual identifier is the
`email_id` field embedded in each file's metadata. I checked for
overlap between train and test on that real ID and on exact body
text, and found none, so there's no leakage hiding behind the
filenames.

**Client names are dropped from the text features.** Each email ends
with a signature (a name), and with only 44 training examples those
names are basically unique per email — keeping them in the TF-IDF
vocabulary would just be noise the model could latch onto. I split
the signature out during ingestion and instead check whether it looks
like an internal department vs. a client name, which turned out to
be a genuinely useful signal (`is_internal_signature`).

**TF-IDF + linear models, not a fine-tuned LLM.** The brief allows
either "LLMs or other NLP techniques," and with 44 labelled examples
across 5 classes, fine-tuning a transformer is a good way to
overfit. Two things back this up: Galke & Scherp (2022/2024) show
that well-tuned bag-of-words baselines hold up against much larger
architectures on small text classification tasks, and Borg et al.
(2020, *Neural Computing and Applications*) — an actual email-routing
study — found SVM beats LSTM once the dataset gets too small for the
LSTM to earn its keep. TF-IDF is a legitimate NLP technique, and
given the data volume here, I think it's the right one.

**Subject and message are vectorised separately, not glued together.**
I tested both: separate TF-IDF vectors for subject and message beat a
single fused text field (F1-macro 0.894 vs 0.786), and did so more
consistently across folds (std 0.107 vs 0.190).

## Feature engineering

I didn't add any structured feature just because it seemed plausible —
each one got checked against `true_category` first (see
`notebooks/01_data_exploration.ipynb`, Section 5.1).

| Feature | What I found | Kept? |
|---|---|---|
| `is_internal_signature` | Perfectly separates "Other" (100% vs 0%) | Yes |
| `is_department_greeting` | Strongly separates "Other" (0% vs 57–83%) | Yes |
| `question_mark_count` | Clear gradient across request-type categories | Yes |
| `is_internal_sender` | Weak, but a real, non-zero effect | Yes |
| `has_dollar_amount` | Only 1-2 positive examples per class — too thin to trust | No |

## Model comparison (5-fold stratified CV, F1-macro)

| Representation | Model | F1-macro |
|---|---|---|
| Fused | Logistic Regression | 0.786 (±0.190) |
| Separated | Logistic Regression | 0.894 (±0.107) |
| Separated | **Linear SVM** | **0.947 (±0.076)** |
| Separated | Naive Bayes | 0.655 (±0.070) |

Final model: Linear SVM on separated subject/message TF-IDF plus the
structured features above, calibrated with Platt scaling so the
`confidence_score` output actually means something.

## Calibration, uncertainty, and explainability

The brief asks for a confidence score next to every prediction. I
didn't want that number to just be a raw model score dressed up to
look like a probability, so I put some real work into making it
trustworthy.

**Calibration.** SVM decision scores aren't probabilities out of the
box. I wrapped the model in `CalibratedClassifierCV` (Platt/sigmoid,
`cv=3` — the small classes don't support more folds) and checked the
result with Brier score rather than just eyeballing a plot: it
improved by 26.1% on average across the 5 classes. Reliability
diagrams before/after are in `notebooks/03_calibration.ipynb`.

**Uncertainty**, at three levels:
- Entropy and margin over the calibrated probabilities — cheap, and
  always available (mean entropy 1.53 out of a possible 2.32 bits,
  mean margin 0.48).
- An operating threshold (0.575) picked to get zero routing errors on
  cross-validated predictions, which sorts ~75% of training emails
  into auto-forward and ~25% into human review. One honest caveat
  here: the threshold is chosen and tested on the same predictions,
  which is a bit optimistic — I'd want to re-check it on a proper
  held-out set before trusting it in production.
- A conformal prediction pass using MAPIE. This works, but I'm
  treating it as a demonstration rather than a real result — its
  coverage guarantee needs more calibration data than the ~11
  examples left over after carving out a split from 44 emails.

**Explainability.** I used SHAP's exact linear explainer (no
approximation needed for a linear model) so every prediction can be
traced to the words that drove it, and each one is labelled as either
supporting or working against the predicted category — e.g.
`claim (+0.48, supports)` — rather than just handing over a bare
number. I also plot which features drive auto-forwarded emails versus
deferred ones side by side, which is the kind of evidence that's
actually useful in a compliance conversation.

**One thing worth being upfront about**: the model I actually ship is
retrained on all 44 labelled emails, to squeeze out as much signal as
possible before deployment. That means this exact model is never
tested on data it hasn't seen — there isn't any left. The
cross-validation numbers above measure how well the training
*process* performs, not this specific model instance. Since the
deployed model sees more data than any individual CV fold, it should
do at least as well — but that's a reasonable expectation, not
something I've directly measured. At 44 examples, I don't think
there's a way around this.

## How I'm measuring success

- F1-macro instead of raw accuracy, because the classes aren't
  balanced (6 to 13 examples each) and I want minority categories to
  actually matter in the score.
- A confusion matrix per category, since knowing *which* categories
  get mixed up is more useful than one aggregate number.
- Calibration quality (Brier score, reliability diagrams) — a
  confidence score is only worth anything if it tracks real
  correctness rather than model overconfidence.
- The abstention rate at the chosen threshold, and separately,
  accuracy within the auto-forwarded group versus the deferred group —
  to confirm the threshold is doing what it's supposed to (zero errors
  among the emails sent through automatically).

## What else could help, with more data

- PII detection and masking — not built in this version (see
  Limitations), but it's the kind of thing that would need to run
  before anything leaves the local environment, which is part of why
  I avoided routing email content through a third-party LLM API in
  the first place.
- Feeding back corrections when support staff re-route a
  misclassified email.
- Attachment metadata — a claim form attached is a strong signal for
  Insurance Claims.
- Thread context — right now only the latest message gets classified;
  earlier messages in the same thread could help with ambiguous cases.

## Limitations

- 44 training examples across 5 classes isn't a lot. Cross-validation
  scores move around more than I'd like, and the smallest classes get
  evaluated on 1-2 examples in some folds.
- The deployed model is never directly tested on unseen data (see
  above) — a real constraint of working at this scale, not a gap in
  method.
- The routing threshold is picked and evaluated on the same
  predictions; a nested cross-validation estimate is more trustworthy,
  and I'd defer to that number where it disagrees with the simpler one.
- MAPIE conformal prediction is included to show the approach, not as
  something I'd stand behind as a production number at this sample size.
- No PII detection or masking yet.
