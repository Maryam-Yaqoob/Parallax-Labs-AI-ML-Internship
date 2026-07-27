# Parallax Labs — ML Internship

Single repository for all Parallax Labs internship submissions (updated week over week —
not recreated per task).

**Student:** Maryam Yaqoob (FA23-BAI-025)

---

## Progress Log

### Task 1 — Environment Setup & Dataset Cleaning Pipeline (in progress)

**Status:** Stage 1 complete — environment setup + verification script.

**What's done:**
- `requirements.txt` finalized with all libraries needed for dataset acquisition,
  validation, and text cleaning
- `verify_setup.py` — verification script that imports every dependency **and**
  runs a small functional check for each (not just `import x`), so a broken install
  is caught immediately instead of failing later mid-pipeline
- `load_and_validate_dataset.py` — acquires the 20 Newsgroups dataset (~18,846 docs,
  headers/footers/quotes intentionally kept as real-world noise) and validates it:
  null checks, empty/whitespace checks, duplicate checks, encoding checks, and
  document-length distribution. Outputs `data_raw.csv` and `data_validation_summary.md`.
- `text_cleaning.py` — modular, independently-testable cleaning functions:
  `fix_encoding`, `remove_newsgroup_headers`, `remove_quoted_lines`,
  `remove_signature`, `remove_html_tags`, `remove_urls`, `remove_email_addresses`,
  `remove_emoji`, `normalize_whitespace`, `detect_language`, and a composed
  `clean_text()` pipeline. Every function returns `""` (or `"unknown"` for
  language detection) instead of crashing on `None`/empty/whitespace-only input.
- `test_text_cleaning.py` — 42 unit tests covering every function, including
  edge cases (None, empty, whitespace-only, non-string input, mixed-language
  text, idempotency). All 42 passing.
- `generate_clean_dataset.py` — applies the full cleaning pipeline + language
  detection to every document in `data_raw.csv`, and outputs:
  - `processed_dataset.csv` — final clean dataset, ready for chunking/embedding
  - `data_quality_report.md` — before/after quality report (length reduction,
    docs that became empty after cleaning, language distribution, post-clean
    duplicates)

**Task 1 status: complete.** All 4 stages done — environment, dataset
acquisition/validation, cleaning functions, unit tests + final clean dataset.

**Note:** repo uses a flat file structure (no subfolders) for simplicity.

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key libraries:

| Library | Purpose |
|---|---|
| pandas, numpy | Data handling |
| scikit-learn | Dataset loader (20 Newsgroups) |
| nltk | Tokenization |
| langdetect | Detecting mixed-language text |
| ftfy | Fixing encoding artifacts (mojibake) |
| chardet | Detecting raw byte encoding |
| emoji | Detecting/handling emoji in text |
| pytest | Unit testing |
| tqdm | Progress bars for large-loop operations |
| tiktoken | Token-aware chunking (used in a later task, not this one) |

**Design note (Stage 3):** `text_cleaning.py` does not lowercase or strip
punctuation by default, since this dataset is being prepared for embeddings,
and embedding models generally do better on natural casing/punctuation. This
is flagged here rather than assumed silently — let me know if the internship
task expects lowercasing and we'll add it as an explicit step.

---

## Setup Instructions

```bash
git clone https://github.com/Maryam-Yaqoob/Parallax-Labs-AI-ML-Internship.git
cd Parallax-Labs-AI-ML-Internship
pip install -r requirements.txt

# Verify every library imports and works correctly:
python verify_setup.py

# Acquire and validate the raw dataset:
python load_and_validate_dataset.py

# Run the unit tests for the cleaning functions:
pytest test_text_cleaning.py -v

# Generate the final cleaned dataset + quality report:
python generate_clean_dataset.py
```

Expected output: `Result: 11/11 checks passed` (a `[WARN]` on `tiktoken` is fine if
your network blocks its one-time encoding file download — it does not affect this
task, which doesn't use tiktoken yet).

---

## Repository Structure

Flat structure — everything in the repo root:

```
Parallax-Labs-AI-ML-Internship/
├── README.md
├── requirements.txt
├── .gitignore
├── verify_setup.py
├── load_and_validate_dataset.py
├── data_raw.csv                # generated when you run load_and_validate_dataset.py
├── data_validation_summary.md  # generated when you run load_and_validate_dataset.py
├── text_cleaning.py
├── test_text_cleaning.py
├── generate_clean_dataset.py
├── data_quality_report.md      # generated when you run generate_clean_dataset.py
└── processed_dataset.csv       # generated when you run generate_clean_dataset.py (final output)
```
