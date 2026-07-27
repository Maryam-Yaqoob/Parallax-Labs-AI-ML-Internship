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

**What's next (Stage 3):**
- Write `text_cleaning.py` with robust cleaning functions (empty text,
  mixed-language text, encoding artifacts, HTML/quoted-reply noise)
- Unit tests in `test_text_cleaning.py`
- Data quality report in `data_quality_report.md`
- Final clean dataset (`processed_dataset.csv`), ready for chunking/embedding in a later task

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

---

## Setup Instructions

```bash
git clone https://github.com/Maryam-Yaqoob/Parallax-Labs-AI-ML-Internship.git
cd Parallax-Labs-AI-ML-Internship
pip install -r requirements.txt

# Verify every library imports and works correctly:
python verify_setup.py
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
├── text_cleaning.py           # added in Stage 3
├── test_text_cleaning.py      # added in Stage 4
├── data_quality_report.md     # added in Stage 4
└── processed_dataset.csv      # added in Stage 4/5 (final clean output)
```
