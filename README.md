# Parallax Labs — ML Internship

Single repository for all Parallax Labs internship submissions (updated week over week —
not recreated per task).

**Student:** Maryam Yaqoob (FA23-BAI-025)

---

## Progress Log

### Task 1 — Environment Setup & Dataset Cleaning Pipeline (in progress)

**Status:** Stage 1 complete — environment setup + verification script.

**What's done:**
- Project scaffold created (`src/`, `tests/`, `data/raw`, `data/processed`, `reports/`)
- `requirements.txt` finalized with all libraries needed for dataset acquisition,
  validation, and text cleaning
- `src/verify_setup.py` — verification script that imports every dependency **and**
  runs a small functional check for each (not just `import x`), so a broken install
  is caught immediately instead of failing later mid-pipeline

**What's next (Stage 2):**
- Acquire the dataset (20 Newsgroups, ~18,800 real-world documents)
- Validate it: null checks, encoding checks, duplicate checks
- Write `src/text_cleaning.py` with robust cleaning functions (empty text,
  mixed-language text, encoding artifacts, HTML/quoted-reply noise)
- Unit tests in `tests/test_text_cleaning.py`
- Data quality report in `reports/data_quality_report.md`
- Final clean dataset in `data/processed/`, ready for chunking/embedding in a later task

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
git clone https://github.com/Maryam-Yaqoob/parallax-labs-ml-internship.git
cd parallax-labs-ml-internship
pip install -r requirements.txt

# Verify every library imports and works correctly:
python src/verify_setup.py
```

Expected output: `Result: 11/11 checks passed` (a `[WARN]` on `tiktoken` is fine if
your network blocks its one-time encoding file download — it does not affect this
task, which doesn't use tiktoken yet).

---

## Repository Structure

```
parallax-labs-ml-internship/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # original dataset (not committed — re-downloadable)
│   └── processed/      # cleaned, validated output
├── src/
│   ├── verify_setup.py
│   └── text_cleaning.py       # added in Stage 3
├── tests/
│   └── test_text_cleaning.py  # added in Stage 4
└── reports/
    └── data_quality_report.md # added in Stage 4
```
