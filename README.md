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

### Task 2 — Chunking, Embeddings & Vector DB

**Status: complete.**

**What's done:**

- `chunking.py` — recursive character text splitting (custom implementation,
  same approach as LangChain's `RecursiveCharacterTextSplitter`), chunk
  size 1000 / overlap 100. Produces 35,072 chunks from `processed_dataset.csv`.
- `test_chunking.py` — unit tests covering edge cases (None/empty input,
  overlap >= chunk size, paragraph-boundary splitting, hard character-split
  fallback, unique chunk IDs).
- `embeddings.py` — generates embeddings with `sentence-transformers`
  (`all-MiniLM-L6-v2`, 384-dim) and logs timing to `logs/embedding_perf.csv`.
- `chroma_ingest.py` — sets up a ChromaDB collection and ingests all
  chunks + embeddings.
- `test_retrieval.py` — runs domain-relevant semantic search queries
  against the collection and logs latency to `logs/retrieval_perf.csv`.

See [`NOTES.md`](./NOTES.md) for chunking strategy details, real embedding/
retrieval performance numbers, example query results, and known data
quality limitations.

**Running Task 2** (after Task 1's `processed_dataset.csv` exists):

\```
python chunking.py
python embeddings.py
python chroma_ingest.py
python test_retrieval.py
\```

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
```

## Running the Pipeline

Run these in order — each step depends on the file(s) produced by the one before it.

**1. Verify the environment**
```bash
python verify_setup.py
```
Expected output: `Result: 11/11 checks passed` (a `[WARN]` on `tiktoken` is fine if
your network blocks its one-time encoding file download — it does not affect this
task, which doesn't use tiktoken yet).

**2. Acquire and validate the raw dataset**
```bash
python load_and_validate_dataset.py
```
Downloads 20 Newsgroups (~18,846 docs), then writes `data_raw.csv` and
`data_validation_summary.md`. Expected: `Total documents: 18846`, with
`Null entries`, `Duplicate documents`, and `Encoding issues` all at `0`.

**3. Run the unit tests for the cleaning functions**
```bash
python -m pytest test_text_cleaning.py -v
```
Expected: `42 passed`.

> **Important — use `python -m pytest`, not a bare `pytest` command.**
> On some setups, notably Windows/PowerShell, pip installs `pytest.exe` into a
> `Scripts` folder that isn't on `PATH`, so a bare `pytest` command fails with
> `pytest : The term 'pytest' is not recognized as the name of a cmdlet...`
> even though pytest is installed correctly. `python -m pytest` always works
> since it just asks the already-on-`PATH` `python` interpreter to run pytest
> as a module. **Evaluators should use `python -m pytest test_text_cleaning.py -v`
> to run the tests correctly.**

**4. Generate the final cleaned dataset + quality report**
```bash
python generate_clean_dataset.py
```
Requires `data_raw.csv` from step 2. Outputs `processed_dataset.csv` and
`data_quality_report.md`. Expected: `DONE — Stage 4 complete`.

**All steps in one block** (for copy-paste convenience):
```bash
python verify_setup.py
python load_and_validate_dataset.py
python -m pytest test_text_cleaning.py -v
python generate_clean_dataset.py
```

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
