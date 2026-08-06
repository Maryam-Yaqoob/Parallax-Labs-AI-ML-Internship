# Parallax Labs — ML Internship

Single repository for all Parallax Labs internship submissions (updated week over week —
not recreated per task).

**Student:** Maryam Yaqoob (FA23-BAI-025)

---

## Progress Log

### Task 1 — Environment Setup & Dataset Cleaning Pipeline

**Status: complete.**

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
- `chroma_ingest.py` — sets up a ChromaDB collection (`"documents"`) and
  ingests all chunks + embeddings.
- `test_retrieval.py` — runs domain-relevant semantic search queries
  against the collection and logs latency to `logs/retrieval_perf.csv`.

See [`NOTES.md`](./NOTES.md) for chunking strategy details, real embedding/
retrieval performance numbers, example query results, and known data
quality limitations.

**Running Task 2** (after Task 1's `processed_dataset.csv` exists):

```bash
python chunking.py
python embeddings.py
python chroma_ingest.py
python test_retrieval.py
```

### Task 3 — DeepSeek/OpenRouter Generation (RAG completion)

**Status: complete.** Adds an LLM generation layer on top of the retrieval
pipeline, with prompt engineering, robust API error handling, hallucination/
out-of-domain guards, and end-to-end latency logging.

**Integration note:** `retriever.py` was originally written as a best-guess
against `.gitignore`'s hint of a future `chroma_ingest.py` → `chroma_db/`
setup, before checking whether Task 2 was already on `main` — it was. Wiring
`retriever.py` up to the real `chroma_ingest.py` surfaced two mismatches:
the collection name (`retriever.py` assumed `"parallax_docs"`, the real
collection is `"documents"`), and the embedding path (`retriever.py` queried
with `query_texts=`, letting Chroma embed the query with its own default
embedding function, while `chroma_ingest.py` stores explicit
`all-MiniLM-L6-v2` vectors — two different embedding spaces, which wouldn't
error, just silently degrade retrieval quality). Both are now fixed:
`ChromaRetriever` opens the `"documents"` collection and embeds queries with
the same `all-MiniLM-L6-v2` model via `query_embeddings=`, so queries and
stored chunks live in the same space. A dependency-light `TfidfRetriever`
fallback is kept for environments without a populated `chroma_db/`.

Verified end-to-end after the fix: `test_generate_answer.py` still 18/18
passing, and live runs against the real `chroma_db` return `status=answered`
with grounded, cited answers (e.g. spacecraft-communication and epilepsy
queries pulling real dataset content) instead of falling back to TF-IDF.

**What's done:**
- `retriever.py` — `ChromaRetriever` (opens the `"documents"` collection
  built by `chroma_ingest.py`, embeds queries with the matching
  `all-MiniLM-L6-v2` model) + `TfidfRetriever` fallback + `get_retriever()`
  factory that picks whichever is available.
- `generate_answer.py` — DeepSeek-via-OpenRouter generation:
  - **Prompt engineering:** a system prompt that constrains the model to the
    retrieved context, requires inline citations `[1]`, `[2]`, ..., and gives
    an explicit instruction to abstain rather than guess; context is injected
    as a numbered, token-budgeted block (`build_context_block`).
  - **Error handling:** timeouts, connection errors, 429 rate limits (with
    `Retry-After` support), 5xx retried with exponential backoff, 4xx
    (including context-length/token-limit errors) failed fast without
    retrying, malformed responses caught explicitly. All in
    `call_openrouter()`.
  - **Hallucination / out-of-domain handling:** a retrieval-confidence gate
    (`RETRIEVAL_CONFIDENCE_THRESHOLD`) that returns a canned "out of domain"
    response *without calling the API* when nothing relevant is retrieved,
    plus a lexical-grounding check (`check_grounding`) on every generated
    answer against the retrieved context, plus the system prompt's explicit
    abstain instruction.
  - **Latency logging:** every query logs `retrieval_latency_s`,
    `generation_latency_s`, `total_latency_s`, chunks retrieved, top score,
    and status to `rag_latency_log.jsonl` (and stdout).
- `test_generate_answer.py` — 18 tests, all mocked (no API key or network
  needed), covering prompt building, grounding checks, and every error path
  (timeout, 429, 5xx, 4xx/context-length, malformed response) plus full
  pipeline behavior for in-domain, out-of-domain, abstain, and error cases.
- `.env.example` — copy to `.env` and add your own OpenRouter key.
- `sample_processed_dataset.csv` — a tiny demo dataset so `generate_answer.py`
  runs out of the box even before `processed_dataset.csv` (Task 1's real
  output) exists locally.
- `requirements_additions.txt` — `requests` and `python-dotenv` to add to
  the existing `requirements.txt`.

**Setup:**
```bash
pip install -r requirements.txt          # after adding the two new lines
cp .env.example .env                     # then edit .env with your key from
                                          # https://openrouter.ai/keys
```

**Run:**
```bash
python generate_answer.py "how do spacecraft communicate with mission control over long distances"
# or interactively:
python generate_answer.py
```
Note: a query like "What happened to the Space Shuttle Columbia?" correctly
returns `status=abstained` — the 2003 Columbia disaster postdates the
1995-era 20 Newsgroups corpus, so there's genuinely no relevant context to
ground an answer in. That's the grounding/abstain logic working as intended,
not a bug.

**Test (no API key needed):**
```bash
python -m pytest test_generate_answer.py -v
```
Expected: `18 passed`.

**Model:** defaults to `deepseek/deepseek-chat` (OpenRouter's stable alias
for DeepSeek's current flagship chat model — solid quality, cheap, no dated
snapshot to maintain). Override with `OPENROUTER_MODEL` in `.env`, e.g.
`deepseek/deepseek-chat-v3.1:free` for $0 testing while building (free tier
is rate-limited, not for production use). Current models/pricing:
https://openrouter.ai/deepseek

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key libraries:

| Library | Purpose |
|---|---|
| pandas, numpy | Data handling |
| scikit-learn | Dataset loader (20 Newsgroups); TF-IDF fallback retriever |
| nltk | Tokenization |
| langdetect | Detecting mixed-language text |
| ftfy | Fixing encoding artifacts (mojibake) |
| chardet | Detecting raw byte encoding |
| emoji | Detecting/handling emoji in text |
| pytest | Unit testing |
| tqdm | Progress bars for large-loop operations |
| tiktoken | Token-aware chunking (used in a later task, not this one) |
| sentence-transformers | Embeddings (`all-MiniLM-L6-v2`) for Task 2 & Task 3 retrieval |
| chromadb | Persistent vector store (Task 2 ingestion, Task 3 retrieval) |
| requests | OpenRouter API calls (Task 3) |
| python-dotenv | Loading `.env` config (Task 3) |


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

**5. Chunk, embed, and ingest into ChromaDB (Task 2)**
```bash
python chunking.py
python embeddings.py
python chroma_ingest.py
python test_retrieval.py
```
Requires `processed_dataset.csv` from step 4.

**6. Run the RAG generation layer (Task 3)**
```bash
cp .env.example .env    # then add your OpenRouter key
python -m pytest test_generate_answer.py -v          # 18 passed, no key needed
python generate_answer.py "how do spacecraft communicate with mission control over long distances"
```
Requires `chroma_db/` from step 5 (falls back to TF-IDF over
`processed_dataset.csv` / `sample_processed_dataset.csv` if absent).

**All steps in one block** (for copy-paste convenience):
```bash
python verify_setup.py
python load_and_validate_dataset.py
python -m pytest test_text_cleaning.py -v
python generate_clean_dataset.py
python chunking.py
python embeddings.py
python chroma_ingest.py
python test_retrieval.py
cp .env.example .env    # then add your OpenRouter key
python -m pytest test_generate_answer.py -v
python generate_answer.py "how do spacecraft communicate with mission control over long distances"
```

---

## Repository Structure

Flat structure — everything in the repo root:

```
Parallax-Labs-AI-ML-Internship/
├── README.md
├── requirements.txt
├── requirements_additions.txt   # requests, python-dotenv (Task 3)
├── .gitignore
├── .env.example                 # copy to .env and add your OpenRouter key
├── verify_setup.py
├── load_and_validate_dataset.py
├── data_raw.csv                 # generated when you run load_and_validate_dataset.py
├── data_validation_summary.md   # generated when you run load_and_validate_dataset.py
├── text_cleaning.py
├── test_text_cleaning.py
├── generate_clean_dataset.py
├── data_quality_report.md       # generated when you run generate_clean_dataset.py
├── processed_dataset.csv        # generated when you run generate_clean_dataset.py (final output)
├── chunking.py
├── test_chunking.py
├── embeddings.py
├── chroma_ingest.py
├── test_retrieval.py
├── retriever.py                 # Task 3: ChromaRetriever + TfidfRetriever fallback
├── generate_answer.py           # Task 3: RAG generation via DeepSeek/OpenRouter
├── test_generate_answer.py      # Task 3: 18 mocked unit tests
├── sample_processed_dataset.csv # Task 3: tiny demo dataset
├── rag_latency_log.jsonl        # generated when you run generate_answer.py (gitignored)
├── NOTES.md
└── logs/                        # generated: embedding_perf.csv, retrieval_perf.csv
```
