## Task 3 — DeepSeek/OpenRouter Generation (RAG completion)

**Status: complete.** Adds an LLM generation layer on top of the retrieval
pipeline, with prompt engineering, robust API error handling, hallucination/
out-of-domain guards, and end-to-end latency logging.

**Repo-scan note:** I checked `main` before starting this and it currently
only has the Task 1 (data-cleaning) files — no chunking/embedding script or
`chroma_db/` yet, even though `.gitignore` already anticipates one
(`chroma_ingest.py` → `chroma_db/`). So `retriever.py` supports both: a
`ChromaRetriever` that will pick up that store automatically once it exists,
and a zero-dependency `TfidfRetriever` fallback so the pipeline runs today.
See the note at the top of `retriever.py` for details.

**New files:**
- `retriever.py` — `ChromaRetriever` (matches the `chroma_ingest.py` /
  `chroma_db/` architecture implied by `.gitignore`) + `TfidfRetriever`
  fallback + `get_retriever()` factory that picks whichever is available.
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
```
pip install -r requirements.txt          # after adding the two new lines
cp .env.example .env                     # then edit .env with your key from
                                          # https://openrouter.ai/keys
```

**Run:**
```
python generate_answer.py "What happened to the Space Shuttle Columbia?"
# or interactively:
python generate_answer.py
```

**Test (no API key needed):**
```
python -m pytest test_generate_answer.py -v
```
Expected: `18 passed`.

**Model:** defaults to `deepseek/deepseek-chat` (OpenRouter's stable alias
for DeepSeek's current flagship chat model — solid quality, cheap, no dated
snapshot to maintain). Override with `OPENROUTER_MODEL` in `.env`, e.g.
`deepseek/deepseek-chat-v3.1:free` for $0 testing while building (free tier
is rate-limited, not for production use). Current models/pricing:
https://openrouter.ai/deepseek
