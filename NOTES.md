# Notes — Task 2: Chunking, Embeddings & Vector DB

## Chunking Strategy
- Method: Recursive character text splitting (custom implementation in
  `chunking.py`, same core approach as LangChain's
  RecursiveCharacterTextSplitter — no extra dependency).
- Separators tried in order: paragraph breaks (`\n\n`) -> line breaks (`\n`)
  -> sentence breaks (`. `) -> word breaks (` `) -> hard character split.
- Chunk size: 1000 characters
- Chunk overlap: 100 characters (~10%)
- These are the defaults defined in `chunking.py`
  (`DEFAULT_CHUNK_SIZE = 1000`, `DEFAULT_CHUNK_OVERLAP = 100`), chosen as a
  reasonable balance for 20 Newsgroups-style posts (short-to-medium length,
  mostly plain text).
- Total chunks produced: 35,072 (from processed_dataset.csv)

## Embedding Performance
- Model: `all-MiniLM-L6-v2` (sentence-transformers, 384-dim vectors)
- Source: `logs/embedding_perf.csv`
- Total chunks embedded: 35,072
- Total embedding time: 1618.80 sec (~27.0 minutes)
- Average time per chunk: ~0.0462 sec (~46.2 ms/chunk)

## Retrieval Performance
- Source: `logs/retrieval_perf.csv`
- Test queries: 5 domain-relevant queries covering spacecraft/space,
  epilepsy/medical, password security, encryption, and writing tips —
  chosen to reflect real topics actually present in the 20 Newsgroups
  dataset, replacing the earlier generic/synthetic test queries.
- Latency on the 5 domain queries: min 18.07 ms, max 126.37 ms
  (first query includes one-time model warm-up), avg ~40.4 ms
- Example query -> top result:
  - Query: "how do spacecraft communicate with mission control over long
    distances"
    Top result (distance 1.0878): a post about spacecraft power systems
    (photovoltaic cells and small nuclear generators) for solar system
    missions.
  - Query: "what causes epileptic seizures and how are they treated"
    Top result (distance 0.8210): a post gathering information/discussion
    on the condition, posted to a medical newsgroup — this query returned
    the tightest (lowest-distance) matches of the five, suggesting strong
    topical clustering for epilepsy-related posts in the dataset.
  - Query: "best practices for creating a strong password"
    Top result (distance 1.0720): practical security advice — periodic
    password changes, always logging out, not leaving machines unattended.

## Known Limitations
- **Garbled/binary text in source data**: A subset of the 20 Newsgroups
  documents contain leftover binary/uuencoded content (e.g. encoded
  attachments) that survived the Task 1 cleaning pipeline. These produce
  chunks that embed and get stored as valid-looking vectors but are
  semantically meaningless. Confirmed directly during early retrieval
  testing — an earlier generic test query ("short query") returned several
  top-5 results consisting entirely of encoded garbage, e.g.:
  `'AX>'AX>'AX>'AX>'AX>'AX>'AX>'AS$QQ,1F"PN'AX>...`
  This is a known data quality issue carried over from Task 1; not filtered
  out in this task. A future fix would add a printable-character-ratio
  filter before chunking to drop these documents.
- ChromaDB edge cases handled: empty/whitespace-only documents are skipped
  before chunking (`chunk_dataframe` in `chunking.py` produces zero chunks
  for them rather than crashing); chunk IDs are deterministic and unique
  (`{doc_id}_{chunk_index}`), avoiding collisions on ingest.