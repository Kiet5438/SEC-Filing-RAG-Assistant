# Financial Report Intelligence Assistant

A retrieval-augmented assistant over SEC EDGAR filings (10-K/10-Q/8-K):
ingestion → HTML parsing → chunking → embedding → a single shared FAISS
index → MMR retrieval with optional metadata filtering → an LCEL RAG chain
with citations.

> **Status:** Phase 1 complete — ingestion, HTML parsing, chunking, embedding,
> the shared FAISS store, MMR retrieval with metadata filtering, the LCEL RAG
> chain, and the `main.py` indexing CLI are all implemented and tested end to
> end against live SEC EDGAR filings and the Gemini API.
> `app/streamlit_app.py`, `src/evaluation/metrics.py`, and everything under
> `tests/` and `notebooks/` are docstring-only stubs, deferred to Phase 2.

## Setup

1. Create and activate a virtual environment with **Python 3.13**.
   (The spec targets 3.11; 3.11 wasn't available in this environment and
   pins were resolved/verified against 3.13 instead — code uses no
   3.12+-only syntax, so 3.11 remains a valid target if you have it.)
   ```
   py -3.13 -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy the environment template and fill in your values:
   ```
   cp .env.example .env
   ```

## Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key for Google Gemini (`langchain_google_genai.ChatGoogleGenerativeAI`). |
| `GEMINI_MODEL` | Gemini model id. Defaults to `gemini-flash-latest` (Google's rolling alias for the current-generation flash model — pin to a specific version, e.g. `gemini-3.5-flash`, if you want fully stable behavior instead). Note that on the free tier, older pinned versions (e.g. `gemini-2.5-flash`) can be retired for new API keys without notice; if a model 404s or its quota is exhausted, try `gemini-flash-latest` or another model from `GET https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY`. |
| `SEC_USER_AGENT` | Required by SEC EDGAR for all requests, format: `Your Name your.email@example.com`. Requests without it are rejected with HTTP 403. |

## Usage

Index a filing into the shared vector store:

```
python main.py --ticker NVDA --filing 10-K
```

Running it again with the same `--ticker`/`--filing` is a no-op: it logs
`"already indexed; skipping"` and exits without re-downloading, re-parsing,
or re-embedding anything.

Rebuild the entire index from `data/processed/*.md` (recovering per-file
metadata, notably `url` and `filing_name`, from `data/manifest.json`, with
no SEC network calls):

```
python main.py --ticker NVDA --filing 10-K --force-rebuild
```

**Always run from the repository root** (`python main.py ...`, not from
inside `src/` or elsewhere) — every internal import is absolute
(`from src.utils.config import ...`), and running from the repo root is
what puts it on `sys.path` automatically. No `PYTHONPATH` changes are
needed or should be made.

Querying (asking questions over an indexed filing) has no CLI yet — that's
the Streamlit app, deferred to Phase 2. Until then it's reachable directly
in Python:

```python
from src.embeddings.embedder import get_embedding_model
from src.retrieval.vector_store import load_vectorstore
from src.retrieval.retriever import create_retriever
from src.generation.rag_chain import get_llm, create_rag_chain, ask

embedding = get_embedding_model()
vs = load_vectorstore(embedding)
chain = create_rag_chain(create_retriever(vs), get_llm())
result = ask(chain, "What drove NVIDIA revenue growth?", filters={"ticker": "NVDA"})
print(result["answer"], result["citations"])
```

## Notes

- **FAISS deserialization:** `FAISS.load_local(...)` is called with
  `allow_dangerous_deserialization=True`. This is safe only because the
  index at `data/vectorstore/` is always generated locally by this project's
  own ingestion pipeline (`main.py`) — never point it at a FAISS index you
  did not generate yourself.
- `.env` and generated artifacts (`data/.cache/`, `*.faiss`, `*.pkl`) are
  git-ignored; only `.env.example` is committed.

## Project layout

```
app/                   Streamlit UI
data/raw/              Downloaded filing HTML
data/processed/        Parsed Markdown per filing
data/vectorstore/      Shared FAISS index (index.faiss, index.pkl)
data/manifest.json     doc_id -> ingestion metadata (written at ingestion time)
src/ingestion/         SEC filing discovery + download
src/preprocessing/     HTML -> Markdown parsing, Markdown -> chunks
src/embeddings/        Embedding model construction
src/retrieval/         FAISS vector store + retriever
src/generation/        LCEL RAG chain
src/evaluation/        Retrieval/generation metrics 
src/utils/             Config, logging, shared helpers
tests/                 Unit tests 
main.py                Indexing CLI
```
