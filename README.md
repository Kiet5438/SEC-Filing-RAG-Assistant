# Financial Report Intelligence Assistant

A retrieval-augmented Q&A assistant over SEC EDGAR filings (10-K/10-Q):
ingestion → HTML parsing (sections, sub-headings, tables, fiscal period) →
parent-child chunking → dense + BM25 embedding → a single shared Qdrant
collection → hybrid search (dense + BM25 fused with RRF) over small child
chunks with optional metadata pre-filtering, returning their larger parent
passages (optional cross-encoder re-ranking) → a LangGraph agent that
plans, searches (and, with your approval, fetches and indexes
filings it lacks) and answers with numbered citations, all served through a
Streamlit chat UI.

> **Status:** Implemented and tested end to end against live SEC EDGAR
> filings and the Gemini API — ingestion, HTML parsing, chunking, embedding,
> the shared vector store, parent-child retrieval with metadata filtering, the
> agent (`src/agent/`), the `main.py` indexing CLI, and the `app/streamlit_app.py`
> chat UI (streamed answers + filters + indexing with approval) are all in place. See
> [Known limitations](#known-limitations) below.

## Setup

1. Create and activate a virtual environment with **Python 3.13**.
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
   cp .env.example .env        # Windows: copy .env.example .env
   ```

## Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key for Google Gemini (`langchain_google_genai.ChatGoogleGenerativeAI`). The LLMs are built in `src/generation/llm.py:get_llm`; everything else depends only on LangChain's chat-model interface. |
| `GEMINI_MODELS` | Optional. The **main** role's (planner, answer) **fallback chain**: comma-separated Gemini API model ids, in order. The first serves; a call it can't serve (rate limit or used-up daily quota, server error, timeout, lost connection) is retried on the next. The free tier's quota is **per model**, so every model adds its own daily allowance. Default `gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite`. Free-tier models get retired for new keys without notice (the 2.5 models already answer 404 "no longer available to new users"); an id that 404s stops the chain with an error rather than being skipped, so check `GET https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY` — and a test call, since retired models still appear there — when you change the list. |
| `GEMINI_LITE_MODELS` | Optional. The **lite** role's (executor) chain, same format; default `gemini-3.5-flash-lite,gemini-3.1-flash-lite`. |
| `GEMINI_THINKING_LEVEL` | Optional. Thinking depth of the main chain's Gemini 3 models, default `low` (`gemini-3.8-flash` measured ~34 s per call at its default, ~9 s at `low`); empty for the model's default. |
| `QDRANT_URL` | Optional. URL of a Qdrant server (e.g. `http://localhost:6333`). Unset (the default) runs Qdrant in embedded local mode at `data/qdrant/`. |
| `EMBEDDING_DEVICE` | Optional. Device for the dense embedding model, default `cpu` (e.g. `cuda` to speed up a bulk `--force-rebuild`). |
| `RERANKER_MODEL_NAME` | Optional. A cross-encoder (e.g. `BAAI/bge-reranker-v2-m3`) to re-rank retrieved chunks. Off by default: on CPU it measured ~12 s per retrieval for a small gain (see `docs/DESIGN_DECISIONS.md`). |
| `RERANKER_DEVICE` | Optional. Device for the reranker, default `cpu`. |
| `SEC_USER_AGENT` | Required by SEC EDGAR for all requests, format: `Your Name your.email@example.com`. Requests without it are rejected with HTTP 403. |

## Usage

### Indexing (CLI)

Index a filing into the shared vector store:

```
python main.py --ticker NVDA --filing 10-K
```

Running it again with the same `--ticker`/`--filing` is a no-op: it logs
`"already indexed; skipping"` and exits without re-downloading, re-parsing,
or re-embedding anything.

Index a specific (older) filing by its filing date instead of the latest:

```
python main.py --ticker NVDA --filing 10-K --filing-date 2025-02-26
```

Rebuild the entire store from `data/processed/*.md` (recovering per-file
metadata, notably `url` and `filing_name`, from `data/manifest.json`, with
no SEC network calls):

```
python main.py --force-rebuild
```

Remove one filing (its chunks and its manifest entry; the cached raw HTML
and Markdown stay on disk, so re-indexing it later doesn't re-download):

```
python main.py --remove NVDA_10-K_2026-02-25
```

The vector store runs in Qdrant's embedded **local mode** at `data/qdrant/`,
which locks that folder while a client has it open: run one of the app, the
CLI, the eval script, or the notebook at a time (the error message says so
if you don't). To lift that, run a Qdrant server and set `QDRANT_URL`
(e.g. `http://localhost:6333`) in `.env` — no code changes needed.

**Always run from the repository root** (`python main.py ...`, not from
inside `src/` or elsewhere) — every internal import is absolute
(`from src.utils.config import ...`), and running from the repo root is
what puts it on `sys.path` automatically. No `PYTHONPATH` changes are
needed or should be made.

### Querying (Streamlit app)

```
streamlit run app/streamlit_app.py
```

(`.streamlit/config.toml` turns off Streamlit's hot reload, whose file
watcher trips over `torch` internals; restart the app after code changes.)

Opens a chat UI over the shared index, answered by the agent: ask in
English or Vietnamese, optionally filter by ticker/filing type in the
sidebar (the filters bind every search), watch its progress in the status
box (including which model each call is waiting on, and why one was
skipped) while the answer streams in, and expand any citation to see the exact
passage it came from — or "How this was answered" to see each step (plan,
tool calls, results, review) with the model that served it. The chat is one
conversation: a follow-up such as "and Microsoft?" or "còn năm trước thì
sao?" is read in the context of earlier turns; "Clear conversation" starts a
new one. Ask about a company or period that isn't indexed and
the agent finds the filing on SEC EDGAR and asks before downloading and
indexing it. The sidebar's "Index a new filing" form is the direct
alternative — it calls the exact same `src/ingestion/pipeline.py:index_filing`
code path on the app's already-open store, then reruns so the new filing is
immediately queryable, without losing the current chat history.

How the agent answers (`src/agent/sidekick.py` wires the parts together):

- **memory**: each conversation is one LangGraph thread. A turn starts by
  clearing the previous turn's working state (plan, steps, passages) and
  ends by recording its question and answer; on a follow-up, the planner
  reads the last three turns and rewrites the question to stand alone.
- **planner** (main model): plans the fewest lookups that answer the
  question — one search for a simple question, one step per company or
  period for a comparison, list → index → search for a filing that isn't
  indexed — written in English whatever language you asked in. After every
  step it reviews the results: done, or the steps still to run, including
  new ones for parts still missing (e.g. a figure, searched in tables).
- **executor** (lite model): turns a step into tool calls
  (`search_filings` with its filters, `list_available_filings`,
  `list_sec_filings`, `index_filing`); `index_filing` pauses for your
  approval. At most `AGENT_MAX_STEPS` (4) steps per question.
- **synthesizer** (main model): writes the answer in the question's
  language from the passages found, numbered `[n]` to match the citations,
  says which parts the filings didn't cover, and reports a figure only under
  the label its filing gives it.

Or use the agent directly in Python, without the UI:

```python
from langgraph.types import Command

from src.agent.sidekick import build_sidekick, new_thread
from src.embeddings.embedder import get_embedding_model, get_sparse_embedding_model
from src.generation.llm import get_llm
from src.retrieval.retriever import create_retriever
from src.retrieval.vector_store import open_store

store = open_store(get_embedding_model(), get_sparse_embedding_model())
agent = build_sidekick(store, create_retriever(store), get_llm("main"), get_llm("lite"))
config = new_thread()
result = agent.invoke({"question": "What drove NVIDIA revenue growth?"}, config)
if agent.get_state(config).interrupts:  # it wants to index a filing
    result = agent.invoke(Command(resume={"approved": False}), config)
print(result["answer"], result["citations"])
follow_up = agent.invoke({"question": "And Microsoft?"}, config)  # same config = same conversation
store.close()  # releases the local-mode folder lock
```

The single-shot LCEL chain (`src/generation/rag_chain.py`: one retrieval,
one answer) is kept as a baseline.

## Known limitations

- Only 10-K and 10-Q have real section-detection profiles; other forms (e.g.
  8-K) fall back to coarse heading-only detection and aren't offered as
  options in the indexing UI.
- Conversations live in memory: the agent's checkpointer is in-process, so
  conversations (and an approval waiting for an answer) are lost when the
  app restarts, and the planner reads only the last three turns.
- Free-tier LLM quotas are small (`gemini-3.8-flash`: 20 requests/day; a
  simple question uses 3 main-model calls — plan, review, answer — and a long one up to 5: a plan, 3 reviews
  (none after the last allowed step), the answer; the executor's calls go to the lite model), hence the
  per-role Gemini model chains. Calls can move down a chain mid-question
  (per-minute limits), so one question's steps may be served by models of
  different strength.
- Single-process access in local mode: Qdrant's embedded mode locks
  `data/qdrant/`, so a second process (e.g. the CLI while the app is
  running) fails fast with a clear error instead of corrupting the store.
  Set `QDRANT_URL` to a Qdrant server for concurrent access.

## Notes

- `notebooks/workflow_walkthrough.ipynb` walks through the four flows —
  indexing (into a throwaway in-memory store), retrieval, the agent
  (including a follow-up turn and a declined indexing request) and
  evaluation — running the real functions. Close the app first: it reads the
  real store.

- `.env` and generated artifacts (`data/raw/`, `data/processed/`,
  `data/qdrant/`, `data/parents.sqlite3`, `data/.cache/`, `data/manifest.json`)
  are git-ignored; only
  `.env.example` is committed.

## Project layout

```
app/                   Streamlit chat UI (agent chat + indexing)
data/raw/              Downloaded filing HTML (created at runtime)
data/processed/        Parsed Markdown per filing (created at runtime)
data/qdrant/           Shared Qdrant collection of child chunks, embedded local mode (created at runtime)
data/parents.sqlite3   Parent passages, keyed by (doc_id, parent_id) (created at runtime)
data/manifest.json     doc_id -> ingestion metadata (written at ingestion time)
data/.cache/           Cached SEC ticker -> CIK map (created at runtime)
src/ingestion/         SEC filing discovery + download, indexing pipeline
src/preprocessing/     HTML -> Markdown parsing, Markdown -> chunks
src/embeddings/        Embedding model construction
src/retrieval/         Qdrant vector store, SQLite parent store, parent-child retriever
src/generation/        Chat-model factory (roles, Gemini fallback chains), citations, single-shot LCEL chain
src/utils/             Config, logging, shared helpers
src/agent/             The agent: memory, planner, executor, synthesizer, wired in sidekick.py; its tools and state
notebooks/             End-to-end walkthrough of indexing, retrieval, the agent and evaluation
main.py                Indexing CLI (pipeline in src/ingestion/pipeline.py)
```
