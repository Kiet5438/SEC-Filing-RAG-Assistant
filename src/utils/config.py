"""Project-wide configuration constants.

Loads environment variables via python-dotenv and exposes them alongside
static configuration (paths, chunking/retrieval parameters, SEC endpoints,
and prompt templates) as module-level constants. This module must not
contain any logic beyond constant definitions and .env loading.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

# --- Paths ---
BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
QDRANT_DIR: Path = DATA_DIR / "qdrant"
PARENT_STORE_PATH: Path = DATA_DIR / "parents.sqlite3"
MANIFEST_PATH: Path = DATA_DIR / "manifest.json"
TICKERS_CACHE_PATH: Path = DATA_DIR / ".cache" / "company_tickers.json"

# --- Embedding ---
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # e.g. "cuda" for a faster bulk rebuild
SPARSE_MODEL_NAME: str = "Qdrant/bm25"
SPARSE_MODEL_CACHE_DIR: Path = DATA_DIR / ".cache" / "fastembed"

# --- Vector store (Qdrant) ---
# Unset -> embedded local mode at QDRANT_DIR (one process at a time: the
# storage folder is locked while a client has it open). Set to e.g.
# http://localhost:6333 to use a Qdrant server instead.
QDRANT_URL: str | None = os.getenv("QDRANT_URL")
QDRANT_COLLECTION: str = "filings"
EMBEDDING_BATCH_SIZE: int = 64

# --- Chunking (parent-child; all sizes in characters) ---
# Parents: one sub-section (H3 block) each; what the LLM reads.
PARENT_MIN_CHARS: int = 400  # smaller sub-sections merge into the next one
PARENT_MAX_CHARS: int = 6000  # larger ones split at block boundaries
# Children: what gets embedded and searched.
CHILD_CHUNK_SIZE: int = 800
CHILD_CHUNK_OVERLAP: int = 120
TABLE_CHILD_MAX_CHARS: int = 1500  # longer tables split by rows, header repeated
TABLE_CAPTION_MAX_CHARS: int = 300  # short text right before a table travels with it

# --- Retrieval ---
RETRIEVAL_K: int = 5  # parents returned
CHILD_K: int = 20  # top children ranked, then grouped into parents
HYBRID_PREFETCH_K: int = 50  # candidates from each of the dense and BM25 searches before fusion
RRF_K: int = 60  # Reciprocal Rank Fusion constant: score = sum(1 / (RRF_K + rank))

# --- Re-ranking (cross-encoder over the CHILD_K fused candidates) ---
# Off by default: on CPU, BAAI/bge-reranker-v2-m3 cost ~12 s per retrieval
# for +0.014 MRR over hybrid search, and BAAI/bge-reranker-base made results
# worse (see docs/DESIGN_DECISIONS.md). Set e.g. RERANKER_MODEL_NAME=
# BAAI/bge-reranker-v2-m3 (ideally with RERANKER_DEVICE=cuda) to enable it.
RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "")
RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "cpu")
RERANKER_MAX_LENGTH: int = 512  # tokens per (query, chunk) pair; a child chunk is ~150-400
RERANKER_BATCH_SIZE: int = 16

# --- LLM (Gemini API) ---
# Two model roles: "main" (planner, synthesizer: the judgment and the
# writing) and "lite" (executor: turning a concrete plan step into a tool
# call). Each role is a priority-ordered chain (comma-separated in .env):
# the first model serves, and a call it can't serve (rate limit or used-up
# daily quota, server error, timeout) moves to the next. The free tier's
# quota is per model, so each model in the chain adds its own allowance.
# Measured 2026-10-11, per call: 3.8-flash ~9 s at thinking "low",
# 3.7-flash 6-8 s (now and then a 503 "high demand", which moves the call
# on), 3.6-flash 2-5 s, 3.5-flash-lite 1-2 s, 3.1-flash-lite 6-8 s. The 2.5
# models answer 404 ("no longer available to new users"), and a 404 stops
# the chain, so keep retired ids out of it.
GEMINI_MODELS: list[str] = [  # main
    model.strip()
    for model in os.getenv(
        "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite"
    ).split(",")
    if model.strip()
]
GEMINI_LITE_MODELS: list[str] = [  # lite
    model.strip() for model in os.getenv("GEMINI_LITE_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if model.strip()
]
# Thinking depth for the main chain's Gemini 3 models ("" = the model's
# default). At the default, gemini-3.8-flash took ~34 s per call; at "low", ~9 s.
GEMINI_THINKING_LEVEL: str = os.getenv("GEMINI_THINKING_LEVEL", "low")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
LLM_TEMPERATURE: float = 0.2  # only for non-Gemini-3 models in a chain (see src/generation/llm.py)
LLM_TIMEOUT_SECONDS: float = 120  # per request; a model that times out hands the call to the next

# --- Agent ---
AGENT_MAX_STEPS: int = 4  # executor steps per question; then the synthesizer answers with what it has
AGENT_MAX_TOOL_CALLS_PER_STEP: int = 3  # parallel tool calls the executor may make for one step
AGENT_MAX_EVIDENCE: int = 20  # passages handed to the synthesizer, in the order they were found

# --- SEC EDGAR ---
SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "")
SEC_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL: str = "https://data.sec.gov/submissions/CIK{cik}.json"
# Older filing-history pages, named in a submissions response's filings.files list.
SEC_SUBMISSIONS_PAGE_URL: str = "https://data.sec.gov/submissions/{name}"
SEC_ARCHIVES_BASE: str = "https://www.sec.gov/Archives/edgar/data"

# --- Section detection ---
# Section-detection regexes now live per-form in
# src/preprocessing/filing_profiles.py (the single source of truth); there
# is intentionally no SECTION_PATTERNS constant here anymore.

# --- Prompts ---
SYSTEM_PROMPT: str = (
    "You are a financial analyst. Answer ONLY using the retrieved context. "
    "If the answer is not contained in the context, say 'I cannot find the "
    "answer in the provided filing.' Cite the sources you use by their SOURCE "
    "number in square brackets, e.g. [2]; never invent other identifiers."
)
