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
VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"
MANIFEST_PATH: Path = DATA_DIR / "manifest.json"
TICKERS_CACHE_PATH: Path = DATA_DIR / ".cache" / "company_tickers.json"

# --- Embedding ---
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

# --- Chunking ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# --- Retrieval ---
RETRIEVAL_K: int = 5
FETCH_K: int = 20
FILTER_POOL_MULTIPLIER: int = 5

# --- LLM (Gemini) ---
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_TEMPERATURE: float = 0.2

# --- SEC EDGAR ---
SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "")
SEC_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL: str = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE: str = "https://www.sec.gov/Archives/edgar/data"

# --- Section detection ---
# Section-detection regexes now live per-form in
# src/preprocessing/filing_profiles.py (the single source of truth); there
# is intentionally no SECTION_PATTERNS constant here anymore.

# --- Prompts ---
SYSTEM_PROMPT: str = (
    "You are a financial analyst. Answer ONLY using the retrieved context. "
    "If the answer is not contained in the context, say 'I cannot find the "
    "answer in the provided filing.' Always cite: source, section, chunk_id."
)
