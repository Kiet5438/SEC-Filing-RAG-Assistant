"""Single shared FAISS vector store: build, load, save, update.

The embedding model is always injected by the caller. No metadata filtering
lives here — retriever.py owns that; this module only manages the index
itself, which holds chunks from every ingested filing distinguished purely
by metadata.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.utils.config import VECTORSTORE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILES = ("index.faiss", "index.pkl")


def build_vectorstore(documents: list[Document], embedding: Embeddings) -> FAISS:
    """Build a brand-new FAISS index from scratch.

    Args:
        documents: Chunks to index.
        embedding: Injected embedding model.

    Returns:
        A new FAISS vector store containing only the given documents.
    """
    logger.info("Building new FAISS index from %d documents", len(documents))
    return FAISS.from_documents(documents, embedding)


def load_vectorstore(embedding: Embeddings) -> FAISS | None:
    """Load the shared FAISS index from VECTORSTORE_DIR, if it exists.

    Only ever point this at a locally generated index: allow_dangerous_deserialization
    is enabled because this project's own ingestion pipeline is the sole
    writer of data/vectorstore/.

    Args:
        embedding: Injected embedding model (must match the one used to build the index).

    Returns:
        The loaded FAISS store, or None if no index is present yet, or if
        loading fails (missing/corrupt files) — logged, not raised.
    """
    if not all((VECTORSTORE_DIR / name).exists() for name in _INDEX_FILES):
        logger.info("No existing FAISS index found at %s", VECTORSTORE_DIR)
        return None

    try:
        vectorstore = FAISS.load_local(str(VECTORSTORE_DIR), embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error("Failed to load FAISS index from %s: %s", VECTORSTORE_DIR, e)
        return None

    logger.info("Loaded FAISS index from %s", VECTORSTORE_DIR)
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """Persist a FAISS store to VECTORSTORE_DIR via native save_local()."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info("Saved FAISS index to %s", VECTORSTORE_DIR)


def update_vectorstore(vectorstore: FAISS | None, documents: list[Document], embedding: Embeddings) -> FAISS:
    """Add documents to vectorstore, building it fresh first if it doesn't exist yet.

    Does not save; the caller is responsible for calling save_vectorstore
    afterward.

    Args:
        vectorstore: Existing store, or None to build a new one.
        documents: Chunks to add.
        embedding: Injected embedding model.

    Returns:
        The updated (or newly built) FAISS store.
    """
    if vectorstore is None:
        return build_vectorstore(documents, embedding)
    vectorstore.add_documents(documents)
    logger.info("Added %d documents to existing FAISS index", len(documents))
    return vectorstore


def indexed_doc_ids(vectorstore: FAISS | None) -> set[str]:
    """Collect the distinct doc_ids already present in a FAISS store.

    Args:
        vectorstore: The store to inspect, or None (returns an empty set).

    Returns:
        The set of doc_id values found across all stored chunk metadata.
    """
    if vectorstore is None:
        return set()
    return {
        doc_id for doc in vectorstore.docstore._dict.values() if (doc_id := doc.metadata.get("doc_id")) is not None
    }
