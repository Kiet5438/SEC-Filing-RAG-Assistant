"""Embedding model construction. Model instantiation only; no indexing."""

from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME, SPARSE_MODEL_CACHE_DIR, SPARSE_MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Construct the shared dense embedding model.

    Returns:
        A HuggingFaceEmbeddings instance (sentence-transformers backed,
        BAAI/bge-m3) with normalized embeddings, to be injected
        into vector_store/retriever functions rather than constructed there.
    """
    logger.info("Loading embedding model: %s (device=%s)", EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_sparse_embedding_model() -> SparseTextEmbedding:
    """Construct the shared BM25 sparse model (fastembed, ONNX on CPU).

    BM25 needs no neural weights — the "model" is a tokenizer, stemmer and
    stopword list that turn text into term-frequency sparse vectors; Qdrant
    applies IDF at query time. Cached under data/.cache rather than
    fastembed's default temp folder, which the OS may clear.

    Returns:
        A SparseTextEmbedding, injected alongside the dense model.
    """
    logger.info("Loading sparse model: %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(SPARSE_MODEL_NAME, cache_dir=str(SPARSE_MODEL_CACHE_DIR))
