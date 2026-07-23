"""Embedding model construction. Model instantiation only; no indexing."""

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import EMBEDDING_MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Construct the shared HuggingFace embedding model.

    Returns:
        A HuggingFaceEmbeddings instance (sentence-transformers backed,
        BAAI/bge-small-en-v1.5) with normalized embeddings, to be injected
        into vector_store/retriever functions rather than constructed there.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
