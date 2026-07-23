"""MMR retrieval over the shared vector store, with optional in-memory
metadata filtering. All filter logic lives here; vector_store.py never
filters.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.utils.config import FETCH_K, FILTER_POOL_MULTIPLIER, RETRIEVAL_K
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """MMR retriever over a single shared vector store, with optional metadata filtering.

    With no filters, delegates directly to a pre-built MMR VectorStoreRetriever
    (plain top-k). With filters, pulls an enlarged MMR candidate pool
    straight from the vector store, keeps only documents whose metadata
    matches every provided key/value, and returns the top-k survivors.
    """

    def __init__(self, vectorstore: VectorStore, k: int, fetch_k: int, pool_multiplier: int) -> None:
        self._vectorstore = vectorstore
        self._k = k
        self._fetch_k = fetch_k
        self._pool_multiplier = pool_multiplier
        self._base_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})

    def retrieve(self, question: str, filters: dict | None = None) -> list[Document]:
        """Retrieve up to k relevant documents for question, optionally metadata-filtered.

        Args:
            question: The query text.
            filters: Optional metadata constraints; a document must match
                every key/value to survive. None or {} skips filtering and
                returns the plain MMR top-k.

        Returns:
            Up to k Documents, most relevant first. Fewer than k if
            filtering leaves fewer candidates than requested.
        """
        if not filters:
            return self._base_retriever.invoke(question)

        pool_size = self._fetch_k * self._pool_multiplier
        candidates = self._vectorstore.max_marginal_relevance_search(question, k=pool_size, fetch_k=pool_size)
        matched = [doc for doc in candidates if _matches(doc.metadata, filters)]

        if len(matched) < self._k:
            logger.info(
                "Metadata-filtered retrieval found only %d/%d candidates (pool=%d) for filters=%s",
                len(matched),
                self._k,
                pool_size,
                filters,
            )
        return matched[: self._k]


def create_retriever(vectorstore: VectorStore, k: int = RETRIEVAL_K, fetch_k: int = FETCH_K) -> Retriever:
    """Build a Retriever over vectorstore.

    Args:
        vectorstore: Injected shared vector store.
        k: Number of documents to return.
        fetch_k: MMR candidate pool size for the unfiltered path; also the
            base unit for the enlarged pool (fetch_k * FILTER_POOL_MULTIPLIER)
            used when filters are supplied.

    Returns:
        A Retriever exposing .retrieve(question, filters=None).
    """
    return Retriever(vectorstore, k, fetch_k, FILTER_POOL_MULTIPLIER)


def _matches(metadata: dict, filters: dict) -> bool:
    """True if metadata contains every key/value pair in filters."""
    return all(metadata.get(key) == value for key, value in filters.items())
