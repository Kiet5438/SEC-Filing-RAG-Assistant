"""Cross-encoder re-ranking of retrieved child chunks.

First-stage search (dense, BM25, fused) scores the query and each chunk
independently — fast, but it never reads them together. A cross-encoder
reads each (query, chunk) pair jointly through one transformer and scores
how well that chunk answers that query: far more precise, far too slow to
run over the whole index, so it only re-orders the first stage's candidates.
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.utils.config import RERANKER_BATCH_SIZE, RERANKER_DEVICE, RERANKER_MAX_LENGTH, RERANKER_MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """Re-orders candidate chunks by a cross-encoder's (query, chunk) relevance score."""

    def __init__(self, model: CrossEncoder, batch_size: int = RERANKER_BATCH_SIZE) -> None:
        self._model = model
        self._batch_size = batch_size

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Return documents sorted by relevance to query, most relevant first.

        Every document is kept (truncation is the caller's call); ties keep
        their incoming order. Each chunk is scored on its full page_content,
        contextual header included, so the model sees which company, filing
        and section it comes from.
        """
        if not documents:
            return []
        scores = self._model.predict(
            [(query, doc.page_content) for doc in documents],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        order = sorted(range(len(documents)), key=lambda i: -float(scores[i]))
        return [documents[i] for i in order]


def get_reranker(model_name: str = RERANKER_MODEL_NAME) -> Reranker | None:
    """Construct the shared cross-encoder reranker, to be injected into create_retriever.

    Args:
        model_name: A sentence-transformers CrossEncoder model id; empty or
            "none" means re-ranking is off (the default, see config.py).

    Returns:
        A Reranker running on RERANKER_DEVICE, or None when disabled.
    """
    if not model_name or model_name.lower() == "none":
        logger.info("Re-ranking disabled (RERANKER_MODEL_NAME is unset).")
        return None
    logger.info("Loading reranker: %s (device=%s)", model_name, RERANKER_DEVICE)
    return Reranker(CrossEncoder(model_name, max_length=RERANKER_MAX_LENGTH, device=RERANKER_DEVICE))
