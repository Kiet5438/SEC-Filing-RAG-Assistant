"""Parent-child retrieval over the shared filing index.

Three stages: search the small, precise child chunks (hybrid dense + BM25
by default, with optional metadata pre-filtering); optionally re-rank those
candidates with a cross-encoder; then return their parent passages, which
carry enough surrounding context for the LLM to answer from. Filter policy
(which keys, exact-match semantics) and child-to-parent grouping both live
here; vector_store.py only executes the filter it is handed and looks
parents up.
"""

from langchain_core.documents import Document
from qdrant_client import models

from src.retrieval.parent_store import ParentKey
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import FilingStore, SearchMode
from src.utils.config import CHILD_K, RETRIEVAL_K
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """Child search, parent return.

    Filters are pushed down into the Qdrant query, so only matching children
    are ever ranked. The top child_k children are re-ordered by the reranker
    (if one is injected), then grouped by parent in rank order — a parent
    ranks where its best child ranked — and the first k distinct parents are
    returned. Grouping is also what keeps the result
    diverse: near-duplicate neighboring children collapse into one parent,
    which is the job MMR did when chunks were returned directly (and MMR,
    measured on top of grouping, only cost recall).
    """

    def __init__(
        self,
        store: FilingStore,
        k: int,
        child_k: int,
        mode: SearchMode = "hybrid",
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._k = k
        self._child_k = child_k
        self._mode = mode
        self._reranker = reranker

    def retrieve(self, question: str, filters: dict | None = None) -> list[Document]:
        """Retrieve up to k parent passages for question, optionally metadata-filtered.

        Args:
            question: The query text.
            filters: Optional metadata constraints; a chunk must match every
                key/value exactly to be considered. None or {} skips
                filtering.

        Returns:
            Up to k parent Documents, most relevant first, each with an extra
            "matched_chunk_ids" metadata list naming the children that
            matched, best first. Fewer than k if the matching children span
            fewer than k parents.
        """
        children = self._store.search(question, k=self._child_k, query_filter=_build_filter(filters), mode=self._mode)
        if self._reranker is not None:
            children = self._reranker.rerank(question, children)
        parents = self._expand_to_parents(children)
        if filters and len(parents) < self._k:
            logger.info("Metadata-filtered retrieval found only %d/%d passages for filters=%s", len(parents), self._k, filters)
        return parents

    def _expand_to_parents(self, children: list[Document]) -> list[Document]:
        """Group ranked children by parent and swap in the parent passages, keeping rank order."""
        matched: dict[ParentKey, list[int]] = {}
        best_child: dict[ParentKey, Document] = {}
        for child in children:
            key = (child.metadata["doc_id"], child.metadata["parent_id"])
            if key not in matched:
                if len(matched) == self._k:
                    continue
                matched[key] = []
                best_child[key] = child
            matched[key].append(child.metadata["chunk_id"])

        stored = self._store.get_parents(list(matched))
        passages = []
        for key, chunk_ids in matched.items():
            parent = stored.get(key)
            if parent is None:
                # The index and parent store are out of step (e.g. a hand-deleted
                # SQLite file): fall back to the child rather than lose the hit.
                logger.warning("No parent stored for %s; returning its best-matching child instead.", key)
                parent = best_child[key]
            passages.append(Document(page_content=parent.page_content, metadata={**parent.metadata, "matched_chunk_ids": chunk_ids}))
        return passages


def create_retriever(
    store: FilingStore,
    k: int = RETRIEVAL_K,
    child_k: int = CHILD_K,
    mode: SearchMode = "hybrid",
    reranker: Reranker | None = None,
) -> Retriever:
    """Build a Retriever over store.

    Args:
        store: Injected shared FilingStore.
        k: Number of parent passages to return.
        child_k: Number of top children ranked before grouping by parent;
            raised to at least k so k distinct parents are reachable.
        mode: Child search mode, "hybrid" (dense + BM25 fused by RRF),
            "dense" or "sparse"; the latter two exist for comparison.
        reranker: Injected cross-encoder to re-order the child_k candidates
            before grouping, or None to keep the search order.

    Returns:
        A Retriever exposing .retrieve(question, filters=None).
    """
    return Retriever(store, k, max(child_k, k), mode, reranker)


def _build_filter(filters: dict | None) -> models.Filter | None:
    """Translate {key: value} exact-match constraints into a Qdrant filter (all must match)."""
    if not filters:
        return None
    return models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value)) for key, value in filters.items()]
    )
