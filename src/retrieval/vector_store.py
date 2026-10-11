"""The shared filing index: child chunks in Qdrant, their parents in SQLite.

The embedding models are always injected by the caller. Child chunks from
every ingested filing live in one Qdrant collection, distinguished purely
by payload metadata (stored flat, alongside page_content), and each child
carries two vectors: a dense embedding (semantic match) and a sparse BM25
vector (exact-term match, IDF applied by Qdrant). Their parent passages
live in a ParentStore. FilingStore keeps the two stores in step — a filing
is added, deleted or reset in both at once.

This module executes queries but never decides filter or grouping policy:
retriever.py builds the Qdrant filter (applied during the search itself,
i.e. pre-filtering) and decides how matched children map to parents.
"""

import uuid
import warnings
from collections.abc import Iterable
from typing import Literal, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from src.retrieval.parent_store import ParentKey, ParentStore
from src.utils.config import (
    EMBEDDING_BATCH_SIZE,
    HYBRID_PREFETCH_K,
    PARENT_STORE_PATH,
    QDRANT_COLLECTION,
    QDRANT_DIR,
    QDRANT_URL,
    RRF_K,
)
from src.utils.helpers import parse_doc_id
from src.utils.logger import get_logger

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
SearchMode = Literal["hybrid", "dense", "sparse"]

_PAGE_CONTENT_KEY = "page_content"
# Fixed namespace so a chunk's point id is stable across runs: re-adding the
# same (doc_id, chunk_id) overwrites the point instead of duplicating it.
_POINT_ID_NAMESPACE = uuid.UUID("6f1c2a52-8a0e-4d3b-9a77-3f4f1c0d9e21")
_FACET_LIMIT = 10_000
# Per-filing fields read from one stored child to describe a filing.
_SUMMARY_FIELDS = ("filing_name", "fiscal_year", "fiscal_period", "period_end_date")
# Payload fields given an index: the ones filters and facets use. (A Qdrant
# server requires the index for facet queries on a field.)
_PAYLOAD_INDEXES = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "ticker": models.PayloadSchemaType.KEYWORD,
    "filing_type": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "item_code": models.PayloadSchemaType.KEYWORD,
    "part": models.PayloadSchemaType.KEYWORD,
    "content_type": models.PayloadSchemaType.KEYWORD,
    "fiscal_period": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
}


class SparseVectorLike(Protocol):
    indices: Iterable[int]
    values: Iterable[float]


class SparseEncoder(Protocol):
    """What FilingStore needs from a sparse model (fastembed's SparseTextEmbedding fits)."""

    def embed(self, documents: list[str]) -> Iterable[SparseVectorLike]: ...

    def query_embed(self, query: str) -> Iterable[SparseVectorLike]: ...


class FilingStore:
    """Child chunks (dense + BM25 vectors) in Qdrant, parents in a ParentStore.

    The collection is created lazily on the first add, sized from the first
    embedded vector, so opening a store never needs the embedding dimension
    up front. Every read on a not-yet-created collection behaves as empty.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedding: Embeddings,
        sparse_embedding: SparseEncoder,
        parents: ParentStore,
        collection_name: str = QDRANT_COLLECTION,
    ) -> None:
        self._client = client
        self._embedding = embedding
        self._sparse_embedding = sparse_embedding
        self._parents = parents
        self._collection = collection_name

    def add_filing(self, parents: list[Document], children: list[Document]) -> None:
        """Store a filing's parents, then embed and upsert its children in batches.

        Parents go first so a child is never searchable before the passage it
        expands to exists. Existing (doc_id, parent_id) parents and
        (doc_id, chunk_id) children are overwritten, not duplicated. Does not
        touch the manifest; the caller records it once this returns.
        """
        self._parents.add(parents)
        for start in range(0, len(children), EMBEDDING_BATCH_SIZE):
            batch = children[start : start + EMBEDDING_BATCH_SIZE]
            texts = [doc.page_content for doc in batch]
            dense = self._embedding.embed_documents(texts)
            sparse = list(self._sparse_embedding.embed(texts))
            self._ensure_collection(len(dense[0]))
            points = [
                models.PointStruct(
                    id=_point_id(doc.metadata),
                    vector={DENSE_VECTOR_NAME: dense_vector, SPARSE_VECTOR_NAME: _to_sparse_vector(sparse_vector)},
                    payload=_to_payload(doc),
                )
                for doc, dense_vector, sparse_vector in zip(batch, dense, sparse)
            ]
            self._client.upsert(self._collection, points=points)
            logger.info("Upserted %d/%d children", min(start + EMBEDDING_BATCH_SIZE, len(children)), len(children))

    def get_parents(self, keys: list[ParentKey]) -> dict[ParentKey, Document]:
        """Parent passages by (doc_id, parent_id); missing keys are absent from the result."""
        return self._parents.get(keys)

    def search(
        self,
        query: str,
        k: int,
        query_filter: models.Filter | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[Document]:
        """Search the child chunks, restricted by query_filter during the search.

        Args:
            query: The query text (embedded here with the injected models).
            k: Number of child chunks to return.
            query_filter: Optional pre-built Qdrant filter (see retriever.py).
            mode: "hybrid" runs a dense and a BM25 search (HYBRID_PREFETCH_K
                each, both filtered) and fuses their rankings with Reciprocal
                Rank Fusion (k=RRF_K) inside Qdrant; "dense" or "sparse" runs
                just one of them (for comparison).

        Returns:
            Up to k Documents, most relevant first; [] if the collection
            doesn't exist yet or nothing matches the filter.
        """
        if not self._exists():
            return []

        if mode == "dense":
            response = self._client.query_points(
                self._collection, query=self._embedding.embed_query(query), using=DENSE_VECTOR_NAME,
                query_filter=query_filter, limit=k, with_payload=True,
            )
        elif mode == "sparse":
            response = self._client.query_points(
                self._collection, query=self._sparse_query(query), using=SPARSE_VECTOR_NAME,
                query_filter=query_filter, limit=k, with_payload=True,
            )
        else:
            prefetch_k = max(HYBRID_PREFETCH_K, k)
            response = self._client.query_points(
                self._collection,
                prefetch=[
                    models.Prefetch(query=self._embedding.embed_query(query), using=DENSE_VECTOR_NAME, filter=query_filter, limit=prefetch_k),
                    models.Prefetch(query=self._sparse_query(query), using=SPARSE_VECTOR_NAME, filter=query_filter, limit=prefetch_k),
                ],
                query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
                query_filter=query_filter,
                limit=k,
                with_payload=True,
            )
        return [_to_document(point.payload) for point in response.points]

    def delete_filing(self, doc_id: str) -> int:
        """Delete every child and parent of one filing; returns how many children were deleted."""
        self._parents.delete_filing(doc_id)
        if not self._exists():
            return 0
        selector = _doc_id_filter(doc_id)
        deleted = self._client.count(self._collection, count_filter=selector, exact=True).count
        if deleted:
            self._client.delete(self._collection, points_selector=models.FilterSelector(filter=selector))
        logger.info("Deleted %d children for doc_id=%s", deleted, doc_id)
        return deleted

    def reset(self) -> None:
        """Drop the whole collection and every parent; the next add_filing starts from empty."""
        self._parents.reset()
        if self._exists():
            self._client.delete_collection(self._collection)
            logger.info("Dropped collection %r", self._collection)

    def count(self) -> int:
        """Total number of child chunks across all filings."""
        if not self._exists():
            return 0
        return self._client.count(self._collection, exact=True).count

    def doc_ids(self) -> set[str]:
        """The distinct doc_ids currently present in the collection."""
        return set(self._facet_counts("doc_id"))

    def distinct_values(self, key: str) -> list[str]:
        """Sorted distinct values of one indexed payload field (e.g. "ticker")."""
        return sorted(str(value) for value in self._facet_counts(key))

    def filing_summaries(self) -> list[dict]:
        """One row per indexed filing, read from the collection itself.

        Each row: doc_id, ticker, filing_type, filing_date, filing_name,
        fiscal_year, fiscal_period, period_end_date and chunk_count
        (children). Sorted by ticker ascending, then filing_date descending
        within each ticker. The collection, not manifest.json, is the source
        of truth for what's actually retrievable.
        """
        rows = []
        for doc_id, count in self._facet_counts("doc_id").items():
            points, _ = self._client.scroll(
                self._collection, scroll_filter=_doc_id_filter(doc_id), limit=1, with_payload=list(_SUMMARY_FIELDS)
            )
            payload = points[0].payload if points else {}
            rows.append({"doc_id": doc_id, **parse_doc_id(doc_id), **{key: payload.get(key) for key in _SUMMARY_FIELDS}, "chunk_count": count})
        rows.sort(key=lambda row: row["filing_date"], reverse=True)
        rows.sort(key=lambda row: row["ticker"])
        return rows

    def close(self) -> None:
        """Release the client (and, in local mode, the storage-folder lock)."""
        self._client.close()

    def _sparse_query(self, query: str) -> models.SparseVector:
        return _to_sparse_vector(next(iter(self._sparse_embedding.query_embed(query))))

    def _exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def _facet_counts(self, key: str) -> dict:
        if not self._exists():
            return {}
        response = self._client.facet(self._collection, key=key, limit=_FACET_LIMIT, exact=True)
        return {hit.value: hit.count for hit in response.hits}

    def _ensure_collection(self, vector_size: int) -> None:
        if self._exists():
            return
        self._client.create_collection(
            self._collection,
            vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=vector_size, distance=models.Distance.COSINE)},
            # BM25: the sparse model supplies term frequencies; Qdrant applies
            # IDF from collection statistics at query time.
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Local mode ignores payload indexes (and warns); they matter once
        # QDRANT_URL points at a server, so they're always declared.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")
            for field_name, schema in _PAYLOAD_INDEXES.items():
                self._client.create_payload_index(self._collection, field_name, schema)
        logger.info("Created collection %r (dense size=%d, sparse=%s)", self._collection, vector_size, SPARSE_VECTOR_NAME)


def open_store(embedding: Embeddings, sparse_embedding: SparseEncoder) -> FilingStore:
    """Open the shared store: children in a Qdrant server if QDRANT_URL is set, else local mode at
    QDRANT_DIR; parents in the SQLite file at PARENT_STORE_PATH.

    Args:
        embedding: Injected dense embedding model (must match the one the
            collection was built with).
        sparse_embedding: Injected BM25 sparse model (likewise).

    Returns:
        A FilingStore. The caller owns it and should close() it when done,
        since local mode locks the storage folder for this client's lifetime.

    Raises:
        RuntimeError: Local mode and another client (another process, or an
            unclosed store in this one) already holds the storage folder.
    """
    parents = ParentStore(PARENT_STORE_PATH)
    if QDRANT_URL:
        logger.info("Connecting to Qdrant server at %s", QDRANT_URL)
        return FilingStore(QdrantClient(url=QDRANT_URL), embedding, sparse_embedding, parents)

    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        client = QdrantClient(path=str(QDRANT_DIR))
    except RuntimeError as e:
        raise RuntimeError(
            f"The Qdrant store at {QDRANT_DIR} is already open in another client — local mode allows one "
            "at a time. Close the Streamlit app (or the other script) first, or set QDRANT_URL to use a "
            f"Qdrant server. ({e})"
        ) from e
    logger.info("Opened local Qdrant store at %s", QDRANT_DIR)
    return FilingStore(client, embedding, sparse_embedding, parents)


def _point_id(metadata: dict) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{metadata['doc_id']}:{metadata['chunk_id']}"))


def _to_sparse_vector(embedding: SparseVectorLike) -> models.SparseVector:
    return models.SparseVector(indices=[int(i) for i in embedding.indices], values=[float(v) for v in embedding.values])


def _to_payload(document: Document) -> dict:
    return {**document.metadata, _PAGE_CONTENT_KEY: document.page_content}


def _to_document(payload: dict) -> Document:
    metadata = dict(payload)
    page_content = metadata.pop(_PAGE_CONTENT_KEY, "")
    return Document(page_content=page_content, metadata=metadata)


def _doc_id_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
