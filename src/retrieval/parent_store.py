"""Parent-chunk store: the larger passages retrieval hands to the LLM.

Children (small, embedded) live in Qdrant; their parents (whole
sub-sections, never embedded) live here in a SQLite file, keyed by
(doc_id, parent_id) and looked up after a child search. Kept out of Qdrant
because parents need no vectors, and a plain key-value table keeps them
inspectable with any SQLite tool.

Every operation opens its own short-lived connection, so the store is safe
to share across threads (Streamlit runs each rerun on its own thread) and
holds no lock between calls.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)

ParentKey = tuple[str, int]  # (doc_id, parent_id)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parents (
    doc_id TEXT NOT NULL,
    parent_id INTEGER NOT NULL,
    page_content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    PRIMARY KEY (doc_id, parent_id)
)
"""


class ParentStore:
    """Parent Documents in a SQLite table, keyed by (doc_id, parent_id)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def add(self, parents: list[Document]) -> None:
        """Insert parents, replacing any existing row with the same (doc_id, parent_id)."""
        rows = [
            (doc.metadata["doc_id"], doc.metadata["parent_id"], doc.page_content, json.dumps(doc.metadata))
            for doc in parents
        ]
        with self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO parents VALUES (?, ?, ?, ?)", rows)
        logger.info("Stored %d parents", len(rows))

    def get(self, keys: list[ParentKey]) -> dict[ParentKey, Document]:
        """Look up parents by key; keys with no stored parent are simply absent from the result."""
        if not keys:
            return {}
        found: dict[ParentKey, Document] = {}
        with self._connect() as conn:
            for doc_id, parent_id in keys:
                row = conn.execute(
                    "SELECT page_content, metadata FROM parents WHERE doc_id = ? AND parent_id = ?",
                    (doc_id, parent_id),
                ).fetchone()
                if row is not None:
                    found[(doc_id, parent_id)] = Document(page_content=row[0], metadata=json.loads(row[1]))
        return found

    def delete_filing(self, doc_id: str) -> int:
        """Delete every parent of one filing; returns how many were deleted."""
        with self._connect() as conn:
            deleted = conn.execute("DELETE FROM parents WHERE doc_id = ?", (doc_id,)).rowcount
        logger.info("Deleted %d parents for doc_id=%s", deleted, doc_id)
        return deleted

    def reset(self) -> None:
        """Delete every parent."""
        with self._connect() as conn:
            conn.execute("DELETE FROM parents")

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success, rolls back on error, and is always closed."""
        conn = sqlite3.connect(self._path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()
