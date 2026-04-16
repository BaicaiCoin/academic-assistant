"""
PersistentSqliteStore — SQLite-backed long-term memory store.

Architecture:
  - InMemoryStore  handles all in-memory state and vector similarity search.
  - SQLite         persists every write so memories survive server restarts.

On startup call await store.setup() to create the table and reload saved
memories back into the InMemoryStore for vector indexing.

All LangGraph store operations go through abatch(). We intercept that method
to write to SQLite, then delegate to the inner InMemoryStore.
The convenience methods (aput, asearch, aget, …) inherited from BaseStore
all funnel through abatch, so no extra overrides are needed.
"""

from __future__ import annotations

import json
from typing import Any

import aiosqlite
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

_TABLE = "long_term_memories"


class PersistentSqliteStore(BaseStore):
    """
    Drop-in BaseStore replacement with SQLite persistence.

    Args:
        db_path:  Path to the SQLite file (e.g. "./data/memories.db").
        embed_fn: Callable[[list[str]], list[list[float]]] — used by
                  InMemoryStore to build the vector index for asearch().
        dims:     Embedding dimension (1024 for BGE-M3).
    """

    def __init__(self, db_path: str, embed_fn: Any, dims: int = 1024) -> None:
        self._db_path = db_path
        self._inner = InMemoryStore(index={"embed": embed_fn, "dims": dims})

    async def setup(self) -> None:
        """
        Create the SQLite table (if absent) and reload all stored memories
        into the InMemoryStore so vector search is ready from the first request.
        Call once at application startup, before the first request.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    namespace TEXT NOT NULL,
                    key       TEXT NOT NULL,
                    value     TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
            """)
            await db.commit()

            async with db.execute(
                f"SELECT namespace, key, value FROM {_TABLE}"
            ) as cur:
                async for row in cur:
                    namespace = tuple(json.loads(row[0]))
                    key = row[1]
                    value = json.loads(row[2])
                    # Write directly into inner store to rebuild vector index
                    await self._inner.aput(namespace, key, value)

    # -------------------------------------------------------------------------
    # Core override: intercept writes for SQLite persistence
    # -------------------------------------------------------------------------

    def batch(self, ops):
        """Synchronous batch — not used in this async-only project."""
        raise NotImplementedError("Use abatch instead")

    async def abatch(self, ops):
        """
        All LangGraph store calls funnel through here.
        We persist write ops (PutOp) to SQLite, then delegate everything
        to the inner InMemoryStore which handles vector indexing and reads.
        """
        async with aiosqlite.connect(self._db_path) as db:
            for op in ops:
                if type(op).__name__ == "PutOp":
                    if op.value is not None:
                        await db.execute(
                            f"INSERT OR REPLACE INTO {_TABLE} VALUES (?, ?, ?)",
                            (
                                json.dumps(list(op.namespace)),
                                op.key,
                                json.dumps(op.value),
                            ),
                        )
                    else:
                        # value=None means delete
                        await db.execute(
                            f"DELETE FROM {_TABLE} WHERE namespace=? AND key=?",
                            (json.dumps(list(op.namespace)), op.key),
                        )
            await db.commit()

        return await self._inner.abatch(ops)
