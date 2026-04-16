"""
Evaluation logger node.

Writes (query, contexts, answer, reranker_scores) to SQLite after each turn
so the offline RAGAS evaluation script can consume them later.

Non-fatal: any exception is silently swallowed so logging never breaks the
main conversation flow.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import aiosqlite

from agent.state import OverallState

_TABLE = "eval_log"


async def log_eval_sample(state: OverallState) -> dict:
    """Persist this conversation turn for offline evaluation. Always returns {}."""
    if not state.normalized_query or not state.final_answer:
        return {}

    db_path = os.environ.get("EVAL_LOG_PATH", "./data/eval_log.db")

    contexts = [c.content for c in state.retrieved_context]
    scores = [c.score for c in state.retrieved_context]
    resource_types = [c.resource_type for c in state.retrieved_context]

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    query           TEXT    NOT NULL,
                    contexts        TEXT    NOT NULL,
                    answer          TEXT    NOT NULL,
                    reranker_scores TEXT    NOT NULL,
                    resource_types  TEXT    NOT NULL
                )
            """)
            await db.execute(
                f"""INSERT INTO {_TABLE}
                    (timestamp, query, contexts, answer, reranker_scores, resource_types)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    state.normalized_query,
                    json.dumps(contexts, ensure_ascii=False),
                    state.final_answer,
                    json.dumps(scores),
                    json.dumps(resource_types),
                ),
            )
            await db.commit()
    except Exception:
        pass

    return {}
