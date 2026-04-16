"""
Long-term memory nodes for the Chat Agent.

load_memory  — runs before normalize_input.
               Searches the Store with the current query and injects relevant
               past memories into OverallState.long_term_memory.

save_memory  — runs after render_response.
               Uses the LLM to extract 0-3 memory items from the current
               conversation turn and writes them to the Store.

Storage layout:
  namespace : ("user", USER_ID, "memories")
  key       : UUID string
  value     : {"content": str, "type": "preference|fact", "created_at": str}

Single-user system — all threads share USER_ID = "default_user".
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from agent.state import OverallState
from agent.configuration import Configuration
from agent.prompts import MEMORY_EXTRACT_PROMPT
from agent.utils import get_llm

USER_ID = "default_user"
_NAMESPACE = ("user", USER_ID, "memories")
_SEARCH_LIMIT = 5


async def load_memory(state: OverallState, store: BaseStore) -> dict:
    """
    Retrieve memories relevant to the current user input and inject them
    into long_term_memory for use by normalize_input.
    """
    if not state.raw_user_input:
        return {"long_term_memory": ""}

    results = await store.asearch(_NAMESPACE, query=state.raw_user_input, limit=_SEARCH_LIMIT)

    if not results:
        return {"long_term_memory": ""}

    lines = [f"- {r.value['content']}" for r in results]
    return {"long_term_memory": "\n".join(lines)}


async def save_memory(state: OverallState, config: RunnableConfig, store: BaseStore) -> dict:
    """
    Extract memorable facts/preferences from the current turn and persist
    them to the Store for future conversations.
    """
    if not state.normalized_query or not state.final_answer:
        return {}

    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.chat_model)

    prompt = MEMORY_EXTRACT_PROMPT.format(
        query=state.normalized_query,
        answer=state.final_answer,
    )
    response = await llm.ainvoke(prompt)

    try:
        raw = response.content.strip()
        # Strip markdown code fences if the model wraps its output
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        memories: list[dict] = json.loads(raw)
    except Exception:
        return {}

    now = datetime.now().isoformat()
    for mem in memories:
        content = mem.get("content", "").strip()
        if not content:
            continue
        await store.aput(
            _NAMESPACE,
            str(uuid.uuid4()),
            {
                "content": content,
                "type": mem.get("type", "fact"),
                "created_at": now,
            },
        )

    return {}
