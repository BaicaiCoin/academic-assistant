"""
Runtime configuration for the Academic Agent system.

Uses LangGraph's RunnableConfig pattern so parameters can be overridden
per-request without changing code (useful for A/B testing models, etc.).

Usage:
    config = Configuration.from_runnable_config(runnable_config)
    model_name = config.brain_model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    brain_model: str = field(
        default="deepseek-chat",
        metadata={"description": "Model used by the Brain Agent (planner + executor)."},
    )
    chat_model: str = field(
        default="deepseek-chat",
        metadata={"description": "Model used by the Chat Agent for response rendering."},
    )
    embed_model: str = field(
        default="BAAI/bge-m3",
        metadata={"description": "Embedding model name for GraphRAG retrieval."},
    )

    # ------------------------------------------------------------------
    # Brain Agent limits
    # ------------------------------------------------------------------
    max_replan: int = field(
        default=3,
        metadata={"description": "Max times the planner can be called in one turn."},
    )
    max_executor_retries: int = field(
        default=2,
        metadata={"description": "Max retries for a single failed execution step."},
    )

    # ------------------------------------------------------------------
    # GraphRAG retrieval
    # ------------------------------------------------------------------
    retrieval_top_k: int = field(
        default=5,
        metadata={"description": "Number of top nodes to retrieve from Milvus."},
    )
    graph_hop: int = field(
        default=1,
        metadata={"description": "Number of hops to expand in Neo4j after retrieval."},
    )

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """
        Extract configurable fields from a LangGraph RunnableConfig.
        Falls back to defaults for any missing keys.

        Example override from API call:
            {"configurable": {"brain_model": "gemini-2.0-flash", "retrieval_top_k": 8}}
        """
        configurable = (config or {}).get("configurable", {})
        # Only pass keys that are actual fields on this dataclass
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in configurable.items() if k in known_fields}
        return cls(**filtered)
