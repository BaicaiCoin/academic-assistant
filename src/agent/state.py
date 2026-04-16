"""
Single shared state for the entire Academic Video + Paper Assistant graph.

All subgraphs (Chat Agent, Brain Agent) use OverallState directly.
Each subgraph only reads/writes the fields it owns — see field ownership
comments below for clarity.

Design rationale (方案 A):
  Using one shared state keeps the system transparent during development:
  Brain Agent internals (plan, scratchpad) are visible for debugging,
  and there are no type-mapping functions to maintain.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ---------------------------------------------------------------------------
# Supporting data models (not States themselves)
# ---------------------------------------------------------------------------

class ResourceMeta(BaseModel):
    """Metadata for a loaded video or paper resource."""
    resource_id: str
    title: str
    resource_type: Literal["video", "paper"]
    source_url: str


class RetrievedChunk(BaseModel):
    """
    A single context chunk to be shown to the user.

    Used for all tool result types:
      video_slide   — a PPT slide + subtitle from rag_retrieve
      paper_chunk   — a paper section from rag_retrieve
      github_repo   — a code repository from search_github_repo
      search_result — a web/paper search result from search_video / search_paper
    """
    node_id: str
    resource_id: str
    resource_type: Literal["video_slide", "paper_chunk", "github_repo", "search_result", "job_result"]
    content: str        # main text shown to user
    summary: str = ""  # optional short summary; empty for non-RAG results
    score: float = 0.0 # relevance score; 0.0 for non-RAG results
    # For video slides: timestamp range; for paper chunks: page number
    # For github_repo: repo url; for search_result: source url
    location: str | None = None


class ExecutionStep(BaseModel):
    """A single step in the Brain Agent's execution plan."""
    step_id: int
    description: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    status: Literal["pending", "running", "success", "failed"] = "pending"
    error_msg: str | None = None
    retry_count: int = 0


# ---------------------------------------------------------------------------
# The one and only State
# ---------------------------------------------------------------------------

class OverallState(BaseModel):
    """
    Shared state for the entire graph.

    Field ownership by subgraph:
    +--------------------------+------------------------------------------+
    | Chat Agent owns          | Brain Agent owns                         |
    +--------------------------+------------------------------------------+
    | messages                 | plan                                     |
    | raw_user_input           | current_step_index                       |
    | normalized_query (write) | replan_count                             |
    | final_answer (write)     | executor_scratchpad                      |
    |                          | retrieved_context (write)                |
    |                          | brain_done (write)                       |
    +--------------------------+------------------------------------------+
    | Shared (read by both)                                               |
    |   loaded_resources                                                  |
    |   normalized_query  (read by Brain)                                 |
    |   retrieved_context (read by Chat)                                  |
    +---------------------------------------------------------------------+
    """

    # ---- Conversation (Chat Agent) --------------------------------------
    # add_messages reducer: appends instead of overwriting
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    raw_user_input: str = ""

    # Written by Chat Agent, read by Brain Agent
    normalized_query: str = ""

    # Final fluent answer rendered by Chat Agent
    final_answer: str = ""

    # ---- Resources (shared) ---------------------------------------------
    loaded_resources: list[ResourceMeta] = Field(default_factory=list)

    # ---- Brain Agent internals ------------------------------------------
    plan: list[ExecutionStep] = Field(default_factory=list)
    current_step_index: int = 0
    replan_count: int = 0
    executor_scratchpad: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Brain -> Chat handoff ------------------------------------------
    # Written by Brain Agent, read by Chat Agent
    retrieved_context: list[RetrievedChunk] = Field(default_factory=list)
    brain_done: bool = False

    # ---- Long-term memory (cross-thread) --------------------------------
    # Formatted string of relevant past memories, injected by load_memory
    # and consumed by normalize_input.
    long_term_memory: str = ""

    # ---- Frontend progress display --------------------------------------
    # Written by each node to show current progress to the user via SSE
    status_message: str = ""