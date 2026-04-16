"""
Tool definitions and Pydantic schemas for structured LLM outputs.

Tools are registered here as plain async functions and wrapped with
@tool decorator for use in LangGraph nodes via function calling.

Each tool stub includes a docstring that becomes the tool description
passed to the LLM — write it carefully.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from agent.state import RetrievedChunk


# ===========================================================================
# Pydantic schemas for structured LLM outputs
# ===========================================================================

class ExecutionPlanStep(BaseModel):
    """Single step schema used when the Planner outputs structured JSON."""
    step_id: int
    description: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """Full plan output from the Planner LLM."""
    steps: list[ExecutionPlanStep]


class RefinedToolCall(BaseModel):
    """
    Executor LLM's dynamic tool call decision.
    The Executor sees the current step + scratchpad and outputs refined
    tool_args — potentially different from the Planner's original tool_args
    when prior step results need to be incorporated.
    Note: tool_name is NOT included — the Executor refines args only,
    it does not change which tool to call (that's the Planner's job).
    """
    tool_args: dict[str, Any] = Field(
        description="Arguments to pass to the tool, refined using prior step results."
    )
    reasoning: str = Field(
        description="One sentence explaining how prior results influenced these args."
    )

# ===========================================================================
# Tool stubs
# (Replace the body of each with real implementation as you build modules)
# ===========================================================================

@tool
async def rag_retrieve(
    query: str,
    resource_ids: list[str] | None = None,
    expand_graph: bool = True,
) -> list[dict]:
    """
    Retrieve relevant context from the GraphRAG system (Milvus + Neo4j).
 
    Args:
        query: The search query (will be embedded for vector search).
        resource_ids: Optional filter — accepted values are "paper" and/or "video".
                      If None or empty, searches both collections.
        expand_graph: Whether to expand to neighbouring nodes in Neo4j
                      (fetches prev/next chunks via NEXT relation, and annotates
                       cross-source CORRESPONDS_TO links).
 
    Returns:
        List of RetrievedChunk-compatible dicts with node_id, content, location, score.
    """
    from agent.rag import async_retrieve
 
    TOP_K = 10
 
    search_paper = True
    search_video = True
    if resource_ids:
        search_paper = "paper" in resource_ids
        search_video = "video" in resource_ids
 
    return await async_retrieve(
        query=query,
        top_k=TOP_K,
        expand_graph=expand_graph,
        search_paper=search_paper,
        search_video=search_video,
    )


@tool
async def graph_rag_retrieve(
    template: str,
    params: dict,
) -> list[dict]:
    """
    Retrieve context using a structured graph query on Neo4j (template-based).
    Use this instead of rag_retrieve when the query involves explicit structural
    relationships between video and paper, or entity-centric lookups.

    Available templates and their required params:

    "lookup" — Direct attribute lookup, fetch a node by exact property.
        Use when the query asks what a specific page/timestamp/section contains.
        params: {"node_type": "video"|"paper"|"entity", "by": <field>, "value": <val>}
        node_type="video":  by="page" (int) | by="time" (float, seconds)
        node_type="paper":  by="page" (int) | by="section" (str) | by="chunk_id" (int)
        node_type="entity": by="name" (str, fuzzy-matched)
        Use when: "PPT第5页说了什么", "视频2分30秒在讲什么", "论文第3页内容", "论文3.2节"

    "video_to_paper" — Find paper chunks corresponding to a video slide.
        params: {"page": int} OR {"time": float (seconds)}
        Use when: "视频第X分钟/第X页讲的内容在论文哪里？"

    "paper_to_video" — Find video segments corresponding to a paper section.
        params: {"page": int} OR {"section": str}
        Use when: "论文第X页/第X节在视频哪里讲的？"

    "entity_mentions" — Find all nodes mentioning a specific entity.
        params: {"entity_name": str}
        Use when: "GRPO在视频和论文里分别是怎么介绍的？"

    "entity_co_occur" — Find nodes where two entities are both mentioned.
        params: {"entity1": str, "entity2": str}
        Use when: "PPO和GRPO在哪些地方被一起讨论？"

    "sequential" — Fetch a paper chunk and its following N chunks.
        params: {"chunk_id": int} OR {"page": int}, optionally {"hops": int (default 3)}
        Use when: "论文第X节后面是怎么展开的？"

    Returns:
        List of RetrievedChunk-compatible dicts.
    """
    from agent.rag import async_graph_retrieve
    return await async_graph_retrieve(template=template, params=params)


@tool
async def search_video(
    query: str,
    platform: Literal["bilibili", "youtube"] = "bilibili",
) -> list[dict]:
    """
    Search for academic presentation videos on Bilibili or YouTube.

    Args:
        query: Search query string (e.g., paper title or keywords).
        platform: Target platform.

    Returns:
        List of video candidates with title, url, and description.
    """
    # TODO: implement — call SerpAPI or platform API
    raise NotImplementedError("search_video: not implemented yet")


@tool
async def search_paper(query: str) -> list[dict]:
    """
    Search for academic papers by title or keywords on arXiv.

    Args:
        query: Paper title or keyword query (English preferred; mixed Chinese/English works too).

    Returns:
        List of up to 5 paper candidates, each with title, authors, abstract, and pdf_url
        in the location field. Pass the pdf_url to download_and_process_paper to ingest.
    """
    import asyncio
    import arxiv

    def _search() -> list[dict]:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for r in client.results(search):
            authors = ", ".join(str(a) for a in r.authors[:3])
            if len(r.authors) > 3:
                authors += " et al."
            results.append({
                "node_id": r.entry_id,
                "resource_id": "arxiv",
                "resource_type": "search_result",
                "content": (
                    f"**{r.title}**\n\n"
                    f"Authors: {authors}\n\n"
                    f"Abstract: {r.summary[:600]}"
                ),
                "summary": r.title,
                "score": 0.0,
                "location": r.pdf_url,
            })
        return results

    return await asyncio.to_thread(_search)


@tool
async def download_and_process_video(video_url: str) -> dict:
    """
    Download a video, extract subtitles (Whisper) and key frames (Gemini),
    then ingest into the GraphRAG database.
    The job runs in the background — returns a job_id to track progress via
    GET /ingest/{job_id}.

    Args:
        video_url: Full URL of the video (Bilibili or YouTube).

    Returns:
        job_id and queued status.
    """
    import uuid
    import os
    from arq import create_pool
    from arq.connections import RedisSettings

    job_id = str(uuid.uuid4())
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    pool = await create_pool(RedisSettings.from_dsn(redis_url))
    await pool.enqueue_job("process_video", video_url, job_id)
    await pool.close()
    return {"job_id": job_id, "status": "queued",
            "message": f"视频处理任务已提交，可通过 GET /ingest/{job_id} 查询进度"}


@tool
async def download_and_process_paper(pdf_url: str) -> dict:
    """
    Download a paper PDF, chunk it with Docling, and ingest into GraphRAG.
    The job runs in the background — returns a job_id to track progress via
    GET /ingest/{job_id}.

    Args:
        pdf_url: Direct URL to the PDF file.

    Returns:
        job_id and queued status.
    """
    import uuid
    import os
    from arq import create_pool
    from arq.connections import RedisSettings

    job_id = str(uuid.uuid4())
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    pool = await create_pool(RedisSettings.from_dsn(redis_url))
    await pool.enqueue_job("process_paper", pdf_url, job_id)
    await pool.close()
    return {"job_id": job_id, "status": "queued",
            "message": f"论文处理任务已提交，可通过 GET /ingest/{job_id} 查询进度"}


# ---------------------------------------------------------------------------
# Tool registry — used by the Brain Agent to describe tools to the Planner.
# MCP tools (e.g. GitHub) are injected at runtime by agent.mcp_client.init_mcp().
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    rag_retrieve,
    graph_rag_retrieve,
    search_video,
    search_paper,
    download_and_process_video,
    download_and_process_paper,
]

TOOL_MAP: dict[str, Any] = {t.name: t for t in ALL_TOOLS}


def get_tool_descriptions() -> str:
    """Return a formatted string of all tool names + full descriptions for prompts."""
    lines = []
    for t in ALL_TOOLS:
        lines.append(f"- {t.name}:\n  {t.description.strip()}")
    return "\n\n".join(lines)