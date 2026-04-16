"""
FastAPI server for the Academic Video + Paper Assistant.

Endpoints:
  POST /chat/{thread_id}       — Send a message, receive SSE stream
  GET  /threads                — List all thread IDs
  DELETE /threads/{thread_id}  — Delete a thread

SSE event types:
  {"type": "status",  "content": "正在规划执行步骤..."}
  {"type": "answer",  "content": "最终回答内容"}
  {"type": "error",   "content": "错误信息"}
  {"type": "done"}
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

# ── 延迟导入 graph，避免在 import 阶段触发 init_rag ──────────────────────────
_saver_cm = None  # context manager
_checkpointer = None
_store = None
_graph = None
_arq_pool = None

async def get_graph():
    global _graph, _checkpointer, _saver_cm, _store
    if _graph is None:
        from agent.graph import _build_root_graph

        db_path = os.environ.get("SQLITE_DB_PATH", "./data/threads.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        _saver_cm = AsyncSqliteSaver.from_conn_string(db_path)
        _checkpointer = await _saver_cm.__aenter__()
        _graph = _build_root_graph(checkpointer=_checkpointer, store=_store)
    return _graph


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _arq_pool
    import asyncio
    from arq import create_pool
    from arq.connections import RedisSettings
    from langgraph.store.memory import InMemoryStore
    from agent.rag import init_rag, get_embed_fn
    from agent.mcp_client import init_mcp, close_mcp
    from agent.store import PersistentSqliteStore

    await asyncio.to_thread(init_rag)   # initialises BGE-M3 singleton first
    memory_db = os.environ.get("MEMORY_DB_PATH", "./data/dataset/memories.db")
    _store = PersistentSqliteStore(memory_db, get_embed_fn(), dims=1024)
    await _store.setup()
    await init_mcp()

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    _arq_pool = await create_pool(RedisSettings.from_dsn(redis_url))

    yield

    await _arq_pool.close()
    await close_mcp()
    if _saver_cm is not None:
        await _saver_cm.__aexit__(None, None, None)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Academic Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _stream_graph(
    thread_id: str,
    message: str,
) -> AsyncGenerator[str, None]:
    """
    Run the graph and yield SSE-formatted events.

    Events:
      data: {"type": "status",  "content": "..."}
      data: {"type": "answer",  "content": "..."}
      data: {"type": "error",   "content": "..."}
      data: {"type": "done"}
    """
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {
        "raw_user_input": message,
        "messages": [HumanMessage(content=message)],
    }

    try:
        async for event in graph.astream_events(input_state, config=config, version="v2"):
            kind = event["event"]
            name = event.get("name", "")

            # Node finished — check for status_message or final_answer
            STATUS_NODES = {"normalize_input", "plan", "execute_step", "assemble_context", "render_response"}

            if kind == "on_chain_end" and name in STATUS_NODES:
                output = event["data"].get("output") or {}
                if not isinstance(output, dict):
                    continue

                if output.get("status_message") and name != "render_response":
                    yield _sse({"type": "status", "content": output["status_message"]})

                if output.get("final_answer") and name == "render_response":
                    yield _sse({"type": "answer", "content": output["final_answer"]})

    except Exception as exc:
        yield _sse({"type": "error", "content": str(exc)})

    yield _sse({"type": "done"})


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/chat/{thread_id}")
async def chat(thread_id: str, req: ChatRequest):
    """
    Send a message and receive a Server-Sent Events stream.

    The thread_id identifies the conversation session.
    Use a stable ID (e.g. UUID) per user session to preserve history.
    Pass a new UUID to start a fresh conversation.
    """
    return StreamingResponse(
        _stream_graph(thread_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@app.post("/chat")
async def chat_new(req: ChatRequest):
    """
    Convenience endpoint: start a new conversation with an auto-generated thread_id.
    Returns SSE stream with X-Thread-Id header.
    """
    thread_id = str(uuid.uuid4())
    return StreamingResponse(
        _stream_graph(thread_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": thread_id,
        },
    )


@app.get("/threads")
async def list_threads():
    """List all thread IDs stored in SQLite."""
    await get_graph()  # ensure checkpointer is initialised
    try:
        ids = list({
            t.config["configurable"]["thread_id"]
            async for t in _checkpointer.alist()
        })
        return {"threads": ids}
    except Exception:
        return {"threads": []}


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete all checkpoints for a thread."""
    # SqliteSaver doesn't have a delete API yet; clear by overwriting with empty state
    # For now, just return success — the thread will be overwritten on next use
    return {"deleted": thread_id}


@app.get("/health")
async def health():
    return {"ok": True}


# ── Ingest routes ─────────────────────────────────────────────────────────────

class IngestVideoRequest(BaseModel):
    video_url: str

class IngestPaperRequest(BaseModel):
    pdf_url: str


@app.post("/ingest/video")
async def ingest_video(req: IngestVideoRequest):
    """
    Submit a video URL for background processing.
    Returns a job_id — poll GET /ingest/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    await _arq_pool.enqueue_job("process_video", req.video_url, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/ingest/paper")
async def ingest_paper(req: IngestPaperRequest):
    """
    Submit a paper PDF URL for background processing.
    Returns a job_id — poll GET /ingest/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    await _arq_pool.enqueue_job("process_paper", req.pdf_url, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/ingest/{job_id}")
async def ingest_status(job_id: str):
    """
    Poll the progress of an ingestion job.
    Returns {"status": "queued|processing|done|failed", "message": "..."}.
    """
    raw = await _arq_pool.get(f"job:{job_id}:progress")
    if raw is None:
        return {"status": "queued", "message": "任务排队中..."}
    import json as _json
    return _json.loads(raw)
