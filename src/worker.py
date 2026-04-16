"""
ARQ Worker — background task definitions for slow ingestion jobs.

Start the worker with:
    arq src.worker.WorkerSettings

Each task function:
  - Takes ctx as first arg (ARQ convention, contains redis connection)
  - Updates job progress to Redis so /ingest/{job_id} can report status
  - Returns a dict that ARQ stores as the job result

Video pipeline (process_video):
  download → keyframe_extract → transcribe_extract → merge_keyframe_and_subtitles
  → subtitle_refine → PPT_describe → Milvus insert → Neo4j import
  → arXiv search → enqueue process_paper

Paper pipeline (process_paper):
  download PDF → pdf_chunk → Milvus insert → Neo4j import
"""

from __future__ import annotations

import json
import os
import sys

# Make sure src/ is on the path when running as `arq src.worker.WorkerSettings`
sys.path.insert(0, os.path.dirname(__file__))

import asyncio
from arq.connections import RedisSettings

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

async def _set_progress(redis, job_id: str, status: str, message: str) -> None:
    """Write job progress into Redis. TTL 1 hour."""
    await redis.set(
        f"job:{job_id}:progress",
        json.dumps({"status": status, "message": message}),
        ex=3600,
    )


# ---------------------------------------------------------------------------
# arXiv paper search helper
# ---------------------------------------------------------------------------

def _find_paper_pdf_url(title: str) -> str | None:
    """
    Search arXiv for a paper matching the given title/keywords.
    Returns the PDF URL of the top result, or None if not found.
    """
    try:
        import arxiv
        client = arxiv.Client()
        results = list(client.results(
            arxiv.Search(query=title, max_results=1, sort_by=arxiv.SortCriterion.Relevance)
        ))
        if results:
            return results[0].pdf_url
    except Exception as e:
        print(f"[arXiv search] 搜索失败: {e}")
    return None


# ---------------------------------------------------------------------------
# Task: process video
# ---------------------------------------------------------------------------

async def process_video(ctx, video_url: str, job_id: str) -> dict:
    """
    Full video ingestion pipeline:
      1. Download video (yt-dlp)
      2. Extract keyframes (scene detection + ffmpeg)
      3. Whisper transcription
      4. Merge keyframes + subtitles
      5. Subtitle refinement (Gemini)
      6. PPT content extraction (Gemini)
      7. Ingest into Milvus (BGE-M3 embeddings)
      8. Import into Neo4j (NEXT chain)
      9. Search arXiv for corresponding paper → enqueue process_paper
    """
    redis = ctx["redis"]

    try:
        from scripts.video_processor import VideoProcessor
        from scripts.milvus_processor import MilvusProcessor
        from scripts.graph_processor import GraphProcessor

        project_root = os.path.dirname(os.path.dirname(__file__))
        video_save_dir = os.path.join(project_root, "data", "videos") + "/"
        keyframe_dir = os.path.join(project_root, "data", "keyframes", job_id)
        db_path = os.environ.get("MILVUS_DB_PATH", os.path.join(project_root, "data", "dataset", "milvus_academic.db"))

        # ── 1. Download video ──────────────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "下载视频中...")
        result = await asyncio.to_thread(
            VideoProcessor.download_bilibili_video, video_url, video_save_dir
        )
        if result["status"] != "success":
            await _set_progress(redis, job_id, "failed", result.get("message", "下载失败"))
            return result

        video_path = result["file_path"]
        title = result["title"]

        # ── 2. Keyframe extraction ─────────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "提取关键帧中...")
        await asyncio.to_thread(
            VideoProcessor.keyframe_extract, video_path, keyframe_dir
        )
        metadata_path = os.path.join(keyframe_dir, "metadata.json")

        # ── 3. Whisper transcription ───────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "Whisper 字幕提取中...")
        await asyncio.to_thread(
            VideoProcessor.transcribe_extract, video_path, keyframe_dir
        )
        subscribe_path = os.path.join(keyframe_dir, "subscribe.json")

        # ── 4. Merge keyframes + subtitles ─────────────────────────────────
        await _set_progress(redis, job_id, "processing", "合并关键帧与字幕中...")
        group_path = os.path.join(keyframe_dir, "group.json")
        await asyncio.to_thread(
            VideoProcessor.merge_keyframe_and_subtitles,
            metadata_path, subscribe_path, group_path,
        )

        # ── 5. Subtitle refinement (Gemini, async) ─────────────────────────
        await _set_progress(redis, job_id, "processing", "Gemini 字幕精炼中...")
        group_1_path = os.path.join(keyframe_dir, "group_1.json")
        await VideoProcessor.subtitle_refine(group_path, keyframe_dir, group_1_path)

        # ── 6. PPT content extraction (Gemini, async) ──────────────────────
        await _set_progress(redis, job_id, "processing", "Gemini 解析 PPT 内容中...")
        group_2_path = os.path.join(keyframe_dir, "group_2.json")
        await VideoProcessor.PPT_describe(group_1_path, keyframe_dir, group_2_path)

        # ── 7. Milvus insert ───────────────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "写入 Milvus 向量库中...")
        def _milvus_ingest_video():
            mp = MilvusProcessor(db_path=db_path)
            mp.initialize_collections()
            mp.insert_video_segments(group_2_path)
        await asyncio.to_thread(_milvus_ingest_video)

        # ── 8. Neo4j import ────────────────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "写入 Neo4j 图数据库中...")
        def _neo4j_import_video():
            gp = GraphProcessor()
            gp.import_video_json(group_2_path)
            gp.close()
        await asyncio.to_thread(_neo4j_import_video)

        # ── 9. Search arXiv and enqueue paper job ──────────────────────────
        await _set_progress(redis, job_id, "processing", "搜索对应论文中...")
        pdf_url = await asyncio.to_thread(_find_paper_pdf_url, title)
        if pdf_url:
            import uuid
            from arq import create_pool
            paper_job_id = str(uuid.uuid4())
            pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
            await pool.enqueue_job("process_paper", pdf_url, paper_job_id)
            await pool.close()
            paper_msg = f"已自动提交论文处理任务，job_id: {paper_job_id}"
        else:
            paper_msg = "未在 arXiv 找到对应论文，请手动提交"

        await _set_progress(
            redis, job_id, "done",
            f"视频处理完成：{title}。{paper_msg}"
        )
        return {
            "status": "success",
            "title": title,
            "chunks_path": group_2_path,
            "paper_message": paper_msg,
        }

    except Exception as e:
        await _set_progress(redis, job_id, "failed", str(e))
        raise


# ---------------------------------------------------------------------------
# Task: process paper
# ---------------------------------------------------------------------------

async def process_paper(ctx, pdf_url: str, job_id: str) -> dict:
    """
    Full paper ingestion pipeline:
      1. Download PDF
      2. Docling chunking
      3. Ingest into Milvus (BGE-M3 embeddings)
      4. Import into Neo4j (NEXT chain)
    """
    redis = ctx["redis"]

    try:
        import urllib.request
        from scripts.pdf_processor import PDFProcessor
        from scripts.milvus_processor import MilvusProcessor
        from scripts.graph_processor import GraphProcessor

        project_root = os.path.dirname(os.path.dirname(__file__))
        db_path = os.environ.get("MILVUS_DB_PATH", os.path.join(project_root, "data", "dataset", "milvus_academic.db"))
        pdf_dir = os.path.join(project_root, "data", "papers", job_id)
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = os.path.join(pdf_dir, "paper.pdf")
        chunks_path = os.path.join(pdf_dir, "chunks.json")

        # ── 1. Download PDF ────────────────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "下载 PDF 中...")
        await asyncio.to_thread(urllib.request.urlretrieve, pdf_url, pdf_path)

        # ── 2. Docling chunking ────────────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "Docling 分块中...")
        await asyncio.to_thread(PDFProcessor.pdf_chunk, pdf_path, chunks_path)

        # ── 3. Milvus insert ───────────────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "写入 Milvus 向量库中...")
        def _milvus_ingest_paper():
            mp = MilvusProcessor(db_path=db_path)
            mp.initialize_collections()
            mp.insert_paper_chunks(chunks_path)
        await asyncio.to_thread(_milvus_ingest_paper)

        # ── 4. Neo4j import ────────────────────────────────────────────────
        await _set_progress(redis, job_id, "processing", "写入 Neo4j 图数据库中...")
        def _neo4j_import_paper():
            gp = GraphProcessor()
            gp.import_paper_json(chunks_path)
            gp.close()
        await asyncio.to_thread(_neo4j_import_paper)

        # Count chunks for summary
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_count = len(json.load(f))

        await _set_progress(redis, job_id, "done", f"处理完成：{chunk_count} 个 chunk")
        return {"status": "success", "chunk_count": chunk_count, "chunks_path": chunks_path}

    except Exception as e:
        await _set_progress(redis, job_id, "failed", str(e))
        raise


# ---------------------------------------------------------------------------
# ARQ WorkerSettings
# ---------------------------------------------------------------------------

class WorkerSettings:
    functions = [process_video, process_paper]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs = 4          # 同时最多执行 4 个任务
    job_timeout = 1800    # 单个任务最长 30 分钟
