"""
GraphRAG retrieval module.

Wraps Milvus hybrid search + Neo4j graph expansion into a single
retrieve() function called by the rag_retrieve tool.

Pipeline:
  1. Embed the query with BGE-M3
  2. Hybrid search in Milvus (paper_chunks + video_segments)
  3. Fetch full node content from Neo4j by ID
  4. Optionally expand: fetch NEXT neighbours (prev/next chunks)
  5. Return list of RetrievedChunk-compatible dicts
"""

from __future__ import annotations

import os
from typing import Any

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from neo4j import GraphDatabase
import numpy as np

PAPER_COLLECTION = "paper_chunks"
VIDEO_COLLECTION = "video_segments"
DENSE_DIM = 1024


# ---------------------------------------------------------------------------
# Singletons — initialised once, reused across requests
# ---------------------------------------------------------------------------

_milvus_client: MilvusClient | None = None
_neo4j_driver = None
_ef: BGEM3EmbeddingFunction | None = None
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True,
            device="cuda",
        )
    return _reranker



def _get_milvus() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        db_path = os.environ["MILVUS_DB_PATH"]
        _milvus_client = MilvusClient(db_path)
    return _milvus_client


def _get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
    return _neo4j_driver


def _get_ef() -> BGEM3EmbeddingFunction:
    global _ef
    if _ef is None:
        _ef = BGEM3EmbeddingFunction(use_fp16=True, device="cuda")
    return _ef


def _get_neo4j_db() -> str:
    return os.environ.get("NEO4J_DATABASE", "neo4j")


def get_embed_fn():
    """
    Return an async embed function compatible with LangGraph Store's index API.
    Signature: (texts: list[str]) -> list[list[float]]
    Runs BGE-M3 inference in a thread pool to avoid blocking the event loop.
    """
    import asyncio
    import numpy as np

    async def _embed(texts: list[str]) -> list[list[float]]:
        ef = _get_ef()
        result = await asyncio.to_thread(ef, texts)
        return [np.array(v).astype(float).tolist() for v in result["dense"]]

    return _embed


def init_rag() -> None:
    """
    Pre-initialise all singletons (Milvus, Neo4j, BGE-M3, Reranker) at application startup.
    Call this once before the event loop starts to avoid blocking calls
    during async tool execution.
    """
    _get_milvus()
    _get_neo4j()
    _get_ef()
    _get_reranker()
    print("RAG singletons initialised.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sparse_to_dict(sparse_row) -> dict:
    coo = sparse_row.tocoo()
    return {int(col): float(val) for col, val in zip(coo.col, coo.data)}


def _milvus_hybrid_search(
    collection_name: str,
    query: str,
    top_k: int,
) -> list[dict]:
    """Run hybrid search on one Milvus collection, return raw hit dicts."""
    client = _get_milvus()
    ef = _get_ef()

    emb = ef([query])
    dense_vec = np.array(emb["dense"][0]).astype(np.float32)
    sparse_vec = _sparse_to_dict(emb["sparse"][0])

    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {}},
        limit=top_k,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {}},
        limit=top_k,
    )

    if collection_name == PAPER_COLLECTION:
        output_fields = ["chunk_id", "page", "section", "content"]
    else:
        output_fields = ["page", "start_time", "end_time", "retrieve_text", "ppt_raw"]

    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=output_fields,
    )

    hits = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        entity["_score"] = hit.get("distance", 0.0)
        entity["_collection"] = collection_name
        hits.append(entity)
    return hits


def _fetch_paper_neighbours(driver, chunk_ids: list[int], db: str) -> list[dict]:
    """Fetch NEXT-adjacent PaperChunk nodes for graph expansion."""
    result = driver.execute_query(
        """
        UNWIND $ids AS id
        MATCH (p:PaperChunk {chunk_id: id})
        OPTIONAL MATCH (prev:PaperChunk)-[:NEXT]->(p)
        OPTIONAL MATCH (p)-[:NEXT]->(nxt:PaperChunk)
        RETURN
            p.chunk_id   AS chunk_id,
            prev.chunk_id AS prev_id,
            prev.content  AS prev_content,
            prev.page     AS prev_page,
            nxt.chunk_id  AS next_id,
            nxt.content   AS next_content,
            nxt.page      AS next_page
        """,
        ids=chunk_ids,
        database_=db,
    )
    return [r.data() for r in result.records]


def _fetch_video_neighbours(driver, pages: list[int], db: str) -> list[dict]:
    """Fetch NEXT-adjacent VideoSegment nodes for graph expansion."""
    result = driver.execute_query(
        """
        UNWIND $pages AS pg
        MATCH (v:VideoSegment {page: pg})
        OPTIONAL MATCH (prev:VideoSegment)-[:NEXT]->(v)
        OPTIONAL MATCH (v)-[:NEXT]->(nxt:VideoSegment)
        RETURN
            v.page        AS page,
            prev.page     AS prev_page,
            prev.subtitle AS prev_subtitle,
            nxt.page      AS next_page,
            nxt.subtitle  AS next_subtitle
        """,
        pages=pages,
        database_=db,
    )
    return [r.data() for r in result.records]


def _fetch_corresponds_to(driver, chunk_ids: list[int], pages: list[int], db: str) -> list[dict]:
    """
    Fetch CORRESPONDS_TO links between retrieved PaperChunks and VideoSegments.
    Used when query seems to ask about cross-source relationships.
    """
    result = driver.execute_query(
        """
        UNWIND $chunk_ids AS cid
        MATCH (p:PaperChunk {chunk_id: cid})-[r:CORRESPONDS_TO]->(v:VideoSegment)
        RETURN
            p.chunk_id            AS chunk_id,
            v.page                AS video_page,
            r.common_entity_count AS common_count,
            r.common_entities     AS common_entities
        UNION
        UNWIND $pages AS pg
        MATCH (p:PaperChunk)-[r:CORRESPONDS_TO]->(v:VideoSegment {page: pg})
        RETURN
            p.chunk_id            AS chunk_id,
            v.page                AS video_page,
            r.common_entity_count AS common_count,
            r.common_entities     AS common_entities
        """,
        chunk_ids=chunk_ids,
        pages=pages,
        database_=db,
    )
    return [r.data() for r in result.records]


# ---------------------------------------------------------------------------
# Main retrieve function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 5,
    expand_graph: bool = True,
    search_paper: bool = True,
    search_video: bool = True,
) -> list[dict]:
    """
    Run GraphRAG retrieval and return RetrievedChunk-compatible dicts.
    This is the synchronous implementation called via asyncio.to_thread.
    """
    driver = _get_neo4j()
    db = _get_neo4j_db()
    chunks: list[dict] = []

    paper_hits: list[dict] = []
    video_hits: list[dict] = []

    if search_paper:
        paper_hits = _milvus_hybrid_search(PAPER_COLLECTION, query, top_k)
    if search_video:
        video_hits = _milvus_hybrid_search(VIDEO_COLLECTION, query, top_k)

    for hit in paper_hits:
        chunks.append({
            "node_id": f"paper_{hit['chunk_id']}",
            "resource_id": hit.get("source", "paper"),
            "resource_type": "paper_chunk",
            "content": hit.get("content", ""),
            "summary": "",
            "score": float(hit.get("_score", 0.0)),
            "location": f"p.{hit.get('page', '')} {hit.get('section', '')}".strip(),
        })

    for hit in video_hits:
        start = hit.get("start_time", 0)
        end = hit.get("end_time", 0)
        chunks.append({
            "node_id": f"video_{hit['page']}",
            "resource_id": "video",
            "resource_type": "video_slide",
            "content": hit.get("retrieve_text", ""),
            "summary": "",
            "score": float(hit.get("_score", 0.0)),
            "location": f"slide {hit['page']} ({_fmt_time(start)}-{_fmt_time(end)})",
        })

    # neighbour_node_id -> set of primary_node_ids it was fetched from.
    # Built during graph expansion and used to filter neighbours after reranking.
    neighbour_to_primary: dict[str, set[str]] = {}

    if expand_graph:
        paper_chunk_ids = [int(h["chunk_id"]) for h in paper_hits]
        video_pages = [int(h["page"]) for h in video_hits]

        if paper_chunk_ids:
            for row in _fetch_paper_neighbours(driver, paper_chunk_ids, db):
                primary_node_id = f"paper_{row['chunk_id']}"
                for side, content, page in [
                    ("prev", row.get("prev_content"), row.get("prev_page")),
                    ("next", row.get("next_content"), row.get("next_page")),
                ]:
                    if content and page is not None:
                        node_id = f"paper_{row[f'{side}_id']}"
                        if not any(c["node_id"] == node_id for c in chunks):
                            chunks.append({
                                "node_id": node_id,
                                "resource_id": "paper",
                                "resource_type": "paper_chunk",
                                "content": content,
                                "summary": "",
                                "score": 0.0,
                                "location": f"p.{page} (context)",
                            })
                        neighbour_to_primary.setdefault(node_id, set()).add(primary_node_id)

        if video_pages:
            for row in _fetch_video_neighbours(driver, video_pages, db):
                primary_node_id = f"video_{row['page']}"
                for side in ["prev", "next"]:
                    subtitle = row.get(f"{side}_subtitle")
                    page = row.get(f"{side}_page")
                    if subtitle and page is not None:
                        node_id = f"video_{page}"
                        if not any(c["node_id"] == node_id for c in chunks):
                            chunks.append({
                                "node_id": node_id,
                                "resource_id": "video",
                                "resource_type": "video_slide",
                                "content": subtitle,
                                "summary": "",
                                "score": 0.0,
                                "location": f"slide {page} (context)",
                            })
                        neighbour_to_primary.setdefault(node_id, set()).add(primary_node_id)

        if paper_chunk_ids and video_pages:
            for row in _fetch_corresponds_to(driver, paper_chunk_ids, video_pages, db):
                paper_node_id = f"paper_{row['chunk_id']}"
                video_node_id = f"video_{row['video_page']}"
                for c in chunks:
                    if c["node_id"] == paper_node_id:
                        c["summary"] = (
                            f"Corresponds to video slide {row['video_page']} "
                            f"(shared entities: {', '.join(row.get('common_entities', [])[:3])})"
                        )
                    if c["node_id"] == video_node_id:
                        c["summary"] = (
                            f"Corresponds to paper chunk {row['chunk_id']} "
                            f"(shared entities: {', '.join(row.get('common_entities', [])[:3])})"
                        )

    # ── Rerank ───────────────────────────────────────────────────────────────
    # Separate primary chunks (from vector search) from neighbour chunks (from NEXT expansion).
    # Only rerank primary chunks; neighbours follow their parent and are never reranked.
    primary_node_ids = (
        {f"paper_{h['chunk_id']}" for h in paper_hits} |
        {f"video_{h['page']}" for h in video_hits}
    )
    primary_chunks = [c for c in chunks if c["node_id"] in primary_node_ids]
    neighbour_chunks = [c for c in chunks if c["node_id"] not in primary_node_ids]

    if primary_chunks:
        try:
            reranker = _get_reranker()
            pairs = [[query, c["content"]] for c in primary_chunks]
            scores = reranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, (list, tuple)) and not hasattr(scores, "__len__"):
                scores = [scores]
            for c, s in zip(primary_chunks, scores):
                c["score"] = float(s)
            primary_chunks.sort(key=lambda x: x["score"], reverse=True)
            primary_chunks = primary_chunks[:top_k]
        except Exception:
            # Reranker failure is non-fatal — keep original order
            primary_chunks = primary_chunks[:top_k]

    # Keep only neighbours whose parent primary chunk survived the rerank cutoff.
    # Use the neighbour_to_primary mapping built during graph expansion (exact NEXT links),
    # rather than an integer-difference heuristic.
    surviving_ids = {c["node_id"] for c in primary_chunks}
    kept_neighbours = [
        c for c in neighbour_chunks
        if neighbour_to_primary.get(c["node_id"], set()) & surviving_ids
    ]

    return primary_chunks + kept_neighbours


async def async_retrieve(
    query: str,
    top_k: int = 5,
    expand_graph: bool = True,
    search_paper: bool = True,
    search_video: bool = True,
) -> list[dict]:
    """Async wrapper for retrieve() — runs in a thread to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(
        retrieve, query, top_k, expand_graph, search_paper, search_video
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Graph retrieval — Template-based Cypher queries
# ---------------------------------------------------------------------------

# Template identifiers — passed in as retrieval_template parameter
TEMPLATE_LOOKUP           = "lookup"            # 精确属性查找 → 直接定位节点
TEMPLATE_VIDEO_TO_PAPER   = "video_to_paper"    # 视频定位 → 找对应论文块
TEMPLATE_PAPER_TO_VIDEO   = "paper_to_video"    # 论文定位 → 找对应视频片段
TEMPLATE_ENTITY_MENTIONS  = "entity_mentions"   # 实体中心 → 找所有提及节点
TEMPLATE_ENTITY_CO_OCCUR  = "entity_co_occur"   # 两实体共现 → 找共同出现的节点
TEMPLATE_SEQUENTIAL       = "sequential"        # 顺序推理 → 找某节点的后续块

ALL_TEMPLATES = [
    TEMPLATE_LOOKUP,
    TEMPLATE_VIDEO_TO_PAPER,
    TEMPLATE_PAPER_TO_VIDEO,
    TEMPLATE_ENTITY_MENTIONS,
    TEMPLATE_ENTITY_CO_OCCUR,
    TEMPLATE_SEQUENTIAL,
]


def _normalize_entity_name(driver, name: str, db: str) -> str:
    """
    Fuzzy-match an entity name against Neo4j to handle case/spelling differences.
    Returns the best matching name stored in the graph, or the original if no match.
    """
    result = driver.execute_query(
        """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(e.name)
        RETURN e.name AS name
        ORDER BY size(e.name) ASC
        LIMIT 1
        """,
        name=name,
        database_=db,
    )
    records = result.records
    if records:
        return records[0].data()["name"]
    return name


def _tmpl_lookup(driver, db: str, params: dict) -> list[dict]:
    """
    Template 0: Direct attribute lookup — fetch a node by exact property value.
    Covers all simple "what does X say" queries without vector search.

    params:
        node_type (str): "video", "paper", or "entity"
        by (str):
            video  → "page" | "time" (seconds, finds slide containing that timestamp)
            paper  → "page" | "section" (partial match) | "chunk_id"
            entity → "name" (fuzzy-matched)
        value: the lookup value (int, float, or str depending on `by`)

    Examples:
        {"node_type": "video",  "by": "page",     "value": 5}
        {"node_type": "video",  "by": "time",      "value": 300}
        {"node_type": "paper",  "by": "page",      "value": 3}
        {"node_type": "paper",  "by": "section",   "value": "3.2"}
        {"node_type": "paper",  "by": "chunk_id",  "value": 42}
        {"node_type": "entity", "by": "name",      "value": "GRPO"}
    """
    node_type = params.get("node_type", "").lower()
    by = params.get("by", "").lower()
    value = params.get("value")

    if value is None:
        return []

    # ── Video node ───────────────────────────────────────────────────────────
    if node_type == "video":
        if by == "page":
            result = driver.execute_query(
                """
                MATCH (v:VideoSegment {page: $value})
                RETURN v.page AS page, v.subtitle AS subtitle,
                       v.ppt_content AS ppt_content,
                       v.start_time AS start_time, v.end_time AS end_time
                """,
                value=int(value), database_=db,
            )
        elif by == "time":
            result = driver.execute_query(
                """
                MATCH (v:VideoSegment)
                WHERE v.start_time <= $value AND v.end_time >= $value
                RETURN v.page AS page, v.subtitle AS subtitle,
                       v.ppt_content AS ppt_content,
                       v.start_time AS start_time, v.end_time AS end_time
                """,
                value=float(value), database_=db,
            )
        else:
            return []

        chunks = []
        for r in result.records:
            d = r.data()
            content = d.get("subtitle") or d.get("ppt_content") or ""
            chunks.append({
                "node_id": f"video_{d['page']}",
                "resource_id": "video",
                "resource_type": "video_slide",
                "content": content,
                "summary": "",
                "score": 1.0,
                "location": (
                    f"slide {d['page']} "
                    f"({_fmt_time(d.get('start_time', 0))}-{_fmt_time(d.get('end_time', 0))})"
                ),
            })
        return chunks

    # ── Paper node ───────────────────────────────────────────────────────────
    elif node_type == "paper":
        if by == "page":
            result = driver.execute_query(
                """
                MATCH (p:PaperChunk {page: $value})
                RETURN p.chunk_id AS chunk_id, p.content AS content,
                       p.page AS page, p.section AS section
                ORDER BY p.chunk_id
                """,
                value=int(value), database_=db,
            )
        elif by == "chunk_id":
            result = driver.execute_query(
                """
                MATCH (p:PaperChunk {chunk_id: $value})
                RETURN p.chunk_id AS chunk_id, p.content AS content,
                       p.page AS page, p.section AS section
                """,
                value=int(value), database_=db,
            )
        elif by == "section":
            result = driver.execute_query(
                """
                MATCH (p:PaperChunk)
                WHERE toLower(p.section) CONTAINS toLower($value)
                RETURN p.chunk_id AS chunk_id, p.content AS content,
                       p.page AS page, p.section AS section
                ORDER BY p.chunk_id
                """,
                value=str(value), database_=db,
            )
        else:
            return []

        chunks = []
        for r in result.records:
            d = r.data()
            chunks.append({
                "node_id": f"paper_{d['chunk_id']}",
                "resource_id": "paper",
                "resource_type": "paper_chunk",
                "content": d.get("content", ""),
                "summary": "",
                "score": 1.0,
                "location": f"p.{d.get('page', '')} {d.get('section', '')}".strip(),
            })
        return chunks

    # ── Entity node ──────────────────────────────────────────────────────────
    elif node_type == "entity":
        entity_name = _normalize_entity_name(driver, str(value), db)
        result = driver.execute_query(
            """
            MATCH (e:Entity {name: $name})
            RETURN e.name AS name, e.entity_type AS entity_type
            """,
            name=entity_name, database_=db,
        )
        chunks = []
        for r in result.records:
            d = r.data()
            chunks.append({
                "node_id": f"entity_{d['name'].replace(' ', '_')}",
                "resource_id": "graph",
                "resource_type": "search_result",
                "content": f"{d['name']} ({d.get('entity_type', '')})",
                "summary": "",
                "score": 1.0,
                "location": None,
            })
        return chunks

    return []


def _tmpl_video_to_paper(driver, db: str, params: dict) -> list[dict]:
    """
    Template 1: Given a video slide page or timestamp, find corresponding paper chunks.

    params:
        page (int, optional): slide page number
        time (float, optional): timestamp in seconds (uses start_time <= time <= end_time)
    """
    page = params.get("page")
    time = params.get("time")

    if page is not None:
        result = driver.execute_query(
            """
            MATCH (v:VideoSegment {page: $page})
            MATCH (p:PaperChunk)-[:CORRESPONDS_TO]->(v)
            RETURN
                p.chunk_id AS chunk_id,
                p.content  AS content,
                p.page     AS paper_page,
                p.section  AS section,
                v.page     AS video_page,
                v.subtitle AS subtitle,
                v.start_time AS start_time,
                v.end_time   AS end_time
            ORDER BY p.page
            """,
            page=int(page),
            database_=db,
        )
    elif time is not None:
        result = driver.execute_query(
            """
            MATCH (v:VideoSegment)
            WHERE v.start_time <= $time AND v.end_time >= $time
            MATCH (p:PaperChunk)-[:CORRESPONDS_TO]->(v)
            RETURN
                p.chunk_id AS chunk_id,
                p.content  AS content,
                p.page     AS paper_page,
                p.section  AS section,
                v.page     AS video_page,
                v.subtitle AS subtitle,
                v.start_time AS start_time,
                v.end_time   AS end_time
            ORDER BY p.page
            """,
            time=float(time),
            database_=db,
        )
    else:
        return []

    chunks = []
    for r in result.records:
        d = r.data()
        chunks.append({
            "node_id": f"paper_{d['chunk_id']}",
            "resource_id": "paper",
            "resource_type": "paper_chunk",
            "content": d["content"],
            "summary": (
                f"Corresponds to video slide {d['video_page']} "
                f"({_fmt_time(d['start_time'])}-{_fmt_time(d['end_time'])})"
            ),
            "score": 1.0,
            "location": f"p.{d['paper_page']} {d.get('section', '')}".strip(),
        })
    return chunks


def _tmpl_paper_to_video(driver, db: str, params: dict) -> list[dict]:
    """
    Template 2: Given a paper page or section, find corresponding video segments.

    params:
        page (int, optional): paper page number
        section (str, optional): section name (partial match)
    """
    page = params.get("page")
    section = params.get("section")

    if page is not None:
        result = driver.execute_query(
            """
            MATCH (p:PaperChunk {page: $page})
            MATCH (p)-[:CORRESPONDS_TO]->(v:VideoSegment)
            RETURN
                v.page       AS video_page,
                v.start_time AS start_time,
                v.end_time   AS end_time,
                v.subtitle   AS subtitle,
                v.ppt_content AS ppt_content,
                p.chunk_id   AS chunk_id,
                p.content    AS paper_content
            ORDER BY v.page
            """,
            page=int(page),
            database_=db,
        )
    elif section is not None:
        result = driver.execute_query(
            """
            MATCH (p:PaperChunk)
            WHERE toLower(p.section) CONTAINS toLower($section)
            MATCH (p)-[:CORRESPONDS_TO]->(v:VideoSegment)
            RETURN
                v.page       AS video_page,
                v.start_time AS start_time,
                v.end_time   AS end_time,
                v.subtitle   AS subtitle,
                v.ppt_content AS ppt_content,
                p.chunk_id   AS chunk_id,
                p.content    AS paper_content
            ORDER BY v.page
            """,
            section=section,
            database_=db,
        )
    else:
        return []

    chunks = []
    for r in result.records:
        d = r.data()
        chunks.append({
            "node_id": f"video_{d['video_page']}",
            "resource_id": "video",
            "resource_type": "video_slide",
            "content": d["subtitle"] or d.get("ppt_content", ""),
            "summary": f"Corresponds to paper chunk {d['chunk_id']}",
            "score": 1.0,
            "location": (
                f"slide {d['video_page']} "
                f"({_fmt_time(d['start_time'])}-{_fmt_time(d['end_time'])})"
            ),
        })
    return chunks


def _tmpl_entity_mentions(driver, db: str, params: dict) -> list[dict]:
    """
    Template 3: Find all nodes (paper chunks + video segments) that mention an entity.

    params:
        entity_name (str): entity name (will be fuzzy-matched against graph)
    """
    raw_name = params.get("entity_name", "")
    entity_name = _normalize_entity_name(driver, raw_name, db)

    result = driver.execute_query(
        """
        MATCH (e:Entity {name: $name})-[r:MENTIONED_IN]->(n)
        RETURN
            labels(n)        AS node_labels,
            n.chunk_id       AS chunk_id,
            n.page           AS page,
            n.content        AS content,
            n.subtitle       AS subtitle,
            n.ppt_content    AS ppt_content,
            n.start_time     AS start_time,
            n.end_time       AS end_time,
            n.section        AS section,
            r.importance     AS importance,
            r.mention_context AS mention_context
        ORDER BY r.importance DESC
        """,
        name=entity_name,
        database_=db,
    )

    chunks = []
    for r in result.records:
        d = r.data()
        is_paper = "PaperChunk" in (d.get("node_labels") or [])
        if is_paper:
            chunks.append({
                "node_id": f"paper_{d['chunk_id']}",
                "resource_id": "paper",
                "resource_type": "paper_chunk",
                "content": d.get("content", ""),
                "summary": f"Mentions '{entity_name}': {d.get('mention_context', '')}",
                "score": float(d.get("importance", 0.0)),
                "location": f"p.{d.get('page', '')} {d.get('section', '')}".strip(),
            })
        else:
            chunks.append({
                "node_id": f"video_{d['page']}",
                "resource_id": "video",
                "resource_type": "video_slide",
                "content": d.get("subtitle") or d.get("ppt_content", ""),
                "summary": f"Mentions '{entity_name}': {d.get('mention_context', '')}",
                "score": float(d.get("importance", 0.0)),
                "location": (
                    f"slide {d['page']} "
                    f"({_fmt_time(d.get('start_time', 0))}-{_fmt_time(d.get('end_time', 0))})"
                ),
            })
    return chunks


def _tmpl_entity_co_occur(driver, db: str, params: dict) -> list[dict]:
    """
    Template 4: Find nodes where two entities are both mentioned.

    params:
        entity1 (str): first entity name
        entity2 (str): second entity name
    """
    e1 = _normalize_entity_name(driver, params.get("entity1", ""), db)
    e2 = _normalize_entity_name(driver, params.get("entity2", ""), db)

    result = driver.execute_query(
        """
        MATCH (e1:Entity {name: $e1})-[:MENTIONED_IN]->(n)
        MATCH (e2:Entity {name: $e2})-[:MENTIONED_IN]->(n)
        RETURN
            labels(n)     AS node_labels,
            n.chunk_id    AS chunk_id,
            n.page        AS page,
            n.content     AS content,
            n.subtitle    AS subtitle,
            n.ppt_content AS ppt_content,
            n.start_time  AS start_time,
            n.end_time    AS end_time,
            n.section     AS section
        """,
        e1=e1,
        e2=e2,
        database_=db,
    )

    chunks = []
    for r in result.records:
        d = r.data()
        is_paper = "PaperChunk" in (d.get("node_labels") or [])
        if is_paper:
            chunks.append({
                "node_id": f"paper_{d['chunk_id']}",
                "resource_id": "paper",
                "resource_type": "paper_chunk",
                "content": d.get("content", ""),
                "summary": f"Co-mentions '{e1}' and '{e2}'",
                "score": 1.0,
                "location": f"p.{d.get('page', '')} {d.get('section', '')}".strip(),
            })
        else:
            chunks.append({
                "node_id": f"video_{d['page']}",
                "resource_id": "video",
                "resource_type": "video_slide",
                "content": d.get("subtitle") or d.get("ppt_content", ""),
                "summary": f"Co-mentions '{e1}' and '{e2}'",
                "score": 1.0,
                "location": (
                    f"slide {d['page']} "
                    f"({_fmt_time(d.get('start_time', 0))}-{_fmt_time(d.get('end_time', 0))})"
                ),
            })
    return chunks


def _tmpl_sequential(driver, db: str, params: dict) -> list[dict]:
    """
    Template 5: Fetch a paper chunk and its following N chunks via NEXT chain.

    params:
        chunk_id (int, optional): starting chunk_id
        page (int, optional): starting paper page (uses first chunk on that page)
        hops (int): how many NEXT steps to follow (default 3)
    """
    chunk_id = params.get("chunk_id")
    page = params.get("page")
    hops = int(params.get("hops", 3))

    if chunk_id is not None:
        result = driver.execute_query(
            """
            MATCH (start:PaperChunk {chunk_id: $chunk_id})
            MATCH path = (start)-[:NEXT*0..$hops]->(following)
            RETURN
                following.chunk_id AS chunk_id,
                following.content  AS content,
                following.page     AS page,
                following.section  AS section
            ORDER BY following.chunk_id
            """,
            chunk_id=int(chunk_id),
            hops=hops,
            database_=db,
        )
    elif page is not None:
        result = driver.execute_query(
            """
            MATCH (start:PaperChunk {page: $page})
            WITH start ORDER BY start.chunk_id LIMIT 1
            MATCH path = (start)-[:NEXT*0..$hops]->(following)
            RETURN
                following.chunk_id AS chunk_id,
                following.content  AS content,
                following.page     AS page,
                following.section  AS section
            ORDER BY following.chunk_id
            """,
            page=int(page),
            hops=hops,
            database_=db,
        )
    else:
        return []

    chunks = []
    for r in result.records:
        d = r.data()
        chunks.append({
            "node_id": f"paper_{d['chunk_id']}",
            "resource_id": "paper",
            "resource_type": "paper_chunk",
            "content": d.get("content", ""),
            "summary": "",
            "score": 1.0,
            "location": f"p.{d.get('page', '')} {d.get('section', '')}".strip(),
        })
    return chunks


# Template dispatch table
_TEMPLATE_FN = {
    TEMPLATE_LOOKUP:          _tmpl_lookup,
    TEMPLATE_VIDEO_TO_PAPER:  _tmpl_video_to_paper,
    TEMPLATE_PAPER_TO_VIDEO:  _tmpl_paper_to_video,
    TEMPLATE_ENTITY_MENTIONS: _tmpl_entity_mentions,
    TEMPLATE_ENTITY_CO_OCCUR: _tmpl_entity_co_occur,
    TEMPLATE_SEQUENTIAL:      _tmpl_sequential,
}


def graph_retrieve(template: str, params: dict) -> list[dict]:
    """Synchronous template-based graph query on Neo4j."""
    if template not in _TEMPLATE_FN:
        raise ValueError(
            f"Unknown graph retrieval template '{template}'. "
            f"Valid templates: {ALL_TEMPLATES}"
        )
    driver = _get_neo4j()
    db = _get_neo4j_db()
    return _TEMPLATE_FN[template](driver, db, params)


async def async_graph_retrieve(template: str, params: dict) -> list[dict]:
    """Async wrapper for graph_retrieve() — runs in a thread to avoid blocking."""
    import asyncio
    return await asyncio.to_thread(graph_retrieve, template, params)