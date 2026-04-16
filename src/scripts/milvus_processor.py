# milvus_processor.py
import json
import os
from pymilvus import (
    MilvusClient, DataType,
    AnnSearchRequest, RRFRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from tqdm import tqdm
import numpy as np


PAPER_COLLECTION = "paper_chunks"
VIDEO_COLLECTION = "video_segments"
DENSE_DIM = 1024  # BGE-M3 dense 维度


class MilvusProcessor:
    def __init__(self, db_path: str = "milvus_academic.db"):
        """
        db_path: Milvus Lite 本地文件路径，如 './data/milvus_academic.db'
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.client = MilvusClient(db_path)
        print("正在加载 BGE-M3 模型...")
        self.ef = BGEM3EmbeddingFunction(use_fp16=True, device="cuda")
        print("BGE-M3 加载完成。")

    # ── 1. 建库 ───────────────────────────────────────────────────

    def _create_paper_collection(self):
        """创建论文块 Collection"""
        if self.client.has_collection(PAPER_COLLECTION):
            print(f"[跳过] {PAPER_COLLECTION} 已存在")
            return

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("chunk_id",     DataType.INT64,          is_primary=True)
        schema.add_field("page",         DataType.INT64)
        schema.add_field("section",      DataType.VARCHAR,         max_length=512)
        schema.add_field("source",       DataType.VARCHAR,         max_length=256)
        schema.add_field("content",      DataType.VARCHAR,         max_length=8192)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR,    dim=DENSE_DIM)
        schema.add_field("sparse_vector",DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index("dense_vector",  index_type="AUTOINDEX", metric_type="IP")
        index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        self.client.create_collection(
            collection_name=PAPER_COLLECTION,
            schema=schema,
            index_params=index_params
        )
        print(f"✅ 创建 Collection: {PAPER_COLLECTION}")

    def _create_video_collection(self):
        """创建视频块 Collection"""
        if self.client.has_collection(VIDEO_COLLECTION):
            print(f"[跳过] {VIDEO_COLLECTION} 已存在")
            return

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("page",          DataType.INT64,         is_primary=True)
        schema.add_field("start_time",    DataType.FLOAT)
        schema.add_field("end_time",      DataType.FLOAT)
        schema.add_field("retrieve_text", DataType.VARCHAR,        max_length=8192)  # subtitle + PPT_wo_md
        schema.add_field("ppt_raw",       DataType.VARCHAR,        max_length=8192)  # PPT 原字段，仅存储
        schema.add_field("dense_vector",  DataType.FLOAT_VECTOR,   dim=DENSE_DIM)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index("dense_vector",  index_type="AUTOINDEX", metric_type="IP")
        index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        self.client.create_collection(
            collection_name=VIDEO_COLLECTION,
            schema=schema,
            index_params=index_params
        )
        print(f"✅ 创建 Collection: {VIDEO_COLLECTION}")

    def initialize_collections(self):
        self._create_paper_collection()
        self._create_video_collection()

    # ── 2. 插入数据 ───────────────────────────────────────────────

    def _sparse_to_dict(self, sparse_row) -> dict:
        """将 BGE-M3 sparse 输出的单行转成 Milvus 需要的 {int: float} dict"""
        # csr_matrix 单行转 coo 再转 dict
        coo = sparse_row.tocoo()
        return {int(col): float(val) for col, val in zip(coo.col, coo.data)}

    def insert_paper_chunks(self, paper_chunks_path: str, batch_size: int = 16):
        """读取论文 JSON，嵌入后插入 Milvus"""
        with open(paper_chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # 过滤掉 content 为空的块
        chunks = [c for c in chunks if c.get("content", "").strip()]
        print(f"论文块共 {len(chunks)} 条，开始插入...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="论文块插入", unit="batch"):
            batch = chunks[i: i + batch_size]
            texts = [c["content"] for c in batch]

            embeddings = self.ef(texts)
            dense_vecs = np.array(embeddings["dense"]).astype(np.float32)
            sparse_vecs = embeddings["sparse"]

            rows = []
            for j, chunk in enumerate(batch):
                rows.append({
                    "chunk_id":     chunk["chunk_id"],
                    "page":         chunk["metadata"].get("page", 0),
                    "section":      chunk["metadata"].get("section", "")[:512],
                    "source":       chunk["metadata"].get("source", "")[:256],
                    "content":      chunk["content"][:8192],
                    "dense_vector": dense_vecs[j],
                    "sparse_vector": self._sparse_to_dict(sparse_vecs[j]),
                })

            self.client.insert(collection_name=PAPER_COLLECTION, data=rows)

        print(f"✅ 论文块插入完成，共 {len(chunks)} 条")

    def insert_video_segments(self, video_chunks_path: str, batch_size: int = 16):
        """读取视频 JSON，嵌入后插入 Milvus"""
        with open(video_chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        chunks = [c for c in chunks if c.get("subtitle", "").strip() or c.get("PPT_wo_md", "").strip()]
        print(f"视频块共 {len(chunks)} 条，开始插入...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="视频块插入", unit="batch"):
            batch = chunks[i: i + batch_size]

            # subtitle + PPT_wo_md 拼接作为检索文本
            texts = []
            for c in batch:
                subtitle  = c.get("subtitle", "").strip()
                ppt_wo_md = c.get("PPT_wo_md", "").strip()
                texts.append(f"{subtitle}\n\n{ppt_wo_md}".strip()[:8192])

            embeddings = self.ef(texts)
            dense_vecs = np.array(embeddings["dense"]).astype(np.float32)
            sparse_vecs = embeddings["sparse"]

            rows = []
            for j, chunk in enumerate(batch):
                rows.append({
                    "page":          chunk["page"],
                    "start_time":    float(chunk.get("start", 0)),
                    "end_time":      float(chunk.get("end", 0)),
                    "retrieve_text": texts[j],
                    "ppt_raw":       chunk.get("PPT", "")[:8192],
                    "dense_vector":  dense_vecs[j],
                    "sparse_vector": self._sparse_to_dict(sparse_vecs[j]),
                })

            self.client.insert(collection_name=VIDEO_COLLECTION, data=rows)

        print(f"✅ 视频块插入完成，共 {len(chunks)} 条")

    # ── 3. 混合检索 ───────────────────────────────────────────────

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        output_fields: list = None,
    ) -> list:
        query_emb = self.ef([query])
        dense_vec  = np.array(query_emb["dense"][0]).astype(np.float32)
        sparse_vec = self._sparse_to_dict(query_emb["sparse"][0])  # ← 同样转 dict

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

        if output_fields is None:
            output_fields = (
                ["chunk_id", "page", "section", "content"]
                if collection_name == PAPER_COLLECTION
                else ["page", "start_time", "end_time", "retrieve_text"]
            )

        results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=output_fields,
        )
        return results


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    mp = MilvusProcessor(db_path=os.path.join(base, "../../data/dataset/GRPO.db"))
    # mp.initialize_collections()
    # mp.insert_paper_chunks(os.path.join(base, "../../data/pdf_tmp/GRPO.json"))
    # mp.insert_video_segments(os.path.join(base, "../../data/video_tmp/GRPO/group_2.json"))
    results = mp.hybrid_search(
        collection_name=VIDEO_COLLECTION,
        query="PPO如何推导到GRPO",
        top_k=3,
    )
    for hit in results[0]:
        print(hit)