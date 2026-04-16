from enum import Enum
import json
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from tqdm import tqdm

from neo4j import GraphDatabase
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class EntityType(str, Enum):
    CONCEPT = "Concept"
    METHOD = "Method"
    METRIC = "Metric"
    EQUATION = "Equation"
    EXPERIMENT = "Experiment"

class EntityItem(BaseModel):
    name: str = Field(description="实体的标准名称。如果是已知实体，必须使用已知列表中的名字。")
    entity_type: EntityType = Field(description="实体类别，必须是指定的枚举值之一")
    importance: float = Field(ge=0.0, le=1.0, description="在该段落中的重要性权重，0.0到1.0之间")
    mention_context: str = Field(description="该实体在本段落中的具体作用或提及背景")

class EntityExtractionResult(BaseModel):
    entities: List[EntityItem]


class GraphProcessor:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        self.db = "19c7cf4c"

    def close(self):
        self.driver.close()

    def initialize_schema(self):
        """初始化：创建唯一性约束"""
        print("正在初始化数据库约束...")
        self.driver.execute_query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PaperChunk) REQUIRE p.chunk_id IS UNIQUE",
            database_=self.db
        )
        self.driver.execute_query(
            # ✅ Bug 2 修复：seg_id → page
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:VideoSegment) REQUIRE v.page IS UNIQUE",
            database_=self.db
        )
        print("约束初始化完成。")

    def import_paper_json(self, file_path):
        """导入论文 JSON 并建立 NEXT 关系"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"正在导入论文数据 ({len(data)} 条)...")
        self.driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (p:PaperChunk {chunk_id: row.chunk_id})
            SET p.content = row.content,
                p.page = row.metadata.page,
                p.section = row.metadata.section,
                p.source = row.metadata.source
            """,
            rows=data,
            database_=self.db
        )
        self.driver.execute_query(
            """
            MATCH (p1:PaperChunk), (p2:PaperChunk)
            WHERE p2.chunk_id = p1.chunk_id + 1
            MERGE (p1)-[:NEXT]->(p2)
            """,
            database_=self.db
        )
        print("论文数据导入及链式关系建立完毕。")

    def import_video_json(self, file_path):
        """导入视频/PPT JSON 并建立 NEXT 关系"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"正在导入视频数据 ({len(data)} 条)...")
        self.driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (v:VideoSegment {page: row.page})
            SET v.start_time = row.start,
                v.end_time = row.end,
                v.file = row.file,
                v.subtitle = row.subtitle,
                v.ppt_content = row.PPT
            """,
            rows=data,
            database_=self.db
        )
        self.driver.execute_query(
            """
            MATCH (v:VideoSegment)
            WITH v ORDER BY v.page ASC
            WITH collect(v) AS nodes
            FOREACH (i in RANGE(0, size(nodes)-2) |
                FOREACH (node1 in [nodes[i]] |
                    FOREACH (node2 in [nodes[i+1]] |
                        MERGE (node1)-[:NEXT]->(node2)
                    )
                )
            )
            """,
            database_=self.db
        )
        print("视频数据导入及链式关系建立完毕。")

    # ✅ Bug 1 修复：@retry 装饰器移到这里
    @retry(
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            Exception,
        )),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(10),
        before_sleep=lambda retry_state: print(
            f"    ⚠️  第 {retry_state.attempt_number} 次失败，"
            f"{retry_state.next_action.sleep:.1f}s 后重试..."
        )
    )
    def _call_gemini(self, prompt: str) -> dict:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL"),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=EntityExtractionResult,
                temperature=0.1,
                http_options=types.HttpOptions(timeout=30000)  # 单位是毫秒
            ),
        )
        return json.loads(response.text)

    def extract_entities(self, content: str, existing_entities: List[str]) -> list:
        existing_entities_str = ", ".join(existing_entities) if existing_entities else "无"

        prompt = f"""
        你是一个资深的 AI 论文分析专家，擅长从复杂的学术文本中提取结构化知识。请从下方的【论文/PPT文本块】中提取核心学术实体的**英文名称**。
        
        ### 任务规则：
        1. **名称一致性**：如果提取到的实体与下方的【已知实体列表】中的某项语义相同，请**务必**直接使用已知列表中的名称。
        2. **实体细分**：
            - Concept: 基础概念或术语
            - Method: 具体的算法、模型或技术方案
            - Metric: 评估指标、实验参数
            - Equation: 数学公式或关键变量名
            - Experiment: 实验阶段（如预训练、消融实验）
        3. **评估重要性**：范围 0.0 - 1.0。文本核心讨论对象重要性应接近1.0，若只是单纯提及（如话题背景等），则应该获得较小分数。
        4. **实体角色**：描述该实体在本段落中扮演的角色（例如：作为被改进的对象、作为对比的Baseline、或是本文新提出的核心创新）。

        【已知实体列表】: {existing_entities_str}

        【待处理论文/PPT文本块】: 
        {content}

        【输出格式要求】:
        仅返回一个 JSON 数组，格式如下：
        [
            {{
                "name": "实体名称",
                "type": "实体类别",
                "importance": 0.0 - 1.0,
                "mention_context": "本实体在文本中扮演的角色"
            }}
        ]
        """

        try:
            result = self._call_gemini(prompt)
            return result.get("entities", [])
        except Exception as e:
            print(f"    ❌ 实体提取最终失败，跳过本块: {e}")
            return []

    def build_entity_nodes(
        self,
        paper_chunks_path: str,
        video_chunks_path: str,
        output_path: str = "entity_nodes.json",
        checkpoint_path: str = "entity_extraction_checkpoint.json"
    ) -> List[Dict[str, Any]]:

        with open(paper_chunks_path, "r", encoding="utf-8") as f:
            paper_chunks = json.load(f)
        with open(video_chunks_path, "r", encoding="utf-8") as f:
            video_chunks = json.load(f)

        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            processed_ids = set(checkpoint.get("processed_ids", []))
            existing_entities = checkpoint.get("existing_entities", [])
            all_entity_nodes = checkpoint.get("all_entity_nodes", [])
            print(f"[断点恢复] 已处理 {len(processed_ids)} 块，已知实体 {len(existing_entities)} 个")
        else:
            processed_ids = set()
            existing_entities = []
            all_entity_nodes = []

        def save_checkpoint():
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({
                    "processed_ids": list(processed_ids),
                    "existing_entities": existing_entities,
                    "all_entity_nodes": all_entity_nodes
                }, f, ensure_ascii=False, indent=2)

        paper_todo = [c for c in paper_chunks if f"paper_{c['chunk_id']}" not in processed_ids]
        with tqdm(total=len(paper_chunks), initial=len(paper_chunks) - len(paper_todo),
                  desc="论文块", unit="块", dynamic_ncols=True) as pbar:
            for chunk in paper_todo:
                chunk_id = chunk["chunk_id"]
                block_uid = f"paper_{chunk_id}"
                content = chunk.get("content", "").strip()
                if not content:
                    processed_ids.add(block_uid)
                    pbar.update(1)
                    continue

                entities = self.extract_entities(content, existing_entities)
                for entity in entities:
                    node = entity if isinstance(entity, dict) else entity.model_dump()
                    node["source_type"] = "paper"
                    node["source_id"] = chunk_id
                    node["source_meta"] = chunk.get("metadata", {})
                    all_entity_nodes.append(node)

                new_names = [
                    e["name"] if isinstance(e, dict) else e.name
                    for e in entities
                    if (e["name"] if isinstance(e, dict) else e.name) not in existing_entities
                ]
                existing_entities.extend(new_names)
                processed_ids.add(block_uid)
                save_checkpoint()
                pbar.update(1)
                pbar.set_postfix({
                    "新增实体": len(new_names),
                    "累计实体": len(existing_entities),
                    "累计记录": len(all_entity_nodes)
                })

        video_todo = [c for c in video_chunks if f"video_{c['page']}" not in processed_ids]
        with tqdm(total=len(video_chunks), initial=len(video_chunks) - len(video_todo),
                  desc="视频块", unit="块", dynamic_ncols=True) as pbar:
            for chunk in video_todo:
                page = chunk["page"]
                block_uid = f"video_{page}"
                subtitle = chunk.get("subtitle", "").strip()
                ppt_wo_md = chunk.get("PPT_wo_md", "").strip()
                content_parts = []
                if subtitle:
                    content_parts.append(f"[字幕摘要]\n{subtitle}")
                if ppt_wo_md:
                    content_parts.append(f"[PPT内容]\n{ppt_wo_md}")
                content = "\n\n".join(content_parts)

                if not content:
                    processed_ids.add(block_uid)
                    pbar.update(1)
                    continue

                entities = self.extract_entities(content, existing_entities)
                for entity in entities:
                    node = entity if isinstance(entity, dict) else entity.model_dump()
                    node["source_type"] = "video"
                    node["source_id"] = page
                    node["source_meta"] = {
                        "start": chunk.get("start"),
                        "end": chunk.get("end"),
                        "file": chunk.get("file")
                    }
                    all_entity_nodes.append(node)

                new_names = [
                    e["name"] if isinstance(e, dict) else e.name
                    for e in entities
                    if (e["name"] if isinstance(e, dict) else e.name) not in existing_entities
                ]
                existing_entities.extend(new_names)
                processed_ids.add(block_uid)
                save_checkpoint()
                pbar.update(1)
                pbar.set_postfix({
                    "新增实体": len(new_names),
                    "累计实体": len(existing_entities),
                    "累计记录": len(all_entity_nodes)
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_entity_records": len(all_entity_nodes),
                "unique_entity_names": len(set(e["name"] for e in all_entity_nodes)),
                "existing_entities": existing_entities,
                "entity_nodes": all_entity_nodes
            }, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 完成！共 {len(all_entity_nodes)} 条实体记录 / {len(existing_entities)} 个唯一实体 → {output_path}")
        return all_entity_nodes
    
    def filter_entities(
        self,
        entity_nodes_path: str,
        importance_stats_path: str,
        output_path: str = "filtered_entities.json"
    ):
        """
        按组合条件筛选实体，输出保留的实体节点列表
        """
        with open(entity_nodes_path, "r", encoding="utf-8") as f:
            entity_data = json.load(f)
        with open(importance_stats_path, "r", encoding="utf-8") as f:
            importance_data = json.load(f)

        entity_nodes = entity_data["entity_nodes"]
        entity_index = {e["name"]: e for e in importance_data["entities"]}

        kept_names = set()
        filtered_reasons = {}

        for name, stats in entity_index.items():
            is_cross_source = stats["paper_mentions"] > 0 and stats["video_mentions"] > 0
            avg_imp = stats["avg_importance"]
            max_imp = stats["max_importance"]
            mentions = stats["total_mentions"]
            entity_type = None

            for node in entity_nodes:
                if node["name"] == name:
                    entity_type = node.get("entity_type", "")
                    break

            if entity_type == "Equation" and mentions <= 2:
                filtered_reasons[name] = "过滤：单次/少量提及的公式变量"
                continue

            reason = None
            if is_cross_source and avg_imp >= 0.5:
                reason = f"跨源实体 + avg_importance={avg_imp:.3f}"
            elif mentions >= 5 and avg_imp >= 0.5:
                reason = f"高频实体 mentions={mentions} + avg_importance={avg_imp:.3f}"
            elif max_imp >= 0.9 and mentions >= 3:
                reason = f"高峰值重要性 max={max_imp:.3f} + mentions={mentions}"

            if reason:
                kept_names.add(name)
                filtered_reasons[name] = f"保留：{reason}"
            else:
                filtered_reasons[name] = (
                    f"过滤：不满足任一条件 "
                    f"(cross={is_cross_source}, avg={avg_imp:.3f}, "
                    f"max={max_imp:.3f}, mentions={mentions})"
                )

        filtered_nodes = [n for n in entity_nodes if n["name"] in kept_names]

        output = {
            "summary": {
                "original_unique_entities": len(entity_index),
                "kept_unique_entities": len(kept_names),
                "original_records": len(entity_nodes),
                "kept_records": len(filtered_nodes),
            },
            "kept_entity_names": sorted(kept_names),
            "filter_log": filtered_reasons,
            "entity_nodes": filtered_nodes
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        kept_stats = [entity_index[n] for n in kept_names]
        cross_count = sum(
            1 for s in kept_stats
            if s["paper_mentions"] > 0 and s["video_mentions"] > 0
        )

        print(f"原始唯一实体: {len(entity_index)} 个")
        print(f"保留唯一实体: {len(kept_names)} 个")
        print(f"  其中跨源实体（论文+视频）: {cross_count} 个")
        print(f"原始实体记录: {len(entity_nodes)} 条")
        print(f"保留实体记录: {len(filtered_nodes)} 条")

        print(f"\n被过滤掉的实体:")
        for name, reason in filtered_reasons.items():
            if reason.startswith("过滤"):
                print(f"  ✗ {name:<45} {reason}")

        return output
    
    def import_entities_and_relations(
        self,
        filtered_entities_path: str,
        overlap_stats_path: str,
        overlap_threshold: int = 4
    ):
        """
        1. 将筛选后的实体写入 Neo4j 作为 Entity 节点
        2. 建立实体与论文块/视频块之间的 MENTIONED_IN 关系
        3. 建立共同实体数 >= overlap_threshold 的论文块与视频块之间的 CORRESPONDS_TO 关系
        """

        with open(filtered_entities_path, "r", encoding="utf-8") as f:
            filtered_data = json.load(f)
        with open(overlap_stats_path, "r", encoding="utf-8") as f:
            overlap_data = json.load(f)

        entity_nodes = filtered_data["entity_nodes"]
        overlap_records = overlap_data["overlap_records"]

        # ── 1. 添加 Entity 节点唯一性约束 ────────────────────────────
        print("初始化 Entity 约束...")
        self.driver.execute_query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            database_=self.db
        )

        # ── 2. 批量 MERGE Entity 节点 ─────────────────────────────────
        # 先按 name 去重，每个唯一实体只建一个节点
        unique_entities = {}
        for node in entity_nodes:
            name = node["name"]
            if name not in unique_entities:
                unique_entities[name] = {
                    "name": name,
                    "entity_type": node.get("entity_type", ""),
                }

        print(f"写入 {len(unique_entities)} 个 Entity 节点...")
        self.driver.execute_query(
            """
            UNWIND $entities AS e
            MERGE (n:Entity {name: e.name})
            SET n.entity_type = e.entity_type
            """,
            entities=list(unique_entities.values()),
            database_=self.db
        )
        print("Entity 节点写入完毕。")

        # ── 3. 建立 MENTIONED_IN 关系 ─────────────────────────────────
        # 分论文和视频分别处理，因为目标节点类型不同

        paper_mentions = [n for n in entity_nodes if n["source_type"] == "paper"]
        video_mentions = [n for n in entity_nodes if n["source_type"] == "video"]

        print(f"建立论文提及关系 ({len(paper_mentions)} 条)...")
        self.driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {name: row.name})
            MATCH (p:PaperChunk {chunk_id: row.source_id})
            MERGE (e)-[r:MENTIONED_IN]->(p)
            SET r.importance = row.importance,
                r.mention_context = row.mention_context
            """,
            rows=paper_mentions,
            database_=self.db
        )

        print(f"建立视频提及关系 ({len(video_mentions)} 条)...")
        self.driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {name: row.name})
            MATCH (v:VideoSegment {page: row.source_id})
            MERGE (e)-[r:MENTIONED_IN]->(v)
            SET r.importance = row.importance,
                r.mention_context = row.mention_context
            """,
            rows=video_mentions,
            database_=self.db
        )
        print("提及关系建立完毕。")

        # ── 4. 建立 CORRESPONDS_TO 关系 ───────────────────────────────
        qualified_pairs = [
            r for r in overlap_records
            if r["common_count"] >= overlap_threshold
        ]

        print(f"建立对应关系（阈值>={overlap_threshold}，共 {len(qualified_pairs)} 对）...")
        self.driver.execute_query(
            """
            UNWIND $pairs AS pair
            MATCH (p:PaperChunk {chunk_id: pair.paper_chunk_id})
            MATCH (v:VideoSegment {page: pair.video_page})
            MERGE (p)-[r:CORRESPONDS_TO]->(v)
            SET r.common_entity_count = pair.common_count,
                r.common_entities = pair.common_entities
            """,
            pairs=qualified_pairs,
            database_=self.db
        )
        print(f"对应关系建立完毕，共 {len(qualified_pairs)} 对。")
        print("\n✅ 全部完成！")


if __name__ == "__main__":
    manager = GraphProcessor()
    # manager.build_entity_nodes(
    #     "./data/pdf_tmp/GRPO.json",
    #     "./data/video_tmp/GRPO/group_2.json",
    #     "data/graph_tmp/GRPO/entity_nodes.json",
    #     "data/graph_tmp/GRPO/entity_extraction_checkpoint.json"
    # )
    # manager.filter_entities(
    #     "data/graph_tmp/GRPO/entity_nodes.json",
    #     "data/graph_tmp/GRPO/importance_stats.json",
    #     "data/graph_tmp/GRPO/filtered_entities.json"
    # )
    manager.import_entities_and_relations(
        "data/graph_tmp/GRPO/filtered_entities.json",
        "data/graph_tmp/GRPO/overlap_stats.json",
        overlap_threshold=4
    )