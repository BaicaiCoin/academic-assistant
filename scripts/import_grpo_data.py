"""
Import pre-built GRPO data into Neo4j.

Run from the Academic_assistant/ root directory:
    conda activate acaAss
    python scripts/import_grpo_data.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

from scripts.graph_processor import GraphProcessor

PAPER_JSON   = "data/pdf_tmp/GRPO.json"
VIDEO_JSON   = "data/video_tmp/GRPO/group_2.json"
ENTITIES_JSON = "data/graph_tmp/GRPO/filtered_entities.json"
OVERLAP_JSON  = "data/graph_tmp/GRPO/overlap_stats.json"

def main():
    for path in [PAPER_JSON, VIDEO_JSON, ENTITIES_JSON, OVERLAP_JSON]:
        if not os.path.exists(path):
            print(f"[错误] 找不到文件：{path}")
            print("请确认你在 Academic_assistant/ 目录下运行此脚本。")
            sys.exit(1)

    print("正在连接 Neo4j...")
    gp = GraphProcessor()

    print("初始化数据库约束...")
    gp.initialize_schema()

    print("导入论文分块数据...")
    gp.import_paper_json(PAPER_JSON)

    print("导入视频分块数据...")
    gp.import_video_json(VIDEO_JSON)

    print("导入实体节点及关系...")
    gp.import_entities_and_relations(
        ENTITIES_JSON,
        OVERLAP_JSON,
        overlap_threshold=4,
    )

    gp.close()
    print("\n✅ 导入完成！现在可以启动服务并对 GRPO 进行问答了。")

if __name__ == "__main__":
    main()
