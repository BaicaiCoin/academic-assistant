import json
from collections import defaultdict
from itertools import product

def analyze_entity_overlap(entity_nodes_path: str, output_path: str = "overlap_stats.json"):
    """
    统计每对 (PaperChunk, VideoSegment) 之间的共同实体数量
    """
    with open(entity_nodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    entity_nodes = data["entity_nodes"]

    # ── 1. 按来源分组：每个块拥有哪些实体 ──────────────────────
    paper_entities = defaultdict(set)   # {chunk_id: {entity_name, ...}}
    video_entities = defaultdict(set)   # {page: {entity_name, ...}}

    for node in entity_nodes:
        name = node["name"]
        if node["source_type"] == "paper":
            paper_entities[node["source_id"]].add(name)
        elif node["source_type"] == "video":
            video_entities[node["source_id"]].add(name)

    # ── 2. 计算每对的共同实体 ────────────────────────────────────
    overlap_records = []
    for (paper_id, p_ents), (video_id, v_ents) in product(
        paper_entities.items(), video_entities.items()
    ):
        common = p_ents & v_ents  # 取交集
        if common:  # 只记录有共同实体的对
            overlap_records.append({
                "paper_chunk_id": paper_id,
                "video_page": video_id,
                "common_count": len(common),
                "common_entities": sorted(common)
            })

    # 按共同实体数量降序排列
    overlap_records.sort(key=lambda x: x["common_count"], reverse=True)

    # ── 3. 汇总统计 ──────────────────────────────────────────────
    all_counts = [r["common_count"] for r in overlap_records]
    count_distribution = defaultdict(int)
    for c in all_counts:
        count_distribution[c] += 1

    stats = {
        "summary": {
            "total_paper_chunks": len(paper_entities),
            "total_video_segments": len(video_entities),
            "total_pairs_with_overlap": len(overlap_records),
            "max_common_entities": max(all_counts) if all_counts else 0,
            "avg_common_entities": round(sum(all_counts) / len(all_counts), 2) if all_counts else 0,
        },
        # 共同实体数量的分布，帮你判断阈值
        "count_distribution": {
            f"共同实体={k}": v
            for k, v in sorted(count_distribution.items())
        },
        # 阈值参考：>=N 的对有多少
        "threshold_reference": {
            f">={threshold}": sum(1 for c in all_counts if c >= threshold)
            for threshold in [1, 2, 3, 5, 8, 10]
        },
        "overlap_records": overlap_records
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── 4. 打印摘要 ──────────────────────────────────────────────
    print(f"论文块数量: {stats['summary']['total_paper_chunks']}")
    print(f"视频块数量: {stats['summary']['total_video_segments']}")
    print(f"有共同实体的对数: {stats['summary']['total_pairs_with_overlap']}")
    print(f"最多共同实体数: {stats['summary']['max_common_entities']}")
    print(f"平均共同实体数: {stats['summary']['avg_common_entities']}")
    print(f"\n共同实体数量分布:")
    for k, v in sorted(count_distribution.items()):
        bar = "█" * min(v, 50)
        print(f"  {k:3d} 个共同实体: {bar} ({v} 对)")
    print(f"\n阈值参考:")
    for threshold, count in stats["threshold_reference"].items():
        print(f"  共同实体 {threshold}: {count} 对")
    # 加在 analyze_entity_overlap 末尾，打印 top10 和边界附近的样本
    print("\n── 共同实体最多的 Top 5 对 ──")
    for r in overlap_records[:5]:
        print(f"  Paper #{r['paper_chunk_id']} <-> Video Page {r['video_page']}: "
            f"{r['common_count']} 个 → {r['common_entities']}")

    print("\n── 共同实体=3 的样本（前5对）──")
    for r in [x for x in overlap_records if x['common_count'] == 3][:5]:
        print(f"  Paper #{r['paper_chunk_id']} <-> Video Page {r['video_page']}: "
            f"{r['common_entities']}")

    print("\n── 共同实体=2 的样本（前5对）──")
    for r in [x for x in overlap_records if x['common_count'] == 2][:5]:
        print(f"  Paper #{r['paper_chunk_id']} <-> Video Page {r['video_page']}: "
            f"{r['common_entities']}")

    return stats


def analyze_entity_importance(entity_nodes_path: str, output_path: str = "importance_stats.json"):
    """
    按实体名聚合，统计每个实体的重要性分布
    """
    with open(entity_nodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entity_nodes = data["entity_nodes"]

    # ── 1. 按实体名聚合 ──────────────────────────────────────────
    from collections import defaultdict
    entity_stats = defaultdict(lambda: {
        "mentions": [],        # 所有提及记录
        "paper_mentions": 0,
        "video_mentions": 0,
    })

    for node in entity_nodes:
        name = node["name"]
        importance = node.get("importance", 0)
        entity_stats[name]["mentions"].append(importance)
        if node["source_type"] == "paper":
            entity_stats[name]["paper_mentions"] += 1
        else:
            entity_stats[name]["video_mentions"] += 1

    # ── 2. 计算统计指标 ──────────────────────────────────────────
    result = []
    for name, stats in entity_stats.items():
        mentions = stats["mentions"]
        result.append({
            "name": name,
            "total_mentions": len(mentions),
            "paper_mentions": stats["paper_mentions"],
            "video_mentions": stats["video_mentions"],
            "max_importance": round(max(mentions), 3),
            "avg_importance": round(sum(mentions) / len(mentions), 3),
            # 综合得分：同时考虑重要性和出现频次
            "score": round(max(mentions) * 0.6 + (sum(mentions) / len(mentions)) * 0.4, 3),
        })

    # 按综合得分降序
    result.sort(key=lambda x: x["score"], reverse=True)

    # ── 3. 重要性分布 ────────────────────────────────────────────
    all_avg = [e["avg_importance"] for e in result]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    importance_distribution = {
        f">={t}": sum(1 for s in all_avg if s >= t)
        for t in thresholds
    }

    output = {
        "summary": {
            "total_unique_entities": len(result),
            "avg_importance_overall": round(sum(all_avg) / len(all_avg), 3),
        },
        "importance_distribution": importance_distribution,
        "entities": result
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── 4. 打印摘要 ──────────────────────────────────────────────
    print(f"唯一实体总数: {len(result)}")
    print(f"全局平均重要性: {output['summary']['avg_importance_overall']}")
    print(f"\n重要性分布（按 avg_importance）:")
    for t, count in importance_distribution.items():
        bar = "█" * min(count, 50)
        print(f"  avg_importance {t}: {bar} ({count} 个实体)")

    print(f"\nTop 20 实体（综合得分）:")
    for e in result[:20]:
        cross = "✓" if e["paper_mentions"] > 0 and e["video_mentions"] > 0 else " "
        print(f"  [{cross}] {e['name']:<40} score={e['score']:.3f}  "
              f"avg={e['avg_importance']:.3f}  max={e['max_importance']:.3f}  "
              f"mentions={e['total_mentions']}(论文{e['paper_mentions']}+视频{e['video_mentions']})")

    return output


if __name__ == "__main__":
    analyze_entity_importance(
        "data/graph_tmp/GRPO/entity_nodes.json",
        "data/graph_tmp/GRPO/importance_stats.json"
    )