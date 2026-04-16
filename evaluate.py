#!/usr/bin/env python
"""
Offline RAGAS evaluation script for the Academic Assistant.

Reads logged (query, contexts, answer) triplets from SQLite and evaluates:
  - Faithfulness:      Does the answer stick to the retrieved context?
  - Answer Relevancy:  Does the answer address the question?
  - Reranker stats:    Distribution of retrieval scores (free, no LLM needed).

Usage:
    python evaluate.py                          # evaluate all samples
    python evaluate.py --limit 50               # last 50 samples
    python evaluate.py --output results.json    # save scores to file
    python evaluate.py --no-ragas               # reranker stats only (fast)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(db_path: str, limit: int | None = None) -> list[dict]:
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    q = "SELECT query, contexts, answer, reranker_scores, resource_types FROM eval_log ORDER BY id DESC"
    if limit:
        q += f" LIMIT {limit}"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [
        {
            "question":        row[0],
            "contexts":        json.loads(row[1]),
            "answer":          row[2],
            "reranker_scores": json.loads(row[3]),
            "resource_types":  json.loads(row[4]),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Reranker statistics (free — no LLM call)
# ---------------------------------------------------------------------------

def print_reranker_stats(samples: list[dict]) -> None:
    top1 = [s["reranker_scores"][0] for s in samples if s["reranker_scores"]]
    all_scores = [sc for s in samples for sc in s["reranker_scores"]]

    print("\n── Reranker Score Statistics ─────────────────────────")
    if top1:
        print(f"  Top-1  mean={statistics.mean(top1):.3f}  "
              f"min={min(top1):.3f}  max={max(top1):.3f}")
    if len(all_scores) > 1:
        print(f"  All    mean={statistics.mean(all_scores):.3f}  "
              f"stdev={statistics.stdev(all_scores):.3f}")

    # Warn if retrieval quality looks poor
    if top1 and statistics.mean(top1) < 0.3:
        print("  ⚠  Mean top-1 score < 0.3 — retrieval quality may be low.")

    # Source type breakdown
    type_counts: dict[str, int] = {}
    for s in samples:
        for t in s["resource_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1
    if type_counts:
        total = sum(type_counts.values())
        breakdown = "  ".join(f"{t}: {c/total:.0%}" for t, c in sorted(type_counts.items()))
        print(f"  Source mix: {breakdown}")


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas(samples: list[dict]) -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings

    from agent.utils import get_llm
    from agent.rag import init_rag, get_embed_fn

    print("\nInitialising models (BGE-M3 + DeepSeek)...")
    init_rag()

    # Wrap BGE-M3 as a LangChain Embeddings for RAGAS answer_relevancy
    _embed = get_embed_fn()

    class BGEEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return _embed(texts)
        def embed_query(self, text: str) -> list[float]:
            return _embed([text])[0]

    ragas_llm = LangchainLLMWrapper(get_llm("deepseek-chat"))
    ragas_emb = LangchainEmbeddingsWrapper(BGEEmbeddings())

    dataset = Dataset.from_dict({
        "question": [s["question"] for s in samples],
        "contexts":  [s["contexts"]  for s in samples],
        "answer":    [s["answer"]    for s in samples],
    })

    print("Running RAGAS (this calls the LLM for each sample)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Academic Assistant RAG quality")
    parser.add_argument(
        "--db", default=os.environ.get("EVAL_LOG_PATH", "./data/eval_log.db"),
        help="Path to eval log SQLite file"
    )
    parser.add_argument("--limit", type=int, default=None, help="Number of recent samples to use")
    parser.add_argument("--output", default=None, help="Save JSON results to this path")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS, show reranker stats only")
    args = parser.parse_args()

    print(f"Loading samples from {args.db} ...")
    samples = load_samples(args.db, args.limit)

    if not samples:
        print("No samples found. Start the server, have a few conversations, then re-run.")
        return

    print(f"  {len(samples)} sample(s) loaded.")
    print_reranker_stats(samples)

    if args.no_ragas:
        return

    scores = run_ragas(samples)

    print("\n── RAGAS Results ──────────────────────────────────────")
    print(f"  Faithfulness:       {scores.get('faithfulness', 'n/a'):.3f}")
    print(f"  Answer Relevancy:   {scores.get('answer_relevancy', 'n/a'):.3f}")
    print(f"  Context Relevancy:  {scores.get('context_relevancy', 'n/a'):.3f}")
    print()
    print("  Faithfulness        1.0 = answer fully grounded in retrieved context")
    print("  Answer Relevancy    1.0 = answer directly addresses the question")
    print("  Context Relevancy   1.0 = retrieved chunks are all relevant to the query")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"\nScores saved to {args.output}")


if __name__ == "__main__":
    main()
