"""HAIP - Retrieval configuration benchmark.

Compares retrieval configurations on the same golden set and index:
  - top_k in {3, 5, 10}
  - cross-encoder re-rank ON vs OFF

Justifies the production choice (k=5, re-rank on) with measured numbers
instead of assertion. Uses the existing FAISS index - no re-embedding,
no API calls.
"""
from __future__ import annotations
import csv, json, math, sys
sys.path.insert(0, ".")
from src.rag.retriever import Retriever

GT = "data/processed/qa_ground_truth.json"
OUT = "data/eval/retrieval_benchmark.csv"


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def score(gold: set, got: list, k: int) -> dict:
    rels = [1 if d in gold else 0 for d in got]
    hit = 1 if any(rels) else 0
    return {
        "hit": hit,
        "mrr": 1 / (rels.index(1) + 1) if hit else 0.0,
        "ndcg": (dcg(rels) / dcg(sorted(rels, reverse=True))) if hit else 0.0,
        "p_at_1": sum(rels[:1]) / 1 if rels else 0.0,
        "p_at_k": sum(rels) / k,
        "recall": len(gold & set(got)) / len(gold) if gold else 0.0,
    }


def run_config(r: Retriever, gt: list, k: int, rerank: bool) -> dict:
    agg = {"hit": 0.0, "mrr": 0.0, "ndcg": 0.0, "p_at_1": 0.0, "p_at_k": 0.0, "recall": 0.0}
    for q in gt:
        gold = set(q["relevant_doc_ids"])
        if rerank:
            hits = r.retrieve(q["question"], top_k_rerank=k)
        else:
            # stage-1 only: bi-encoder FAISS search, no cross-encoder
            hits = r._retrieve(q["question"], k)
        got = [h["doc_id"] for h in hits]
        s = score(gold, got, k)
        for m in agg:
            agg[m] += s[m]
    n = len(gt)
    return {m: round(v / n, 3) for m, v in agg.items()}


def main():
    gt = json.load(open(GT))["questions"]
    r = Retriever(index_path="data/processed/faiss.index",
                  metadata_path="data/processed/chunks.pkl")

    configs = [(k, rr) for rr in (True, False) for k in (3, 5, 10)]
    rows = []
    for k, rr in configs:
        res = run_config(r, gt, k, rr)
        label = f"k={k}, rerank={'ON' if rr else 'OFF'}"
        rows.append({"config": label, "k": k, "rerank": "ON" if rr else "OFF", **res})
        print(f"  done: {label}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    hdr = f"{'Config':<22}{'Hit':>7}{'MRR':>8}{'NDCG':>8}{'P@1':>8}{'P@k':>8}{'Recall':>8}"
    print("\n" + "=" * len(hdr))
    print(f"RETRIEVAL CONFIG BENCHMARK  (n={len(gt)})")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r_ in rows:
        print(f"{r_['config']:<22}{r_['hit']:>7}{r_['mrr']:>8}{r_['ndcg']:>8}"
              f"{r_['p_at_1']:>8}{r_['p_at_k']:>8}{r_['recall']:>8}")
    print("=" * len(hdr))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
