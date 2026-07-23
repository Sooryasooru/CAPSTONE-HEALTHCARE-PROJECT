"""HAIP - Retrieval evaluation (Hit Rate, MRR, NDCG@k, Precision@k, Recall@k)."""
from __future__ import annotations
import json, math, sys, csv
sys.path.insert(0, ".")
from src.rag.retriever import Retriever

import os
GT = os.getenv("HAIP_GOLDEN_SET", "data/processed/qa_ground_truth.json")
OUT = os.getenv("HAIP_EVAL_OUT", "data/eval/retrieval_results.csv")
K = 5

def dcg(rels): return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

def main():
    gt = json.load(open(GT))["questions"]
    r = Retriever(index_path="data/processed/faiss.index",
                  metadata_path="data/processed/chunks.pkl")
    rows, agg = [], {"hit": 0, "mrr": 0.0, "ndcg": 0.0, "p": 0.0, "rec": 0.0}
    for q in gt:
        gold = set(q["relevant_doc_ids"])
        hits = r.retrieve(q["question"], top_k_rerank=K)
        got = [h["doc_id"] for h in hits]
        rels = [1 if d in gold else 0 for d in got]
        hit = 1 if any(rels) else 0
        rr = 1 / (rels.index(1) + 1) if hit else 0.0
        ndcg = dcg(rels) / dcg(sorted(rels, reverse=True)) if hit else 0.0
        prec = sum(rels) / K
        p1 = sum(rels[:1]) / 1
        p3 = sum(rels[:3]) / 3
        rec = len(gold & set(got)) / len(gold) if gold else 0.0
        agg["hit"] += hit; agg["mrr"] += rr; agg["ndcg"] += ndcg
        agg["p"] += prec; agg["rec"] += rec
        agg["p1"] = agg.get("p1", 0.0) + p1
        agg["p3"] = agg.get("p3", 0.0) + p3
        rows.append({"id": q["id"], "question": q["question"], "topic": q["topic"],
                     "hit": hit, "mrr": round(rr, 3), "ndcg": round(ndcg, 3),
                     "precision_at_1": round(p1, 3), "precision_at_3": round(p3, 3),
                     "precision_at_k": round(prec, 3), "recall_at_k": round(rec, 3),
                     "retrieved_doc_ids": "|".join(got)})
    n = len(gt)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n" + "=" * 55)
    print(f"RETRIEVAL EVALUATION  (n={n}, k={K})")
    print("=" * 55)
    print(f"  Hit Rate@{K}    : {agg['hit']/n*100:.1f}%")
    print(f"  MRR           : {agg['mrr']/n:.3f}")
    print(f"  NDCG@{K}       : {agg['ndcg']/n:.3f}")
    print(f"  Precision@1   : {agg['p1']/n:.3f}")
    print(f"  Precision@3   : {agg['p3']/n:.3f}")
    print(f"  Precision@{K}  : {agg['p']/n:.3f}")
    print(f"  Recall@{K}     : {agg['rec']/n:.3f}")
    print("=" * 55)
    print(f"Saved -> {OUT}")

if __name__ == "__main__":
    main()
