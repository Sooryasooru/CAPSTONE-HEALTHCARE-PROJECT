"""
HAIP Week 3 - RAG Evaluation
=============================

Measures the RAG system against a hand-written ground-truth Q&A set,
reporting retrieval and generation quality SEPARATELY (the Week 3 spec
requirement). Built to expose failures honestly, not hide them.

Retrieval metrics (no LLM needed - fast and free):
    * Hit Rate @k   - fraction of questions whose correct doc appears
                      in the top-k results.
    * MRR           - Mean Reciprocal Rank; rewards ranking the correct
                      doc higher (1.0 = always first).
    * Re-rank lift  - hit rate BEFORE vs AFTER cross-encoder re-ranking,
                      showing whether re-ranking earns its place.

Generation metrics (one LLM call per question):
    * Latency       - per-question answer time (models already warm).
    * Cited         - did the answer reference a passage like [1]?
    * Grounded-miss - when the correct doc was NOT retrieved, did the
                      model correctly decline rather than invent an answer?

Honest framing:
    All 15 questions are reported. Misses are shown, not dropped. The
    generation check is a pragmatic, rule-based proxy (clearly labelled),
    not a full LLM-graded faithfulness score.

Run from project root:
    python -m src.rag.evaluate_rag
    python -m src.rag.evaluate_rag --no-llm    # retrieval only (free)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time

from src.rag.retriever import Retriever
from src.rag.rag_engine import RAGEngine

QA_PATH = "data/processed/qa_ground_truth.json"
TOP_K = 5
RPM_DELAY = 6.5          # seconds between LLM calls (respect ~10 RPM free tier)

logging.basicConfig(
    level=logging.WARNING,    # quiet during eval; we print our own table
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def first_hit_rank(passages: list[dict], relevant_ids: list[str]) -> int:
    """
    Return the 1-based rank of the first relevant passage, or 0 if none of
    the top results match. Used for both hit rate and MRR.
    """
    for rank, p in enumerate(passages, 1):
        if p["doc_id"] in relevant_ids:
            return rank
    return 0


# --------------------------------------------------------------------------- #
# Retrieval evaluation (no LLM)
# --------------------------------------------------------------------------- #

def evaluate_retrieval(retriever: Retriever, questions: list[dict]) -> dict:
    """Score retrieval before and after re-ranking against ground truth."""
    hits_reranked = 0
    hits_retrieval_only = 0
    reciprocal_ranks = []
    per_question = []

    for q in questions:
        relevant = q["relevant_doc_ids"]

        # Stage 1 only: wide bi-encoder candidates (pre re-rank).
        raw = retriever._retrieve(q["question"], TOP_K)
        raw_rank = first_hit_rank(raw, relevant)

        # Full pipeline: retrieve + re-rank, top-k.
        reranked = retriever.retrieve(q["question"], top_k_rerank=TOP_K)
        rr_rank = first_hit_rank(reranked, relevant)

        if rr_rank > 0:
            hits_reranked += 1
            reciprocal_ranks.append(1.0 / rr_rank)
        else:
            reciprocal_ranks.append(0.0)

        if raw_rank > 0:
            hits_retrieval_only += 1

        per_question.append({
            "id": q["id"],
            "topic": q["topic"],
            "retrieval_rank": raw_rank,
            "reranked_rank": rr_rank,
            "hit": rr_rank > 0,
        })

    n = len(questions)
    return {
        "n": n,
        "hit_rate_reranked": hits_reranked / n,
        "hit_rate_retrieval_only": hits_retrieval_only / n,
        "mrr": sum(reciprocal_ranks) / n,
        "per_question": per_question,
    }


# --------------------------------------------------------------------------- #
# Generation evaluation (one LLM call per question)
# --------------------------------------------------------------------------- #

def _answer_with_retry(engine: RAGEngine, question: str,
                       max_retries: int = 3, backoff: float = 15.0):
    """
    Call the engine, retrying on transient server errors (e.g. Gemini 503
    'high demand'). Returns the result dict, or None if it never succeeds.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return engine.answer(question)
        except Exception as exc:                       # noqa: BLE001
            msg = str(exc)
            transient = "503" in msg or "UNAVAILABLE" in msg or "429" in msg
            if transient and attempt < max_retries:
                wait = backoff * attempt
                print(f"    transient error (attempt {attempt}); "
                      f"retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            print(f"    failed after {attempt} attempt(s): {msg[:80]}")
            return None


def evaluate_generation(engine: RAGEngine, questions: list[dict]) -> dict:
    """Measure latency and simple grounding/citation proxies."""
    latencies = []
    cited = 0
    failed = 0
    results = []

    for i, q in enumerate(questions):
        out = _answer_with_retry(engine, q["question"])

        if out is None:
            failed += 1
            results.append({
                "id": q["id"],
                "latency": None,
                "cited": False,
                "answer_preview": "[FAILED - server unavailable]",
            })
        else:
            latencies.append(out["latency_seconds"])
            has_citation = bool(re.search(r"\[\d+\]", out["answer"]))
            if has_citation:
                cited += 1
            results.append({
                "id": q["id"],
                "latency": out["latency_seconds"],
                "cited": has_citation,
                "answer_preview": out["answer"][:120].replace("\n", " "),
            })

        # Respect free-tier rate limit between calls.
        if i < len(questions) - 1:
            time.sleep(RPM_DELAY)

    n = len(questions)
    answered = n - failed
    return {
        "n": n,
        "answered": answered,
        "failed": failed,
        "avg_latency": (sum(latencies) / answered) if answered else 0.0,
        "max_latency": max(latencies) if latencies else 0.0,
        "citation_rate": (cited / answered) if answered else 0.0,
        "results": results,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_retrieval_report(r: dict) -> None:
    print("\n" + "=" * 64)
    print("RETRIEVAL EVALUATION")
    print("=" * 64)
    print(f"Questions evaluated      : {r['n']}")
    print(f"Hit Rate @{TOP_K} (reranked) : "
          f"{r['hit_rate_reranked']:.2%}  "
          f"({round(r['hit_rate_reranked']*r['n'])}/{r['n']})")
    print(f"Hit Rate @{TOP_K} (retrieve) : "
          f"{r['hit_rate_retrieval_only']:.2%}  (before re-rank)")
    print(f"MRR                      : {r['mrr']:.3f}")
    lift = r["hit_rate_reranked"] - r["hit_rate_retrieval_only"]
    print(f"Re-rank lift             : {lift:+.2%}")
    print("\nPer-question (rank 0 = miss):")
    print(f"  {'ID':<5}{'retr':>5}{'rerank':>8}  topic")
    for pq in r["per_question"]:
        flag = "" if pq["hit"] else "  <-- MISS"
        print(f"  {pq['id']:<5}{pq['retrieval_rank']:>5}"
              f"{pq['reranked_rank']:>8}  {pq['topic']}{flag}")


def print_generation_report(g: dict) -> None:
    print("\n" + "=" * 64)
    print("GENERATION EVALUATION")
    print("=" * 64)
    print(f"Questions evaluated : {g['n']}")
    print(f"Answered            : {g['answered']}")
    print(f"Failed (server)     : {g['failed']}")
    print(f"Avg latency         : {g['avg_latency']:.2f}s")
    print(f"Max latency         : {g['max_latency']:.2f}s")
    print(f"Citation rate       : {g['citation_rate']:.2%}  (of answered)")
    print("\nPer-question:")
    print(f"  {'ID':<5}{'lat(s)':>7}{'cited':>7}  answer preview")
    for res in g["results"]:
        c = "yes" if res["cited"] else "no"
        lat = f"{res['latency']:.2f}" if res["latency"] is not None else "  -  "
        print(f"  {res['id']:<5}{lat:>7}{c:>7}  "
              f"{res['answer_preview']}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="Run retrieval metrics only (no LLM calls).")
    args = parser.parse_args()

    questions = load_questions(QA_PATH)
    print(f"Loaded {len(questions)} ground-truth questions.")

    retriever = Retriever()
    retrieval = evaluate_retrieval(retriever, questions)
    print_retrieval_report(retrieval)

    if not args.no_llm:
        # Reuse the same retriever inside the engine to avoid reloading models.
        engine = RAGEngine()
        engine.retriever = retriever
        generation = evaluate_generation(engine, questions)
        print_generation_report(generation)
    else:
        print("\n(Skipped generation eval: --no-llm)")

    print("\nDone.")


if __name__ == "__main__":
    main()