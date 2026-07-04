"""Engine dispatch — turns a routing decision into a real engine answer.

Each engine returns a compact, JSON-safe digest (NOT raw dumps). A router
answer is a summary for a hospital user, not a 4000-row table.

Engines are added one at a time and tested in isolation:
  analytics  -> live (admissions digest)
  prediction -> TODO
  rag        -> TODO (loads a model; added last, carefully)
"""
import pandas as pd

from analytics import engine as analytics_engine
from prediction.forecast import forecast_admissions

# RAG engine is heavy (loads an embedding model). Cache it module-level so it
# loads ONCE, on the first RAG question — never per request, never at startup.
_rag_engine = None


def _analytics_answer(question: str) -> dict:
    """Digest of admissions: overall totals by encounter class, recent window.

    Returns a small summary, not the full 4000+ row table.
    """
    df = analytics_engine.get_admissions_summary()

    # Totals by encounter class across the whole dataset.
    by_class = (
        df.groupby("encounterclass")["total_admissions"]
        .sum()
        .sort_values(ascending=False)
    )

    # Most recent 12 months of activity (data has synthetic-old dates).
    recent = df.sort_values("admission_month").tail(12)
    recent_total = int(recent["total_admissions"].sum())

    return {
        "engine": "analytics",
        "answer_type": "admissions_summary",
        "total_admissions_all_time": int(df["total_admissions"].sum()),
        "by_encounter_class": {k: int(v) for k, v in by_class.items()},
        "recent_12mo_admissions": recent_total,
        "note": "Digest of admissions summary. Full detail lives in the dashboard.",
    }


def _prediction_answer(question: str) -> dict:
    """6-month admissions forecast (Holt-Winters on the monthly series).

    Honest disclosure: this is a statistical projection on synthetic data,
    not a guarantee. Values are point forecasts, no confidence band shown.
    """
    df = forecast_admissions(6)
    points = [
        {"month": row["month"].strftime("%Y-%m"), "forecast": int(row["forecast"])}
        for _, row in df.iterrows()
    ]
    return {
        "engine": "prediction",
        "answer_type": "admissions_forecast",
        "horizon_months": len(points),
        "forecast": points,
        "method": "Holt-Winters exponential smoothing",
        "note": "Point forecast on synthetic data. A projection, not a guarantee.",
    }


def _get_rag_engine():
    """Construct the RAG engine once and cache it. Loads the model lazily."""
    global _rag_engine
    if _rag_engine is None:
        from rag.rag_engine import RAGEngine
        _rag_engine = RAGEngine()
    return _rag_engine


def _rag_answer(question: str) -> dict:
    """Answer from the guideline knowledge base (retrieval + LLM).

    Digests the engine output: keeps the answer, citations, and latency;
    trims full passages to a count. Fails gracefully if the LLM key is absent
    so the endpoint never hard-crashes during a demo.
    """
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        return {
            "engine": "rag",
            "answer_type": "unavailable",
            "note": "RAG needs GEMINI_API_KEY set. Retrieval works, but the "
                    "answer step is disabled without it.",
        }

    try:
        engine = _get_rag_engine()
        result = engine.answer(question)
        return {
            "engine": "rag",
            "answer_type": "grounded_answer",
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "passages_used": len(result.get("passages", [])),
            "latency_seconds": result.get("latency_seconds"),
            "note": "Answer grounded in retrieved guidelines (WHO/CDC/NICE corpus).",
        }
    except Exception as exc:
        return {
            "engine": "rag",
            "answer_type": "error",
            "note": f"RAG failed: {type(exc).__name__}: {exc}",
        }


def run_engine(engine_name: str, question: str) -> dict:
    """Call the chosen engine and return its digest.

    Unknown/not-yet-wired engines return an explicit placeholder so the
    contract stays honest — no silent failures.
    """
    if engine_name == "analytics":
        return _analytics_answer(question)

    if engine_name == "prediction":
        return _prediction_answer(question)

    if engine_name == "rag":
        return _rag_answer(question)

    # prediction and rag are wired in later steps.
    return {
        "engine": engine_name,
        "answer_type": "not_implemented",
        "note": f"'{engine_name}' engine not yet connected.",
    }
