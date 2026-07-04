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


def run_engine(engine_name: str, question: str) -> dict:
    """Call the chosen engine and return its digest.

    Unknown/not-yet-wired engines return an explicit placeholder so the
    contract stays honest — no silent failures.
    """
    if engine_name == "analytics":
        return _analytics_answer(question)

    # prediction and rag are wired in later steps.
    return {
        "engine": engine_name,
        "answer_type": "not_implemented",
        "note": f"'{engine_name}' engine not yet connected.",
    }
