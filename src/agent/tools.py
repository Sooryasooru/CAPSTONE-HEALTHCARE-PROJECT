"""HAIP Agent Tools.

Each tool wraps an existing HAIP engine function and exposes it to the
LangGraph agent through a typed, documented interface. Tools are the ONLY
way the agent can touch data — this is the "whitelisted tools" boundary
from the Week 1 architecture review. The agent cannot run arbitrary code;
it can only call these four functions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DOCTORS_CSV = Path(__file__).resolve().parents[2] / "data" / "samples" / "hospital_doctors_large.csv"


@tool
def search_guidelines(question: str) -> str:
    """Search the hospital's clinical guideline knowledge base (RAG).

    Use this for any clinical or protocol question: sepsis screening,
    pneumonia discharge criteria, delirium management, medication protocols,
    etc. Returns an evidence-grounded answer with citations.

    Args:
        question: A natural-language clinical question.
    """
    # Imported lazily so the agent process starts fast and the heavy
    # embedding models load only when this tool is first used.
    from src.rag.rag_engine_langchain import LangChainRAGEngine

    engine = LangChainRAGEngine()
    result = engine.answer(question)
    citations = "; ".join(result.get("citations", [])) or "none"
    return f"{result['answer']}\n\n(Sources: {citations})"


@tool
def get_hospital_kpis() -> str:
    """Get the hospital's current key performance indicators (KPIs).

    Use this when asked about hospital-level metrics: mortality rate,
    discharge-against-medical-advice rate, ICU sepsis rate, ICU readmission
    rate, or comorbidity burden. Returns each KPI with its value and unit.
    """
    from src.analytics import kpis as kpi_module

    # Compute each KPI independently so one broken metric doesn't sink the
    # whole tool. A production agent should degrade gracefully: return the
    # KPIs that work, and note the ones that are unavailable.
    kpi_fns = [
        kpi_module.mortality_rate,
        kpi_module.dama_rate,
        kpi_module.icu_sepsis_rate,
        kpi_module.icu_readmission_rate,
        kpi_module.comorbidity_burden,
    ]

    available, unavailable = [], []
    for fn in kpi_fns:
        try:
            k = fn()
            available.append(f"- {k['label']}: {k['value']}{k['unit']}")
        except Exception as exc:  # noqa: BLE001 - we deliberately swallow to degrade gracefully
            logger.warning("KPI %s unavailable: %s", fn.__name__, exc)
            unavailable.append(fn.__name__)

    result = "Current hospital KPIs:\n" + "\n".join(available)
    if unavailable:
        result += f"\n\n(Note: {len(unavailable)} metric(s) are currently unavailable.)"
    return result


@tool
def forecast_admissions(months_ahead: int = 6) -> str:
    """Forecast future monthly hospital admission volume.

    Use this for planning and capacity questions: "how many admissions
    next quarter", "what's the trend", "will we need more beds". Uses a
    Holt-Winters seasonal model on historical encounter data.

    Args:
        months_ahead: Number of future months to project (default 6).
    """
    from src.prediction.forecast import forecast_admissions as _forecast

    try:
        df = _forecast(months_ahead=months_ahead)
    except ValueError as exc:
        return f"Forecast unavailable: {exc}"

    lines = [
        f"- {row['month'].strftime('%b %Y')}: {int(row['forecast'])} admissions"
        for _, row in df.iterrows()
    ]
    return f"Forecast for the next {months_ahead} months:\n" + "\n".join(lines)


@tool
def get_doctor_stats(department: str = "") -> str:
    """Get provider/department statistics from the hospital's doctor dataset.

    Use this for questions about doctors, departments, or provider counts:
    "how many cardiologists", "which departments do we have", "provider
    headcount". If a department is given, results are filtered to it.

    Args:
        department: Optional department name to filter by (e.g. "Cardiology").
                    Leave empty for a hospital-wide summary.
    """
    if not DOCTORS_CSV.exists():
        return "Doctor dataset not found."

    df = pd.read_csv(DOCTORS_CSV)

    # Find the department column flexibly (schema may vary by hospital).
    dept_col = next(
        (c for c in df.columns if c.lower() in {"department", "dept", "specialty"}),
        None,
    )
    if dept_col is None:
        return f"Loaded {len(df)} providers, but no department column was found."

    if department:
        subset = df[df[dept_col].str.contains(department, case=False, na=False)]
        return f"{len(subset)} providers found in departments matching '{department}'."

    counts = df[dept_col].value_counts().head(10)
    lines = [f"- {dept}: {n} providers" for dept, n in counts.items()]
    return (
        f"Total providers: {len(df)} across {df[dept_col].nunique()} departments.\n"
        f"Top departments:\n" + "\n".join(lines)
    )


# The whitelist. The agent is constructed with exactly these tools — nothing else.
ALL_TOOLS = [
    search_guidelines,
    get_hospital_kpis,
    forecast_admissions,
    get_doctor_stats,
]