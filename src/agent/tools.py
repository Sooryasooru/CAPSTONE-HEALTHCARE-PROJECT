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
    import pandas as pd
    import os
    CSV = "/app/data/samples/hospital_dataset.csv"
    if not os.path.exists(CSV):
        CSV = "data/samples/hospital_dataset.csv"
    try:
        df = pd.read_csv(CSV)
    except Exception as exc:  # noqa: BLE001
        logger.warning("KPI data unavailable: %s", exc)
        return "Hospital KPI data is currently unavailable."
    lines = [f"- Total encounters: {len(df):,}"]
    if "patient_id" in df:
        lines.append(f"- Unique patients: {df['patient_id'].nunique():,}")
    if "readmitted" in df:
        lines.append(f"- 30-day readmission rate: {100*pd.to_numeric(df['readmitted'],errors='coerce').mean():.1f}%")
    if "outcome" in df:
        lines.append(f"- Mortality rate: {100*(df['outcome'].astype(str).str.lower()=='deceased').mean():.1f}%")
    if "treatment_cost" in df:
        lines.append(f"- Avg treatment cost: ${pd.to_numeric(df['treatment_cost'],errors='coerce').mean():,.0f}")
    if "length_of_stay" in df:
        lines.append(f"- Avg length of stay: {pd.to_numeric(df['length_of_stay'],errors='coerce').mean():.1f} days")
    if "patient_satisfaction" in df:
        lines.append(f"- Avg patient satisfaction: {pd.to_numeric(df['patient_satisfaction'],errors='coerce').mean():.1f}/10")
    return "Current hospital KPIs:\n" + "\n".join(lines)
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
    import pandas as pd
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import os
    CSV = "/app/data/samples/hospital_dataset.csv"
    if not os.path.exists(CSV):
        CSV = "data/samples/hospital_dataset.csv"
    try:
        df = pd.read_csv(CSV)
        df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
        monthly = df.dropna(subset=["admission_date"]).set_index("admission_date").resample("MS").size()
        if len(monthly) < 4:
            return "Forecast unavailable: not enough monthly history."
        model = ExponentialSmoothing(monthly, trend="add", seasonal=None).fit()
        fc = model.forecast(months_ahead).round().astype(int)
    except Exception as exc:  # noqa: BLE001
        return f"Forecast unavailable: {exc}"
    lines = [f"- {idx.strftime('%b %Y')}: {int(val)} admissions" for idx, val in fc.items()]
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