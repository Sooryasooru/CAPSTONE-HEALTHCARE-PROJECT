"""Healthcare KPIs — named clinical metrics computed from engine outputs.

Each KPI function returns a self-describing dict: {value, label, unit}.
get_all_kpis() collects them into a list for the dashboard to render.
The engine handles data access; this layer handles clinical meaning.
"""

from etl.utils import get_logger
from analytics import engine

logger = get_logger(__name__)


def mortality_rate() -> dict:
    """In-hospital mortality rate = Expiry patients / all patients.

    A core hospital quality measure. Lower is better.
    """
    df = engine.get_outcome_distribution()
    row = df.loc[df["outcome"] == "Expiry", "pct"]
    value = float(row.iloc[0]) if not row.empty else 0.0
    return {"value": value, "label": "Mortality Rate", "unit": "%"}


def dama_rate() -> dict:
    """Discharge Against Medical Advice rate = DAMA patients / all patients.

    A care-gap signal: patients leaving before advised. Lower is better.
    """
    df = engine.get_outcome_distribution()
    row = df.loc[df["outcome"] == "Dama", "pct"]
    value = float(row.iloc[0]) if not row.empty else 0.0
    return {"value": value, "label": "DAMA Rate", "unit": "%"}


def icu_sepsis_rate() -> dict:
    """ICU sepsis rate = total sepsis cases / total ICU patients.

    Sepsis is a leading driver of ICU mortality. Higher is worse.
    """
    df = engine.get_icu_severity_summary()
    total_patients = df["patients"].sum()
    total_sepsis = df["sepsis_cases"].sum()
    value = float(round(100 * total_sepsis / total_patients, 1)) if total_patients else 0.0
    return {"value": value, "label": "ICU Sepsis Rate", "unit": "%"}


def icu_readmission_rate() -> dict:
    """30-day ICU readmission rate = 30-day readmissions / total ICU patients.

    A standard CMS quality measure; reflects care continuity. Lower is better.
    """
    df = engine.get_icu_severity_summary()
    total_patients = df["patients"].sum()
    total_readmits = df["readmissions_30d"].sum()
    value = float(round(100 * total_readmits / total_patients, 1)) if total_patients else 0.0
    return {"value": value, "label": "30-Day ICU Readmission Rate", "unit": "%"}


def comorbidity_burden() -> dict:
    """Average comorbidity burden = mean prevalence across tracked conditions.

    Indicates how multi-morbid the cohort is. Higher means sicker patients.
    """
    df = engine.get_comorbidity_prevalence()
    value = float(round(df["prevalence_pct"].mean(), 1)) if not df.empty else 0.0
    return {"value": value, "label": "Avg Comorbidity Burden", "unit": "%"}


def get_all_kpis() -> list[dict]:
    """Return every KPI as a list of dicts — the dashboard's single entry point.

    Returns:
        List of {value, label, unit} dicts, one per KPI.
    """
    kpis = [
        mortality_rate(),
        dama_rate(),
        icu_sepsis_rate(),
        icu_readmission_rate(),
        comorbidity_burden(),
    ]
    logger.info("Computed %d KPIs", len(kpis))
    return kpis