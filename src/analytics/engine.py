"""Analytics engine — executes Gold-layer queries and returns DataFrames.

The Gold views already aggregate the data, so each function here is a thin
wrapper: it runs a SELECT against one Gold view and returns the result as a
pandas DataFrame. The dashboard and reports call these functions and never
touch SQL or the database connection directly.

Updated for the connected Synthea schema — views now JOIN across tables.
"""

import pandas as pd
from etl.utils import get_engine, get_logger

logger = get_logger(__name__)

# --- Query definitions: one SELECT per Gold view ---------------------------
QUERIES = {
    "outcome_distribution":   "SELECT * FROM gold.outcome_distribution;",
    "comorbidity_prevalence": "SELECT * FROM gold.comorbidity_prevalence;",
    "cost_by_condition":      "SELECT * FROM gold.cost_by_condition;",
    "admissions_summary":     "SELECT * FROM gold.admissions_summary;",
    "revenue_by_month":       "SELECT * FROM gold.revenue_by_month;",
    "patient_360":            "SELECT * FROM gold.patient_360;",
}


def _run_query(name: str) -> pd.DataFrame:
    """Run a named Gold query and return a DataFrame.

    Args:
        name: Key into QUERIES (the Gold view name).

    Returns:
        DataFrame with columns matching the Gold view.
    """
    sql = QUERIES[name]
    engine = get_engine()
    logger.info("Running analytics query: %s", name)
    df = pd.read_sql(sql, engine)
    logger.info("Query '%s' returned %d rows", name, len(df))
    return df


# --- Public functions: one per analytic ------------------------------------
def get_outcome_distribution() -> pd.DataFrame:
    """Patient outcome breakdown: Alive / Deceased."""
    return _run_query("outcome_distribution")


def get_comorbidity_prevalence() -> pd.DataFrame:
    """Prevalence of each common chronic condition (joined from conditions)."""
    return _run_query("comorbidity_prevalence")


def get_cost_by_condition() -> pd.DataFrame:
    """Cost per condition (conditions JOINed to encounters)."""
    return _run_query("cost_by_condition")


def get_admissions_summary() -> pd.DataFrame:
    """Admission counts and patterns by month + encounter class."""
    return _run_query("admissions_summary")


def get_revenue_by_month() -> pd.DataFrame:
    """Monthly revenue from encounters (total, payer-paid, patient-paid)."""
    return _run_query("revenue_by_month")


def get_patient_360() -> pd.DataFrame:
    """One row per patient — journey rolled up across four tables."""
    return _run_query("patient_360")