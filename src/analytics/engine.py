"""Analytics engine — executes Gold-layer queries and returns DataFrames.

The Gold views already aggregate the data, so each function here is a thin
wrapper: it runs a SELECT against one Gold view and returns the result as a
pandas DataFrame. The dashboard and reports call these functions and never
touch SQL or the database connection directly.
"""

import pandas as pd
from etl.utils import get_engine, get_logger

logger = get_logger(__name__)

# --- Query definitions: one SELECT per Gold view ---------------------------
QUERIES = {
    "outcome_distribution":   "SELECT * FROM gold.outcome_distribution;",
    "comorbidity_prevalence": "SELECT * FROM gold.comorbidity_prevalence;",
    "billing_by_condition":   "SELECT * FROM gold.billing_by_condition;",
    "icu_severity_summary":   "SELECT * FROM gold.icu_severity_summary;",
    "lab_test_summary":       "SELECT * FROM gold.lab_test_summary;",
    "admissions_summary":     "SELECT * FROM gold.admissions_summary;",
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
    """Patient outcome breakdown: Discharge / Expiry / DAMA."""
    return _run_query("outcome_distribution")


def get_comorbidity_prevalence() -> pd.DataFrame:
    """Prevalence of each comorbidity (CAD, HTN, DM, ...)."""
    return _run_query("comorbidity_prevalence")


def get_billing_by_condition() -> pd.DataFrame:
    """Billing totals/averages grouped by condition."""
    return _run_query("billing_by_condition")


def get_icu_severity_summary() -> pd.DataFrame:
    """ICU severity metrics (SOFA bands, sepsis rates, ...)."""
    return _run_query("icu_severity_summary")


def get_lab_test_summary() -> pd.DataFrame:
    """Summary statistics across lab tests."""
    return _run_query("lab_test_summary")


def get_admissions_summary() -> pd.DataFrame:
    """Admission counts and patterns."""
    return _run_query("admissions_summary")