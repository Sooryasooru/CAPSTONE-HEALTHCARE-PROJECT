"""Load layer: write extracted DataFrames into bronze tables.

Idempotent — truncates each bronze table before insert, so re-runs
never duplicate rows.
"""

from sqlalchemy import text

from etl.extract import extract_tabular
from etl.utils import get_engine, get_logger

logger = get_logger(__name__)

BRONZE_COLUMNS = {
    "patients": [
        "sno", "mrd_no", "doa", "dod", "age", "gender", "rural",
        "type_of_admission", "month_year", "duration_of_stay",
        "duration_of_icu_stay", "outcome", "smoking", "alcohol", "dm", "htn",
        "cad", "prior_cmp", "ckd", "hb", "tlc", "platelets", "glucose", "urea",
        "creatinine", "bnp", "raised_cardiac_enzymes", "ef", "severe_anaemia",
        "anaemia", "stable_angina", "acs", "stemi", "atypical_chestpain",
        "heart_failure", "hfref", "hfnef", "valvular", "chb", "sss", "aki",
        "cva_infract", "cva_bleed", "af", "vt", "psvt", "congenital", "uti",
        "neuro_cardiogenic_syncope", "orthostatic", "infective_endocarditis",
        "dvt", "cardiogenic_shock", "shock", "pulmonary_embolism",
        "chest_infection",
    ],
    "mortality": ["sno", "mrd", "age", "gender", "rural_urban", "date_of_brought_dead"],
    "billing": [
        "patient_id", "age", "gender", "condition", "procedure", "cost",
        "length_of_stay", "readmission", "outcome", "satisfaction",
    ],
    "labs": [
        "date", "test_name", "result", "unit", "reference_range", "status",
        "comment", "min_reference", "max_reference", "unit_description",
        "recommended_followup",
    ],
}


def load_to_bronze() -> None:
    """Load all extracted datasets into bronze tables."""
    engine = get_engine()
    datasets = extract_tabular()

    for name, df in datasets.items():
        if name in BRONZE_COLUMNS:
            df = df.copy()
            df.columns = BRONZE_COLUMNS[name]

        try:
            with engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE bronze.{name}"))
                df.to_sql(name, conn, schema="bronze", if_exists="append", index=False)
            logger.info("Loaded %d rows into bronze.%s", len(df), name)
        except Exception as exc:
            logger.error("Failed to load bronze.%s: %s", name, exc)
            raise


if __name__ == "__main__":
    load_to_bronze()
    