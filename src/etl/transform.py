"""Transform layer: clean + type bronze data into silver tables.

Pattern per source:
    1. read bronze (all TEXT)
    2. map categorical codes to readable labels
    3. coerce numeric / date types (bad values -> NULL)
    4. write to typed silver table (idempotent)
"""

import pandas as pd
from sqlalchemy import text

from etl.utils import get_engine, get_logger

logger = get_logger(__name__)

# Value mappings (codes -> readable labels)

GENDER_MAP = {"M": "Male", "F": "Female"}
RURAL_MAP = {"R": "Rural", "U": "Urban"}
ADMISSION_MAP = {"E": "Emergency", "O": "OPD"}
OUTCOME_MAP = {"DISCHARGE": "Discharge", "EXPIRY": "Expiry", "DAMA": "Dama"}
YESNO_MAP = {"Yes": True, "No": False}

PATIENTS_NUMERIC = [
    "sno", "age", "duration_of_stay", "duration_of_icu_stay",
    "smoking", "alcohol", "dm", "htn", "cad", "prior_cmp", "ckd",
    "hb", "tlc", "platelets", "glucose", "urea", "creatinine", "bnp",
    "raised_cardiac_enzymes", "ef", "severe_anaemia", "anaemia",
    "stable_angina", "acs", "stemi", "atypical_chestpain", "heart_failure",
    "hfref", "hfnef", "valvular", "chb", "sss", "aki", "cva_infract",
    "cva_bleed", "af", "vt", "psvt", "congenital", "uti",
    "neuro_cardiogenic_syncope", "orthostatic", "infective_endocarditis",
    "dvt", "cardiogenic_shock", "shock", "pulmonary_embolism",
    "chest_infection",
]

ICU_TEXT = ["subject_id", "gender", "ethnicity", "insurance", "hospital_admit_source"]

# Helpers

def _read_bronze(engine, table: str) -> pd.DataFrame:
    """Read a bronze table into a DataFrame."""
    return pd.read_sql(f"SELECT * FROM bronze.{table}", engine)


def _write_silver(engine, df: pd.DataFrame, table: str) -> None:
    """Write a DataFrame to a silver table (idempotent)."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE silver.{table}"))
        df.to_sql(table, conn, schema="silver", if_exists="append", index=False)
    logger.info("Loaded %d rows into silver.%s", len(df), table)


# Transforms (one per source)

def transform_patients(engine) -> pd.DataFrame:
    """Clean and type the patients source."""
    df = _read_bronze(engine, "patients")
    df = df.drop(columns=["_loaded_at"])

    df["doa"] = pd.to_datetime(df["doa"], format="%m/%d/%Y", errors="coerce")
    df["dod"] = pd.to_datetime(df["dod"], format="%m/%d/%Y", errors="coerce")

    df["gender"] = df["gender"].str.strip().map(GENDER_MAP)
    df["rural"] = df["rural"].str.strip().map(RURAL_MAP)
    df["type_of_admission"] = df["type_of_admission"].str.strip().map(ADMISSION_MAP)
    df["outcome"] = df["outcome"].str.strip().map(OUTCOME_MAP)

    for col in PATIENTS_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("patients transformed: %d rows", len(df))
    return df


def transform_mortality(engine) -> pd.DataFrame:
    """Clean and type the mortality source."""
    df = _read_bronze(engine, "mortality")
    df = df.drop(columns=["_loaded_at"])

    df["date_of_brought_dead"] = pd.to_datetime(
        df["date_of_brought_dead"], format="%m/%d/%Y", errors="coerce"
    )

    df["gender"] = df["gender"].str.strip().map(GENDER_MAP)
    df["rural_urban"] = df["rural_urban"].str.strip().map(RURAL_MAP)

    df["sno"] = pd.to_numeric(df["sno"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    logger.info("mortality transformed: %d rows", len(df))
    return df


def transform_billing(engine) -> pd.DataFrame:
    """Clean and type the billing source."""
    df = _read_bronze(engine, "billing")
    df = df.drop(columns=["_loaded_at"])

    for col in ["gender", "condition", "procedure", "outcome"]:
        df[col] = df[col].str.strip()

    df["readmission"] = df["readmission"].str.strip().map(YESNO_MAP)

    for col in ["age", "cost", "length_of_stay", "satisfaction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("billing transformed: %d rows", len(df))
    return df


def transform_icu(engine) -> pd.DataFrame:
    """Clean and type the ICU source (77 columns)."""
    df = _read_bronze(engine, "icu")
    df = df.drop(columns=["_loaded_at"])

    df["gender"] = df["gender"].str.strip().map(GENDER_MAP)

    for col in ["ethnicity", "insurance", "hospital_admit_source"]:
        df[col] = df[col].str.strip()

    numeric_cols = [c for c in df.columns if c not in ICU_TEXT]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("icu transformed: %d rows", len(df))
    return df


def transform_labs(engine) -> pd.DataFrame:
    """Clean and type the labs source.

    Note: `result` stays TEXT because it holds both numeric values
    (e.g. 28.9) and qualitative results (e.g. Normal, Negatif).
    """
    df = _read_bronze(engine, "labs")
    df = df.drop(columns=["_loaded_at"])

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    text_cols = [
        "test_name", "result", "unit", "reference_range", "status",
        "comment", "unit_description", "recommended_followup",
    ]
    for col in text_cols:
        df[col] = df[col].str.strip()

    df["min_reference"] = pd.to_numeric(df["min_reference"], errors="coerce")
    df["max_reference"] = pd.to_numeric(df["max_reference"], errors="coerce")

    logger.info("labs transformed: %d rows", len(df))
    return df

# Main Entry Point

def run() -> None:
    """Run all silver transforms."""
    engine = get_engine()
    _write_silver(engine, transform_patients(engine), "patients")
    _write_silver(engine, transform_mortality(engine), "mortality")
    _write_silver(engine, transform_billing(engine), "billing")
    _write_silver(engine, transform_icu(engine), "icu")
    _write_silver(engine, transform_labs(engine), "labs")


if __name__ == "__main__":
    run()