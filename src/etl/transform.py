"""Transform layer: clean + type bronze data into silver tables.

Synthea data is already well-formed, so transforms are light:
    1. read bronze (all TEXT)
    2. select the columns silver keeps
    3. coerce date / numeric types (bad values -> NULL)
    4. write to typed silver table (idempotent)

Keys (id / patient / encounter) stay TEXT UUIDs so joins survive.
"""

import pandas as pd
from sqlalchemy import text

from etl.utils import get_engine, get_logger

logger = get_logger(__name__)


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
    """Clean and type the patients source. Keyed by id."""
    df = _read_bronze(engine, "patients")
    keep = [
        "id", "birthdate", "deathdate", "marital", "race", "ethnicity",
        "gender", "city", "state", "county", "zip",
        "healthcare_expenses", "healthcare_coverage", "income",
    ]
    df = df[keep]

    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
    df["deathdate"] = pd.to_datetime(df["deathdate"], errors="coerce")

    for col in ["healthcare_expenses", "healthcare_coverage", "income"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("patients transformed: %d rows", len(df))
    return df


def transform_encounters(engine) -> pd.DataFrame:
    """Clean and type encounters. patient -> patients.id."""
    df = _read_bronze(engine, "encounters")
    keep = [
        "id", "start", "stop", "patient", "organization", "payer",
        "encounterclass", "code", "description", "base_encounter_cost",
        "total_claim_cost", "payer_coverage", "reasoncode", "reasondescription",
    ]
    df = df[keep]

    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["stop"] = pd.to_datetime(df["stop"], errors="coerce")

    for col in ["base_encounter_cost", "total_claim_cost", "payer_coverage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("encounters transformed: %d rows", len(df))
    return df


def transform_conditions(engine) -> pd.DataFrame:
    """Clean and type conditions. patient + encounter."""
    df = _read_bronze(engine, "conditions")
    keep = ["start", "stop", "patient", "encounter", "system", "code", "description"]
    df = df[keep]

    df["start"] = pd.to_datetime(df["start"], errors="coerce").dt.date
    df["stop"] = pd.to_datetime(df["stop"], errors="coerce").dt.date

    logger.info("conditions transformed: %d rows", len(df))
    return df


def transform_observations(engine) -> pd.DataFrame:
    """Clean and type observations (labs + vitals). patient + encounter.

    Note: `value` stays TEXT — it holds both numeric (120) and
    qualitative (Never smoked) results, like the old labs.result.
    """
    df = _read_bronze(engine, "observations")
    keep = [
        "date", "patient", "encounter", "category", "code",
        "description", "value", "units", "type",
    ]
    df = df[keep]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.info("observations transformed: %d rows", len(df))
    return df


def transform_procedures(engine) -> pd.DataFrame:
    """Clean and type procedures (operations). patient + encounter."""
    df = _read_bronze(engine, "procedures")
    keep = [
        "start", "stop", "patient", "encounter", "system", "code",
        "description", "base_cost", "reasoncode", "reasondescription",
    ]
    df = df[keep]

    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["stop"] = pd.to_datetime(df["stop"], errors="coerce")
    df["base_cost"] = pd.to_numeric(df["base_cost"], errors="coerce")

    logger.info("procedures transformed: %d rows", len(df))
    return df


# Main Entry Point

def run() -> None:
    """Run all silver transforms."""
    engine = get_engine()
    _write_silver(engine, transform_patients(engine), "patients")
    _write_silver(engine, transform_encounters(engine), "encounters")
    _write_silver(engine, transform_conditions(engine), "conditions")
    _write_silver(engine, transform_observations(engine), "observations")
    _write_silver(engine, transform_procedures(engine), "procedures")


if __name__ == "__main__":
    run()