"""Validation layer: quality checks on silver tables.

Checks per table:
    - row count (non-empty)
    - null percentage per column
    - domain rules (impossible values flagged)

Results are logged. Failures are warnings, not crashes — this is a
report, so you can decide what to clean further.
"""

import pandas as pd

from etl.utils import get_engine, get_logger

logger = get_logger(__name__)

EXPECTED_ROWS = {
    "patients": 15757,
    "mortality": 359,
    "billing": 984,
    "icu": 5000,
    "labs": 27,
}

DOMAIN_RULES = [
    ("patients", "age", "age < 0 OR age > 120", "age out of range"),
    ("patients", "duration_of_stay", "duration_of_stay < 0", "negative stay"),
    ("mortality", "age", "age < 0 OR age > 120", "age out of range"),
    ("billing", "age", "age < 0 OR age > 120", "age out of range"),
    ("billing", "cost", "cost < 0", "negative cost"),
    ("billing", "satisfaction", "satisfaction < 1 OR satisfaction > 5", "satisfaction out of 1-5"),
    ("icu", "age", "age < 0 OR age > 120", "age out of range"),
    ("icu", "bmi", "bmi < 5 OR bmi > 100", "bmi out of range"),
]


def _read_silver(engine, table: str) -> pd.DataFrame:
    """Read a silver table into a DataFrame."""
    return pd.read_sql(f"SELECT * FROM silver.{table}", engine)


def check_row_counts(engine) -> None:
    """Verify each silver table matches its expected row count."""
    logger.info("--- ROW COUNT CHECK ---")
    for table, expected in EXPECTED_ROWS.items():
        actual = pd.read_sql(f"SELECT count(*) AS n FROM silver.{table}", engine)["n"][0]
        status = "OK" if actual == expected else "MISMATCH"
        logger.info("%s: %d rows (expected %d) [%s]", table, actual, expected, status)


def check_nulls(engine) -> None:
    """Report null percentage per column for each silver table."""
    logger.info("--- NULL PERCENTAGE CHECK ---")
    for table in EXPECTED_ROWS:
        df = _read_silver(engine, table)
        null_pct = (df.isnull().mean() * 100).round(1)
        high = null_pct[null_pct > 0]
        if high.empty:
            logger.info("%s: no nulls", table)
        else:
            logger.info("%s null %%: %s", table, high.to_dict())


def check_domain_rules(engine) -> None:
    """Flag rows that violate domain rules (impossible values)."""
    logger.info("--- DOMAIN RULE CHECK ---")
    for table, column, condition, description in DOMAIN_RULES:
        query = f"SELECT count(*) AS n FROM silver.{table} WHERE {condition}"
        bad = pd.read_sql(query, engine)["n"][0]
        if bad == 0:
            logger.info("%s.%s: OK (%s)", table, column, description)
        else:
            logger.warning("%s.%s: %d rows violate '%s'", table, column, bad, description)


def run() -> None:
    """Run all validation checks."""
    engine = get_engine()
    check_row_counts(engine)
    check_nulls(engine)
    check_domain_rules(engine)
    logger.info("--- VALIDATION COMPLETE ---")


if __name__ == "__main__":
    run()