"""Load layer: write extracted Synthea DataFrames into bronze tables.

Idempotent — truncates each bronze table before insert, so re-runs
never duplicate rows. Synthea's UPPERCASE headers are lowercased to
match the bronze schema; no per-column renaming needed.
"""

from sqlalchemy import text

from etl.extract import extract_tabular
from etl.utils import get_engine, get_logger

logger = get_logger(__name__)


def load_to_bronze() -> None:
    """Load all extracted Synthea datasets into bronze tables."""
    engine = get_engine()
    datasets = extract_tabular()

    for name, df in datasets.items():
        df = df.copy()
        # Synthea ships UPPERCASE headers; bronze schema is lowercase.
        df.columns = [c.lower() for c in df.columns]

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