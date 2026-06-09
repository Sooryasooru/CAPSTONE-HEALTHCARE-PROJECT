"""Extract layer: read all 6 raw sources into DataFrames.

No cleaning here — just load and return. Cleaning happens in transform.py.
"""

import logging

import pandas as pd
from datasets import load_from_disk

from etl.config import RAW_FILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _read_csv(name: str) -> pd.DataFrame:
    """Read a single CSV source into a DataFrame."""
    path = RAW_FILES[name]
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")
    logger.info("%s: %d rows, %d cols", name, len(df), df.shape[1])
    return df


def extract_tabular() -> dict[str, pd.DataFrame]:
    """Read all 5 tabular CSV sources. Returns {name: DataFrame}."""
    sources = ["patients", "mortality", "billing", "icu", "labs"]
    data: dict[str, pd.DataFrame] = {}
    for name in sources:
        try:
            data[name] = _read_csv(name)
        except Exception as exc:
            logger.error("Failed to read %s: %s", name, exc)
            raise
    return data


def extract_documents():
    """Load the medical guidelines dataset (for RAG, not bronze)."""
    path = RAW_FILES["documents"]
    try:
        ds = load_from_disk(str(path))
        logger.info("documents: loaded splits %s", list(ds.keys()))
        return ds
    except Exception as exc:
        logger.error("Failed to load documents: %s", exc)
        raise


if __name__ == "__main__":
    tabular = extract_tabular()
    for source_name, frame in tabular.items():
        logger.info("%s columns: %s", source_name, list(frame.columns))