"""Extract layer: read all 5 connected Synthea sources into DataFrames.

No cleaning here — just load and return. Cleaning happens in transform.py.
All five tables join on the patient key (patients.Id <- PATIENT).
"""

import pandas as pd

from etl.config import RAW_FILES
from etl.utils import get_logger

logger = get_logger(__name__)


def _read_csv(name: str) -> pd.DataFrame:
    """Read a single Synthea CSV source into a DataFrame."""
    path = RAW_FILES[name]
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn", low_memory=False)
    logger.info("%s: %d rows, %d cols", name, len(df), df.shape[1])
    return df


def extract_tabular() -> dict[str, pd.DataFrame]:
    """Read all 5 connected Synthea sources. Returns {name: DataFrame}."""
    sources = ["patients", "encounters", "conditions", "observations", "procedures"]
    data: dict[str, pd.DataFrame] = {}
    for name in sources:
        try:
            data[name] = _read_csv(name)
        except Exception as exc:
            logger.error("Failed to read %s: %s", name, exc)
            raise
    return data


if __name__ == "__main__":
    tabular = extract_tabular()
    for source_name, frame in tabular.items():
        logger.info("%s columns: %s", source_name, list(frame.columns))