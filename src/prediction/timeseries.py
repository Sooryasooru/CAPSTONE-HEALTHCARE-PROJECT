"""Prediction layer: build the monthly admissions time series.

Reads silver.patients (per-patient rows) and aggregates into a clean,
chronologically-ordered monthly admissions series — the input the
forecasting model consumes.

The month_year column is stored as text ("Apr-17"), which sorts
alphabetically rather than by time. This module parses it into a real
date so months line up chronologically.

Run from src/ with:  python -m prediction.timeseries
"""

import logging

import pandas as pd

from etl.utils import get_engine, get_logger

logger: logging.Logger = get_logger(__name__)


def build_monthly_admissions() -> pd.DataFrame:
    """Build the monthly admissions series from silver.patients.

    Returns:
        DataFrame with columns:
            month       (datetime64) - first day of each month, sorted
            admissions  (int)        - count of admissions that month
    """
    engine = get_engine()
    logger.info("Reading month_year from silver.patients")
    df = pd.read_sql("SELECT month_year FROM silver.patients", engine)

    # Parse "Apr-17" -> 2017-04-01 so months order by time, not alphabet.
    df["month"] = pd.to_datetime(df["month_year"], format="%b-%y")

    # Count admissions per month, sorted chronologically.
    series = (
        df.groupby("month").size()
        .reset_index(name="admissions")
        .sort_values("month")
        .reset_index(drop=True)
    )
    logger.info("Built monthly series: %d months (%s to %s)",
                len(series), series["month"].min().date(),
                series["month"].max().date())
    return series


if __name__ == "__main__":
    monthly = build_monthly_admissions()
    print(monthly.to_string(index=False))