"""Prediction layer: build the monthly encounter-volume time series.

Reads silver.encounters (real timestamps) and aggregates ALL encounters
into a clean, chronologically-ordered monthly series from a recent window
— the input the forecasting model consumes.

Why all encounters (not just inpatient): inpatient stays are sparse
(~2-3/month across a 42-year simulated span), too thin for a seasonal
forecast. Total encounter volume (~390/month) gives Holt-Winters real
signal. Why a recent window: Synthea spreads encounters over decades;
the last several years are the relevant, dense, forecastable period.

The current (incomplete) month is dropped so the final point isn't a
misleading partial count.

Run from src/ with:  python -m prediction.timeseries
"""

import logging

import pandas as pd

from etl.utils import get_engine, get_logger

logger: logging.Logger = get_logger(__name__)

START_YEAR = "2018-01-01"  # recent dense window for forecasting


def build_monthly_admissions() -> pd.DataFrame:
    """Build the monthly encounter-volume series from silver.encounters.

    Returns:
        DataFrame with columns:
            month       (datetime64) - first day of each month, sorted
            admissions  (int)        - count of encounters that month
    """
    engine = get_engine()
    logger.info("Reading encounter start dates from silver.encounters")
    df = pd.read_sql(
        "SELECT start FROM silver.encounters WHERE start >= %(start)s",
        engine, params={"start": START_YEAR},
    )

    df["start"] = pd.to_datetime(df["start"])
    df["month"] = df["start"].dt.to_period("M").dt.to_timestamp()

    series = (
        df.groupby("month").size()
        .reset_index(name="admissions")
        .sort_values("month")
        .reset_index(drop=True)
    )

    # Drop the final month if it is the current (incomplete) month.
    current_month = pd.Timestamp.now().to_period("M").to_timestamp()
    if not series.empty and series["month"].iloc[-1] >= current_month:
        dropped = series.iloc[-1]["month"].date()
        series = series.iloc[:-1].reset_index(drop=True)
        logger.info("Dropped incomplete current month: %s", dropped)

    logger.info("Built monthly series: %d months (%s to %s)",
                len(series), series["month"].min().date(),
                series["month"].max().date())
    return series


if __name__ == "__main__":
    monthly = build_monthly_admissions()
    print(monthly.tail(12).to_string(index=False))
    print(f"\n{len(monthly)} months, "
          f"{monthly['admissions'].sum()} total encounters, "
          f"avg {monthly['admissions'].mean():.0f}/month")