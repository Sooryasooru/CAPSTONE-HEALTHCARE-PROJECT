"""Prediction layer: forecast future monthly encounter volume.

Fits a Holt-Winters (Exponential Smoothing) model on the monthly encounter
series and projects future months. Holt-Winters is chosen because the series
shows both a trend and a yearly seasonal cycle, and it stays interpretable
on a moderate-length series where heavier ML models would overfit.

Seasonality is additive with a 12-month period (the seasonal swing is
roughly constant in size, not proportional to the level).

Run from src/ with:  python -m prediction.forecast
"""

import logging

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from etl.utils import get_logger
from prediction.timeseries import build_monthly_admissions

logger: logging.Logger = get_logger(__name__)

SEASONAL_PERIODS = 12  # one full yearly cycle


def forecast_admissions(months_ahead: int = 6) -> pd.DataFrame:
    """Forecast monthly encounter volume for the next `months_ahead` months.

    Args:
        months_ahead: Number of future months to project.

    Returns:
        DataFrame with columns:
            month       (datetime64) - first day of each future month
            forecast    (int)        - predicted encounter volume

    Raises:
        ValueError: if the series is too short for seasonal forecasting
                    (needs at least two full 12-month cycles).
    """
    series = build_monthly_admissions()
    if len(series) < 2 * SEASONAL_PERIODS:
        raise ValueError(
            f"series too short for seasonal forecast: {len(series)} months "
            f"(need >= {2 * SEASONAL_PERIODS})")

    y = series.set_index("month")["admissions"]
    y.index.freq = "MS"  # month-start frequency, required by statsmodels

    logger.info("Fitting Holt-Winters on %d months", len(y))
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIODS,
    ).fit()

    logger.info("Forecasting %d months ahead", months_ahead)
    predicted = model.forecast(months_ahead).round().astype(int)

    result = predicted.reset_index()
    result.columns = ["month", "forecast"]
    return result


if __name__ == "__main__":
    last_actual = build_monthly_admissions().iloc[-1]
    fc = forecast_admissions(6)
    print(f"Last actual: {last_actual['month'].date()} = "
          f"{last_actual['admissions']} encounters\n")
    print("Forecast:")
    print(fc.to_string(index=False))