"""Prediction layer: evaluate forecast accuracy, honestly scoped to the data.

A true seasonal backtest needs at least two full seasonal cycles (24 months)
*in the training split*. With only 24 months total, holding any out drops the
training data below that limit, so seasonal Holt-Winters cannot be backtested
here. We therefore evaluate two honest, clearly-labelled ways:

    C - fit quality (in-sample):
        Fit the seasonal model on all 24 months and measure how closely the
        fitted values match the actuals. Shows whether the model captured the
        pattern. NOT a test of predicting unseen months (the model saw this data).

    A - trend-only backtest (out-of-sample):
        Hold out the last `test_months`, train a TREND-ONLY model (seasonality
        dropped so it runs on the shorter split), forecast the held-out window
        and compare to reality. A fair but conservative test: the trend-only
        model is simpler than the production seasonal model, so its error tends
        to look worse than the real forecast's.

Both report MAE / RMSE / MAPE. A full seasonal backtest would need ~36+ months.

Run from src/ with:  python -m prediction.evaluate
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from etl.utils import get_logger
from prediction.forecast import SEASONAL_PERIODS
from prediction.timeseries import build_monthly_admissions

logger: logging.Logger = get_logger(__name__)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE and MAPE for predicted vs actual values."""
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(np.abs(error / actual)) * 100),
    }


def fit_quality() -> dict:
    """Evaluation C: in-sample fit of the seasonal model on all 24 months.

    Returns:
        Dict with mae, rmse, mape of fitted vs actual values.
    """
    series = build_monthly_admissions()
    y = series.set_index("month")["admissions"]
    y.index.freq = "MS"

    logger.info("Fit quality (C): seasonal model on all %d months", len(y))
    model = ExponentialSmoothing(
        y, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIODS,
    ).fit()

    fitted = model.fittedvalues.to_numpy()
    result = _metrics(y.to_numpy(), fitted)
    logger.info("Fit quality: MAE=%.1f RMSE=%.1f MAPE=%.1f%%",
                result["mae"], result["rmse"], result["mape"])
    return result


def trend_only_backtest(test_months: int = 6) -> dict:
    """Evaluation A: out-of-sample backtest with a trend-only model.

    Args:
        test_months: Number of most-recent months to hold out and predict.

    Returns:
        Dict with keys: comparison (DataFrame), mae, rmse, mape.
    """
    series = build_monthly_admissions()
    y = series.set_index("month")["admissions"]
    y.index.freq = "MS"

    train = y.iloc[:-test_months]
    test = y.iloc[-test_months:]
    logger.info("Trend-only backtest (A): train %d, test %d months",
                len(train), len(test))

    model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
    predicted = model.forecast(test_months).round().astype(int)

    actual = test.to_numpy()
    forecast = predicted.to_numpy()
    result = _metrics(actual, forecast)
    result["comparison"] = pd.DataFrame({
        "month": test.index,
        "actual": actual,
        "forecast": forecast,
        "error": forecast - actual,
    })
    logger.info("Trend-only backtest: MAE=%.1f RMSE=%.1f MAPE=%.1f%%",
                result["mae"], result["rmse"], result["mape"])
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATION A — trend-only out-of-sample backtest")
    print("(conservative: simpler model than production seasonal forecast)")
    print("=" * 60)
    a = trend_only_backtest(6)
    print(a["comparison"].to_string(index=False))
    print(f"\nMAE  : {a['mae']:.1f} admissions")
    print(f"RMSE : {a['rmse']:.1f} admissions")
    print(f"MAPE : {a['mape']:.1f}%  (accuracy ≈ {100 - a['mape']:.1f}%)")

    print("\n" + "=" * 60)
    print("EVALUATION C — seasonal model fit quality (in-sample)")
    print("(production model on all 24 months; not an unseen-data test)")
    print("=" * 60)
    c = fit_quality()
    print(f"MAE  : {c['mae']:.1f} admissions")
    print(f"RMSE : {c['rmse']:.1f} admissions")
    print(f"MAPE : {c['mape']:.1f}%  (fit ≈ {100 - c['mape']:.1f}%)")

    print("\nNote: a full seasonal backtest needs ~36+ months of history.")