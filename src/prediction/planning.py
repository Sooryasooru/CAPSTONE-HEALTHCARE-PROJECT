"""Prediction layer: translate the admissions forecast into planning figures.

Turns each forecasted month into operational guidance a hospital can act on:
projected volume, change vs the last actual month, projected cardiac-related
cases, and an illustrative nursing estimate.

Data-derived inputs (from silver.patients):
    CARDIAC_PREVALENCE  - CAD prevalence, the dominant cardiac marker
    AVG_LENGTH_OF_STAY  - mean duration_of_stay (days)

Stated planning assumption (NOT from the data - adjust per hospital policy):
    NURSE_TO_PATIENT_RATIO - patients per nurse on a general ward

Nursing estimate is illustrative: it shows how the forecast feeds a staffing
calculation, with the ratio left explicit so a hospital can substitute its own.

Run from src/ with:  python -m prediction.planning
"""

import logging

import pandas as pd

from etl.utils import get_logger
from prediction.forecast import forecast_admissions
from prediction.timeseries import build_monthly_admissions

logger: logging.Logger = get_logger(__name__)

# Data-derived (from silver.patients EDA)
CARDIAC_PREVALENCE = 0.67   # CAD prevalence in the cohort
AVG_LENGTH_OF_STAY = 6.4    # mean duration_of_stay, days

# Stated assumption - adjust per hospital policy (not from the data)
NURSE_TO_PATIENT_RATIO = 4  # patients per nurse, general ward


def build_planning(months_ahead: int = 6) -> pd.DataFrame:
    """Build a planning table from the admissions forecast.

    Args:
        months_ahead: Number of future months to plan for.

    Returns:
        DataFrame with columns:
            month            (datetime64)
            forecast         (int)   - projected admissions
            change_pct       (float) - % change vs last actual month
            cardiac_cases    (int)   - projected cardiac-related admissions
            nurse_shifts     (int)   - illustrative concurrent nurses needed
    """
    last_actual = build_monthly_admissions().iloc[-1]["admissions"]
    logger.info("Planning baseline (last actual month): %d admissions",
                last_actual)

    fc = forecast_admissions(months_ahead).copy()
    fc["change_pct"] = ((fc["forecast"] - last_actual) / last_actual * 100).round(1)
    fc["cardiac_cases"] = (fc["forecast"] * CARDIAC_PREVALENCE).round().astype(int)

    # Concurrent patients ≈ monthly admissions × avg stay / days per month.
    # Nurses needed ≈ concurrent patients / nurse-to-patient ratio.
    concurrent = fc["forecast"] * AVG_LENGTH_OF_STAY / 30
    fc["nurse_shifts"] = (concurrent / NURSE_TO_PATIENT_RATIO).round().astype(int)

    logger.info("Built planning table for %d months", len(fc))
    return fc


if __name__ == "__main__":
    plan = build_planning(6)
    print("Hospital planning forecast\n")
    print(f"Baseline (last actual month): "
          f"{build_monthly_admissions().iloc[-1]['admissions']} admissions")
    print(f"Assumptions: cardiac prevalence {CARDIAC_PREVALENCE:.0%}, "
          f"avg stay {AVG_LENGTH_OF_STAY} days, "
          f"nurse:patient 1:{NURSE_TO_PATIENT_RATIO}\n")
    print(plan.to_string(index=False))
    print("\nNote: nurse_shifts is illustrative - the ratio is a stated "
          "assumption, not derived from the data.")