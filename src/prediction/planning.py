"""Prediction layer: translate the encounter forecast into planning figures.

Turns each forecasted month into operational guidance a hospital can act on:
projected total encounter volume, change vs the last actual month, projected
inpatient (bed-requiring) cases, and an illustrative nursing estimate.

Data-derived inputs (from silver.encounters, 2018+):
    INPATIENT_SHARE     - fraction of encounters that are inpatient (need a bed)
    AVG_LENGTH_OF_STAY  - mean inpatient stay (days)

Stated planning assumption (NOT from the data - adjust per hospital policy):
    NURSE_TO_PATIENT_RATIO - patients per nurse on a general ward

The nursing estimate is illustrative: it shows how the forecast feeds a
staffing calculation, with the ratio left explicit so a hospital can
substitute its own.

Run from src/ with:  python -m prediction.planning
"""

import logging

import pandas as pd

from etl.utils import get_logger
from prediction.forecast import forecast_admissions
from prediction.timeseries import build_monthly_admissions

logger: logging.Logger = get_logger(__name__)

# Data-derived (from silver.encounters, 2018+)
INPATIENT_SHARE = 0.008    # 0.8% of encounters are inpatient (need a bed)
AVG_LENGTH_OF_STAY = 4.9   # mean inpatient stay, days

# Stated assumption - adjust per hospital policy (not from the data)
NURSE_TO_PATIENT_RATIO = 4  # patients per nurse, general ward


def build_planning(months_ahead: int = 6) -> pd.DataFrame:
    """Build a planning table from the encounter forecast.

    Args:
        months_ahead: Number of future months to plan for.

    Returns:
        DataFrame with columns:
            month            (datetime64)
            forecast         (int)   - projected total encounters
            change_pct       (float) - % change vs last actual month
            inpatient_cases  (int)   - projected bed-requiring admissions
            nurse_shifts     (int)   - illustrative concurrent nurses needed
    """
    last_actual = build_monthly_admissions().iloc[-1]["admissions"]
    logger.info("Planning baseline (last actual month): %d encounters",
                last_actual)

    fc = forecast_admissions(months_ahead).copy()
    fc["change_pct"] = ((fc["forecast"] - last_actual) / last_actual * 100).round(1)
    fc["inpatient_cases"] = (fc["forecast"] * INPATIENT_SHARE).round().astype(int)

    # Concurrent inpatients ≈ inpatient admissions × avg stay / days per month.
    # Nurses needed ≈ concurrent inpatients / nurse-to-patient ratio.
    concurrent = fc["inpatient_cases"] * AVG_LENGTH_OF_STAY / 30
    fc["nurse_shifts"] = (concurrent / NURSE_TO_PATIENT_RATIO).round().astype(int)
    # Floor at 1 nurse whenever any inpatient volume is projected.
    fc.loc[(fc["inpatient_cases"] > 0) & (fc["nurse_shifts"] < 1),
           "nurse_shifts"] = 1

    logger.info("Built planning table for %d months", len(fc))
    return fc


if __name__ == "__main__":
    plan = build_planning(6)
    print("Hospital planning forecast\n")
    print(f"Baseline (last actual month): "
          f"{build_monthly_admissions().iloc[-1]['admissions']} encounters")
    print(f"Assumptions: inpatient share {INPATIENT_SHARE:.1%}, "
          f"avg stay {AVG_LENGTH_OF_STAY} days, "
          f"nurse:patient 1:{NURSE_TO_PATIENT_RATIO}\n")
    print(plan.to_string(index=False))
    print("\nNote: nurse_shifts is illustrative - the ratio is a stated "
          "assumption, not derived from the data.")