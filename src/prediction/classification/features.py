"""Classification layer: build the connected feature matrix and risk targets.

The unit of analysis is one INPATIENT ENCOUNTER (a hospital stay), because
the flagship target — 30-day readmission — is defined per stay. For each
inpatient encounter we join across four connected tables:

    patients      -> demographics (age at admission, gender)
    encounters    -> this stay (length, cost, encounter class) + prior history
    conditions    -> comorbidities recorded BEFORE this stay (real Synthea codes)
    observations  -> vitals recorded DURING this stay (BP, BMI, heart rate)

Targets (all supported off the same shared matrix):
    readmission  -> another inpatient stay within 30 days of discharge
    mortality    -> patient died within 30 days of this stay's discharge
    high_cost    -> this stay's cost is in the top quartile

Leakage guards (the connected-data discipline):
    - TIME: every feature is known at or before discharge. Nothing from after
      the index stay is ever a feature. This is the central guard for
      readmission and is enforced when the feature is built (conditions use
      start < encounter stop; vitals use the index encounter only).
    - TARGET: a target's defining column is never a feature for that target
      (high_cost drops total_claim_cost; see feature_columns_for).

Run from src/ with:  python -m prediction.classification.features
"""

import logging

import pandas as pd

from etl.utils import get_engine, get_logger

logger: logging.Logger = get_logger(__name__)

# Comorbidity flags — real Synthea condition descriptions confirmed present.
# Each becomes a 0/1 feature: was this recorded for the patient before the stay?
COMORBIDITIES = {
    "cmb_hypertension":   "Essential hypertension (disorder)",
    "cmb_ischemic_heart": "Ischemic heart disease (disorder)",
    "cmb_metabolic_syn":  "Metabolic syndrome X (disorder)",
    "cmb_anemia":         "Anemia (disorder)",
    "cmb_obesity":        "Body mass index 30+ - obesity (finding)",
    "cmb_prediabetes":    "Prediabetes (finding)",
}

# Vital sign observations to average over the index stay.
VITALS = {
    "vit_systolic":  "Systolic Blood Pressure",
    "vit_diastolic": "Diastolic Blood Pressure",
    "vit_heart_rate": "Heart rate",
}

NUMERIC_FEATURES = [
    "age", "stay_length_days", "prior_encounters", "total_claim_cost",
] + list(VITALS.keys())

BINARY_FEATURES = list(COMORBIDITIES.keys())

TARGETS = ["readmission", "mortality", "high_cost"]

# Per-target leakage drops (beyond what is structurally excluded).
EXTRA_LEAKAGE = {
    "high_cost": ["total_claim_cost"],  # high_cost is derived from this column
}


def _load_base() -> pd.DataFrame:
    """One row per inpatient encounter with demographics, stay facts, targets.

    Readmission and the next-stay logic are computed in SQL with window
    functions so the time ordering is explicit and correct.
    """
    sql = """
    WITH inpt AS (
        SELECT
            e.id            AS encounter_id,
            e.patient,
            e.start,
            e.stop,
            e.total_claim_cost,
            EXTRACT(EPOCH FROM (e.stop - e.start)) / 86400.0 AS stay_length_days,
            LEAD(e.start) OVER (PARTITION BY e.patient ORDER BY e.start) AS next_start,
            ROW_NUMBER() OVER (PARTITION BY e.patient ORDER BY e.start) - 1 AS prior_encounters
        FROM silver.encounters e
        WHERE e.encounterclass = 'inpatient'
    )
    SELECT
        i.encounter_id,
        i.patient,
        i.start,
        i.stop,
        i.total_claim_cost,
        i.stay_length_days,
        i.prior_encounters,
        date_part('year', age(i.start, p.birthdate)) AS age,
        p.gender,
        p.deathdate,
        -- readmission: another inpatient stay within 30 days of discharge
        (i.next_start IS NOT NULL
         AND i.next_start - i.stop <= INTERVAL '30 days')::int AS readmission,
        -- mortality: died within 30 days of this discharge
        (p.deathdate IS NOT NULL
         AND p.deathdate - i.stop::date <= 30
         AND p.deathdate >= i.stop::date)::int AS mortality
    FROM inpt i
    JOIN silver.patients p ON p.id = i.patient;
    """
    engine = get_engine()
    logger.info("Loading inpatient encounter base table")
    df = pd.read_sql(sql, engine)
    logger.info("Base table: %d inpatient encounters", len(df))
    return df


def _add_comorbidities(engine, base: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 comorbidity flags recorded BEFORE each stay (time-safe).

    A comorbidity counts only if its condition.start is on or before the
    index stay's start — never using a diagnosis made after admission.
    """
    cond = pd.read_sql(
        "SELECT patient, description, start FROM silver.conditions "
        "WHERE description IN %(descs)s",
        engine, params={"descs": tuple(COMORBIDITIES.values())})
    cond["start"] = pd.to_datetime(cond["start"])

    base = base.copy()
    base_start = pd.to_datetime(base["start"]).dt.tz_localize(None)
    for feat, desc in COMORBIDITIES.items():
        # patients who had this condition recorded before each stay
        hits = cond[cond["description"] == desc][["patient", "start"]]
        merged = base.merge(hits, on="patient", how="left", suffixes=("", "_c"))
        merged["start_c"] = pd.to_datetime(merged["start_c"]).dt.tz_localize(None)
        merged["before"] = (merged["start_c"] <=
                            pd.to_datetime(merged["start"]).dt.tz_localize(None))
        flag = merged.groupby("encounter_id")["before"].any().astype(int)
        base[feat] = base["encounter_id"].map(flag).fillna(0).astype(int)
    logger.info("Added %d comorbidity flags", len(COMORBIDITIES))
    return base


def _add_vitals(engine, base: pd.DataFrame) -> pd.DataFrame:
    """Add mean vital signs measured DURING each index stay (time-safe)."""
    obs = pd.read_sql(
        "SELECT encounter, description, value FROM silver.observations "
        "WHERE description IN %(descs)s",
        engine, params={"descs": tuple(VITALS.values())})
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")

    base = base.copy()
    for feat, desc in VITALS.items():
        sub = obs[obs["description"] == desc]
        means = sub.groupby("encounter")["value"].mean()
        base[feat] = base["encounter_id"].map(means)
    logger.info("Added %d vital-sign features", len(VITALS))
    return base


def _build() -> pd.DataFrame:
    """Assemble the full per-encounter frame: base + comorbidities + vitals."""
    engine = get_engine()
    base = _load_base()
    base = _add_comorbidities(engine, base)
    base = _add_vitals(engine, base)
    return base


def get_features() -> pd.DataFrame:
    """Build the predictor matrix (numeric + binary), no missing values.

    Vitals missing for a stay are median-filled with a missingness flag,
    keeping the "not measured is signal" principle from the Week 1 EDA.
    """
    df = _build()

    numeric = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    for col in VITALS:  # only vitals are genuinely missing; flag then fill
        numeric[f"{col}_missing"] = numeric[col].isna().astype(int)
    numeric = numeric.fillna(numeric.median())

    binary = df[BINARY_FEATURES].astype(int)

    features = pd.concat([numeric, binary], axis=1)
    logger.info("Built feature matrix: %d rows x %d features", *features.shape)
    return features


def get_targets() -> pd.DataFrame:
    """Build the three binary targets aligned to the feature matrix rows."""
    df = _build()
    cost = pd.to_numeric(df["total_claim_cost"], errors="coerce")
    high_cost_threshold = cost.quantile(0.75)
    targets = pd.DataFrame({
        "readmission": df["readmission"].astype(int),
        "mortality": df["mortality"].astype(int),
        "high_cost": (cost >= high_cost_threshold).astype(int),
    })
    logger.info("Built targets: %s",
                {c: int(targets[c].sum()) for c in targets.columns})
    return targets


def feature_columns_for(target: str) -> list[str]:
    """Return feature names valid for a target, excluding leakage columns."""
    cols = list(get_features().columns)
    drop = set(EXTRA_LEAKAGE.get(target, []))
    removed = [c for c in cols if c in drop]
    cols = [c for c in cols if c not in drop]
    if removed:
        logger.info("Leakage guard for '%s': dropped %s", target, removed)
    return cols


if __name__ == "__main__":
    X = get_features()
    y = get_targets()
    print("Feature matrix:", X.shape)
    for t in TARGETS:
        print(f"  {t:12s} uses {len(feature_columns_for(t))} features")
    print("\nTargets (positive counts):")
    print(y.sum())
    print("\nMissing values left in X:", int(X.isna().sum().sum()))