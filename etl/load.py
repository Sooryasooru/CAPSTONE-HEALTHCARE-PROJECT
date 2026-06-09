"""
Load Layer

Purpose:
    Load extracted datasets into Bronze tables.

Features:
    - Idempotent (safe to rerun)
    - Truncates existing data before loading
    - Logs loading status
"""

import logging

import pandas as pd
from sqlalchemy import create_engine, text

from etl.config import DB_URL
from etl.extract import extract_tabular


# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Bronze Table Column Mapping
# ------------------------------------------------------------

BRONZE_COLUMNS = {

    "patients": [
        "sno",
        "mrd_no",
        "doa",
        "dod",
        "age",
        "gender",
        "rural",
        "type_of_admission",
        "month_year",
        "duration_of_stay",
        "duration_of_icu_stay",
        "outcome",
        "smoking",
        "alcohol",
        "dm",
        "htn",
        "cad",
        "prior_cmp",
        "ckd",
        "hb",
        "tlc",
        "platelets",
        "glucose",
        "urea",
        "creatinine",
        "bnp",
        "raised_cardiac_enzymes",
        "ef",
        "severe_anaemia",
        "anaemia",
        "stable_angina",
        "acs",
        "stemi",
        "atypical_chestpain",
        "heart_failure",
        "hfref",
        "hfnef",
        "valvular",
        "chb",
        "sss",
        "aki",
        "cva_infract",
        "cva_bleed",
        "af",
        "vt",
        "psvt",
        "congenital",
        "uti",
        "neuro_cardiogenic_syncope",
        "orthostatic",
        "infective_endocarditis",
        "dvt",
        "cardiogenic_shock",
        "shock",
        "pulmonary_embolism",
        "chest_infection"
    ],

    "mortality": [
        "sno",
        "mrd",
        "age",
        "gender",
        "rural_urban",
        "date_of_brought_dead"
    ],

    "billing": [
        "patient_id",
        "age",
        "gender",
        "condition",
        "procedure",
        "cost",
        "length_of_stay",
        "readmission",
        "outcome",
        "satisfaction"
    ],

    "labs": [
        "date",
        "test_name",
        "result",
        "unit",
        "reference_range",
        "status",
        "comment",
        "min_reference",
        "max_reference",
        "unit_description",
        "recommended_followup"
    ]
}


# ------------------------------------------------------------
# Load Data Into Bronze Layer
# ------------------------------------------------------------

def load_to_bronze() -> None:
    """
    Load all extracted datasets into Bronze tables.
    """

    engine = create_engine(DB_URL)

    datasets = extract_tabular()

    for dataset_name, dataframe in datasets.items():

        if dataset_name in BRONZE_COLUMNS:

            dataframe = dataframe.copy()
            dataframe.columns = BRONZE_COLUMNS[dataset_name]

        try:

            with engine.begin() as connection:

                connection.execute(
                    text(
                        f"TRUNCATE TABLE bronze.{dataset_name}"
                    )
                )

                dataframe.to_sql(
                    name=dataset_name,
                    con=connection,
                    schema="bronze",
                    if_exists="append",
                    index=False
                )

            logger.info(
                "Loaded %d rows into bronze.%s",
                len(dataframe),
                dataset_name
            )

        except Exception as error:

            logger.error(
                "Failed to load bronze.%s: %s",
                dataset_name,
                error
            )

            raise


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    load_to_bronze()