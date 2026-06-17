"""Central configuration: paths and database connection.

All scripts import from here so nothing is hardcoded.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Project root: three levels up from src/etl/config.py -> HAIP/ ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Load environment variables from .env at the project root ---
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# --- Raw data paths ---
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_FILES = {
    "patients":   RAW_DIR / "admissions&patients_data" / "patient&admission_data" / "HDHI Admission data.csv",
    "mortality":  RAW_DIR / "admissions&patients_data" / "patient&admission_data" / "HDHI Mortality Data.csv",
    "billing":    RAW_DIR / "billing" / "billing_data" / "hospital data analysis.csv",
    "icu":        RAW_DIR / "icu" / "ICU.csv",
    "labs":       RAW_DIR / "labs" / "lab_test_results_public.csv",
    "documents":  RAW_DIR / "medical_documents" / "guidelines_dataset",
}

# --- Postgres connection (credentials from environment, never hardcoded) ---
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "haip"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

DB_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)