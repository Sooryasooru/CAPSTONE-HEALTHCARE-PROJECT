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

# --- Raw data paths (Synthea connected dataset) ---
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "synthea"

# Five core connected tables. All join on the patient key.
RAW_FILES = {
    "patients":     RAW_DIR / "patients.csv",
    "encounters":   RAW_DIR / "encounters.csv",
    "conditions":   RAW_DIR / "conditions.csv",
    "observations": RAW_DIR / "observations.csv",
    "procedures":   RAW_DIR / "procedures.csv",
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