"""Shared utilities for the ETL pipeline.

Centralizes logging setup and database engine creation so the
extract / transform / load / validate modules don't duplicate it.
"""

import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from etl.config import DB_URL


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Using a shared setup means every module logs in the same format
    without repeating basicConfig.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def get_engine() -> Engine:
    """Create a SQLAlchemy engine from the configured DB_URL."""
    return create_engine(DB_URL)
