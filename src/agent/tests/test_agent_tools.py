"""Tests for the agent's CSV-backed tools (no PostgreSQL dependency)."""
import pandas as pd
import pytest
from src.agent import tools


def test_hospital_kpis_returns_real_numbers():
    out = tools.get_hospital_kpis.func()
    assert "Total encounters" in out
    assert "readmission rate" in out.lower()
    assert "unavailable" not in out.lower()


def test_forecast_returns_projection():
    out = tools.forecast_admissions.func(months_ahead=6)
    assert "Forecast for the next 6 months" in out
    assert "admissions" in out
    assert "unavailable" not in out.lower()


def test_forecast_respects_months_arg():
    out = tools.forecast_admissions.func(months_ahead=3)
    # 3 month lines, each starts with "- "
    assert out.count("- ") == 3
