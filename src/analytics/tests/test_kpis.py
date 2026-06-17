"""Tests for the KPI layer.

Verifies each KPI returns a well-formed, self-describing dict with a
sensible value. Run from the src/ directory with: python -m pytest
"""

from analytics import kpis


def _assert_kpi_shape(result: dict) -> None:
    """Every KPI must be a dict with value/label/unit of the right types."""
    assert isinstance(result, dict)
    assert set(result.keys()) == {"value", "label", "unit"}
    assert isinstance(result["value"], float)
    assert isinstance(result["label"], str) and result["label"]
    assert isinstance(result["unit"], str)


def test_mortality_rate():
    r = kpis.mortality_rate()
    _assert_kpi_shape(r)
    assert 0 <= r["value"] <= 100


def test_dama_rate():
    r = kpis.dama_rate()
    _assert_kpi_shape(r)
    assert 0 <= r["value"] <= 100


def test_icu_sepsis_rate():
    r = kpis.icu_sepsis_rate()
    _assert_kpi_shape(r)
    assert 0 <= r["value"] <= 100


def test_icu_readmission_rate():
    r = kpis.icu_readmission_rate()
    _assert_kpi_shape(r)
    assert 0 <= r["value"] <= 100


def test_comorbidity_burden():
    r = kpis.comorbidity_burden()
    _assert_kpi_shape(r)
    assert 0 <= r["value"] <= 100


def test_get_all_kpis():
    all_kpis = kpis.get_all_kpis()
    assert isinstance(all_kpis, list)
    assert len(all_kpis) == 5
    for k in all_kpis:
        _assert_kpi_shape(k)