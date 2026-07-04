"""End-to-end tests for the HAIP integration layer (/health and /route).

Strategy:
  * Router logic is tested directly (pure, fast, no I/O).
  * Analytics and prediction are light -> tested through real HTTP calls.
  * RAG is heavy (loads an embedding model) -> MOCKED, so tests never load
    a model and stay fast + RAM-safe.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from api.router import route

client = TestClient(app)


# --- health ------------------------------------------------------------- #

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# --- router logic (direct, no HTTP) ------------------------------------- #

def test_route_analytics_keyword():
    d = route("how many admissions did we have")
    assert d["engine"] == "analytics"

def test_route_prediction_verb():
    d = route("forecast admissions for next 6 months")
    assert d["engine"] == "prediction"

def test_route_rag_keyword():
    d = route("what is the treatment guideline for sepsis")
    assert d["engine"] == "rag"

def test_route_empty_defaults_to_rag():
    d = route("")
    assert d["engine"] == "rag"
    assert d["confidence"] == 0.0

def test_route_no_match_defaults_to_rag():
    d = route("hello there")
    assert d["engine"] == "rag"


# --- full HTTP path: analytics (real, light) ---------------------------- #

def test_route_endpoint_analytics():
    r = client.post("/route", json={"question": "how many admissions"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["engine"] == "analytics"
    assert body["answer"]["answer_type"] == "admissions_summary"
    assert body["answer"]["total_admissions_all_time"] > 0


# --- full HTTP path: prediction (real, light) --------------------------- #

def test_route_endpoint_prediction():
    r = client.post("/route", json={"question": "forecast admissions"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["engine"] == "prediction"
    assert body["answer"]["answer_type"] == "admissions_forecast"
    assert len(body["answer"]["forecast"]) == 6


# --- full HTTP path: RAG (MOCKED, no model load) ------------------------ #

def test_route_endpoint_rag_mocked():
    fake = {
        "engine": "rag",
        "answer_type": "grounded_answer",
        "answer": "Mocked answer.",
        "citations": ["NICE: test"],
        "passages_used": 3,
        "latency_seconds": 0.01,
    }
    # Patch run_engine so the rag branch returns instantly, no model load.
    with patch("api.main.run_engine", return_value=fake) as m:
        r = client.post("/route", json={"question": "what is the sepsis guideline"})
        assert r.status_code == 200
        body = r.json()
        assert body["decision"]["engine"] == "rag"
        assert body["answer"]["answer"] == "Mocked answer."
        m.assert_called_once()
