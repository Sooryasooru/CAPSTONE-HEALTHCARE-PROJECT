"""HAIP FastAPI backend — integration layer for Week 5.

Fronts the existing engines (analytics / ML / RAG) behind one API.
This is the health-check scaffold; engine routes are added incrementally.
"""
from fastapi import FastAPI

app = FastAPI(title="HAIP API", version="0.1.0")


@app.get("/health")
def health():
    """Liveness check — confirms the backend is up. No engine/model loading."""
    return {"status": "ok", "service": "haip-api", "version": "0.1.0"}
