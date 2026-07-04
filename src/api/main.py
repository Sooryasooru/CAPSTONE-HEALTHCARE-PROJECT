"""HAIP FastAPI backend — integration layer for Week 5.

Fronts the existing engines (analytics / ML / RAG) behind one API.
This is the health-check scaffold; engine routes are added incrementally.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from api.router import route
from api.engines import run_engine

app = FastAPI(title="HAIP API", version="0.1.0")


@app.get("/health")
def health():
    """Liveness check — confirms the backend is up. No engine/model loading."""
    return {"status": "ok", "service": "haip-api", "version": "0.1.0"}


class RouteRequest(BaseModel):
    """A user's natural-language question to be routed."""
    question: str


@app.post("/route")
def route_question(req: RouteRequest):
    """Route a question and return both the decision and the engine's answer.

    Full path: question -> routing decision -> engine call -> answer.
    The decision (engine, confidence, matched, reason) stays visible alongside
    the answer, so routing is never a black box.
    """
    decision = route(req.question)
    answer = run_engine(decision["engine"], req.question)
    return {"decision": decision, "answer": answer}
