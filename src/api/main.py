"""HAIP FastAPI backend — integration layer for Week 5.

Fronts the existing engines (analytics / ML / RAG) behind one API.
This is the health-check scaffold; engine routes are added incrementally.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from api.router import route

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
    """Return the routing decision for a question.

    This only DECIDES which engine should handle the question — it does not
    call the engine yet. The decision dict is fully transparent (engine,
    confidence, matched keywords, reason).
    """
    return route(req.question)
