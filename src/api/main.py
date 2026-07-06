"""HAIP FastAPI backend — integration layer for Week 5.

Fronts the existing engines (analytics / ML / RAG) behind one API.
This is the health-check scaffold; engine routes are added incrementally.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from api.router import route
from api.engines import run_engine
from api.auth_routes import router as auth_router

app = FastAPI(title="HAIP API", version="0.1.0")

# Allow the React dev server (Vite on :5173) to call this API from the browser.
# Without CORS, the browser blocks cross-origin requests.
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(auth_router)


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
