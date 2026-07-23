"""HAIP FastAPI backend — integration layer for Week 5.

Fronts the existing engines (analytics / ML / RAG) behind one API.
This is the health-check scaffold; engine routes are added incrementally.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from api.router import route
from api.engines import run_engine
from api.auth_routes import router as auth_router
from api.jwt_utils import get_current_user
from fastapi import Depends
import pandas as pd
from pathlib import Path

DOCTORS_CSV = Path(__file__).resolve().parents[2] / "data" / "samples" / "hospital_doctors_large.csv"

app = FastAPI(title="HAIP API", version="0.1.0")

# Allow the React dev server (Vite on :5173) to call this API from the browser.
# Without CORS, the browser blocks cross-origin requests.
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://16.170.171.18:5173",
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
def route_question(req: RouteRequest,
                   user: dict = Depends(get_current_user)):
    """Route a question and return both the decision and the engine's answer.

    Full path: question -> routing decision -> engine call -> answer.
    The decision (engine, confidence, matched, reason) stays visible alongside
    the answer, so routing is never a black box.
    """
    decision = route(req.question)
    answer = run_engine(decision["engine"], req.question)
    return {"decision": decision, "answer": answer}


@app.get("/doctors")
def list_doctors(user: dict = Depends(get_current_user)):
    """Provider directory grouped by department for the Doctors screen.
    Reads the sample dataset; O/E scoring is layered on in a later step.
    """
    df = pd.read_csv(DOCTORS_CSV)

    def clean(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return v

    depts = {}
    for _, r in df.iterrows():
        enc = r["encounters"]
        enc = int(enc) if pd.notna(enc) else 0
        dept = clean(r["department"]) or "Unassigned"
        depts.setdefault(dept, []).append({
            "name": clean(r["doctor_name"]),
            "specialty": clean(r["specialty"]),
            "encounters": enc,
        })
    return {"departments": depts, "total": int(len(df))}
