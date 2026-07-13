"""FastAPI service exposing the HAIP clinical agent over HTTP.

This is the agent's public interface — the React "Ask HAIP" card calls it.
It mirrors the structure of the existing api/main.py (CORS middleware,
health check, typed request/response) so it fits the rest of the codebase.

Endpoints:
    GET  /health       - liveness probe for Docker / the frontend
    POST /agent/chat   - ask the agent a question, get an answer plus the
                         list of tools it used (so the UI can show its work)

Honest-engineering note:
    The Gemini free tier is capped (20 requests/day for gemini-2.5-flash).
    When that quota is hit, the underlying call raises a RESOURCE_EXHAUSTED
    (HTTP 429) error. Rather than surfacing an opaque 500 "Internal Server
    Error", we catch it and return a clean 503 with a plain-language message
    the UI can show directly. This is a deliberate, disclosed limitation of
    running on a free tier — not a hidden failure.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from src.agent.graph import AGENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HAIP Agent", version="1.0.0")

# Same permissive CORS as the other services — the frontend calls this
# directly from the browser at localhost:5173.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Incoming question from the frontend."""
    question: str


class ChatResponse(BaseModel):
    """Agent's answer plus a trace of which tools it used."""
    answer: str
    tools_used: list[str]


def _to_text(content) -> str:
    """Normalise an AI message's content into a single string.

    Gemini returns a plain string for simple answers, but on multi-tool
    synthesis it can return a list of content blocks, e.g.
        [{'type': 'text', 'text': '...'}, ...]
    Pydantic's `answer: str` rejects that list, causing a 500. This
    flattens both shapes into one string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _is_quota_error(exc: Exception) -> bool:
    """Detect a Gemini free-tier quota / rate-limit error.

    We match on the signals present in the raised error text rather than a
    specific exception class, so this stays robust across langchain-google
    versions. Both the daily cap and the per-minute cap surface as 429
    RESOURCE_EXHAUSTED.
    """
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "haip-agent", "version": "1.0.0"}


@app.post("/agent/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Run the agent on a question and return the answer + tools used.

    The agent may call several tools internally. We walk the resulting
    message history to (a) pull out the final answer and (b) collect the
    names of every tool that was invoked, so the UI can display the
    agent's reasoning trail.

    If the Gemini free-tier quota is exhausted, we return a clean 503 with
    a user-friendly message instead of a raw 500.
    """
    logger.info("Agent question: %s", req.question)

    try:
        result = AGENT.invoke({"messages": [HumanMessage(content=req.question)]})
    except Exception as exc:  # noqa: BLE001 - we re-classify below
        if _is_quota_error(exc):
            logger.warning("Gemini quota exhausted (429): %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    "The AI service is temporarily unavailable because the "
                    "free-tier request limit has been reached. Please try "
                    "again in a minute. (This is a known limitation of the "
                    "free Gemini tier used for this proof of concept.)"
                ),
            ) from exc
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=500,
            detail="The agent hit an unexpected error while answering.",
        ) from exc

    messages = result["messages"]

    # Collect tool names from the AI messages that requested them.
    tools_used: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tools_used.extend(tc["name"] for tc in m.tool_calls)

    # The final answer is the content of the last AI message. Gemini may
    # return this as a string or a list of content blocks, so normalise it.
    answer = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            answer = _to_text(m.content)
            if answer:
                break

    return ChatResponse(answer=answer, tools_used=tools_used)