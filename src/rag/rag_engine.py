"""
HAIP Week 3 - RAG Engine
=========================

The full Retrieval-Augmented Generation loop:

    question
        -> safety check (emergency redirect before any retrieval)
        -> Retriever (retrieve + re-rank)  -> top passages
        -> build a grounded prompt (answer ONLY from context)
        -> LLM (Gemini by default; Claude / OpenAI swappable)
        -> answer + citations + latency

Honest-engineering guards:
    * A safety net redirects medical emergencies and mental-health crises
      BEFORE any retrieval or LLM call (see safety.py).
    * The prompt instructs the model to answer ONLY from the retrieved
      context and to say so when the answer is not present. This prevents
      invented clinical facts (e.g. fabricated drug doses).
    * Every answer is returned WITH the source passages it used, so each
      claim is traceable.
    * Latency is measured and returned (a Week 3 evaluation metric).

Provider flexibility:
    LLM_PROVIDER selects the backend. Swapping between Gemini, Claude and
    OpenAI is a one-line config change; the rest of the pipeline is
    untouched. Gemini is the default because Google AI Studio offers a
    permanent free tier (no card) - suitable for a student capstone.

    Note: the corpus is PUBLIC WHO/CDC/NICE guidance, not patient data,
    so sending retrieved chunks to a cloud LLM raises no privacy concern.
    (Gemini's free tier may use inputs for training - acceptable here.)

Requires in .env (only the one for your chosen provider):
    GEMINI_API_KEY=...                (default Gemini backend)
    ANTHROPIC_API_KEY=sk-ant-...      (if LLM_PROVIDER = 'anthropic')
    OPENAI_API_KEY=sk-...             (if LLM_PROVIDER = 'openai')

Run a smoke test from project root:
    python -m src.rag.rag_engine
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv

from src.rag.retriever import Retriever
from src.rag.safety import safety_check

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LLM_PROVIDER = "gemini"             # "gemini" | "anthropic" | "openai"

GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4o-mini"

MAX_TOKENS = 600
MAX_CONTEXT_CHARS = 1500            # per-passage cap fed to the LLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_engine")


SYSTEM_PROMPT = (
    "You are a careful medical-guideline assistant. Answer the user's "
    "question using ONLY the numbered context passages provided. "
    "If the answer is not contained in the context, say clearly that the "
    "provided guidelines do not cover it - do not use outside knowledge and "
    "do not invent clinical details. Cite the passage numbers you used like "
    "[1], [2]. Keep the answer concise and factual."
)


# --------------------------------------------------------------------------- #
# RAG engine
# --------------------------------------------------------------------------- #

class RAGEngine:
    """Ties retrieval to generation and returns grounded, cited answers."""

    def __init__(self, provider: str = LLM_PROVIDER) -> None:
        self.provider = provider
        self.retriever = Retriever()
        self._client = None          # lazily created on first call
        logger.info("RAG engine ready (provider=%s)", provider)

    # -- prompt building ---------------------------------------------------- #

    def _build_context(self, passages: list[dict]) -> str:
        """Format retrieved passages into a numbered context block."""
        blocks = []
        for i, p in enumerate(passages, 1):
            text = p["text"][:MAX_CONTEXT_CHARS].strip()
            blocks.append(f"[{i}] (source: {p['citation']})\n{text}")
        return "\n\n".join(blocks)

    # -- provider-flexible LLM call ----------------------------------------- #

    def _call_llm(self, system: str, user: str) -> str:
        """Dispatch to the configured provider. Swappable by one config line."""
        if self.provider == "gemini":
            return self._call_gemini(system, user)
        if self.provider == "anthropic":
            return self._call_anthropic(system, user)
        if self.provider == "openai":
            return self._call_openai(system, user)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _call_gemini(self, system: str, user: str) -> str:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        # This SDK call has no separate system role; prepend it to the prompt.
        prompt = f"{system}\n\n{user}"
        resp = self._client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return resp.text

    def _call_anthropic(self, system: str, user: str) -> str:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        resp = self._client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    def _call_openai(self, system: str, user: str) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    # -- public API --------------------------------------------------------- #

    def answer(self, question: str) -> dict:
        """
        Run the full RAG loop.

        Returns a dict with:
            question, answer, citations (list), passages (raw retrieved),
            latency_seconds.
        """
        # Safety net: redirect emergencies before any retrieval/LLM call.
        flagged = safety_check(question)
        if flagged is not None:
            logger.warning("Safety redirect: %s", flagged["safety_flag"])
            return flagged

        start = time.time()

        passages = self.retriever.retrieve(question)

        if not passages:
            return {
                "question": question,
                "answer": "No relevant guidelines were found for this question.",
                "citations": [],
                "passages": [],
                "latency_seconds": round(time.time() - start, 2),
            }

        context = self._build_context(passages)
        user_prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}"
        answer_text = self._call_llm(SYSTEM_PROMPT, user_prompt)

        latency = round(time.time() - start, 2)
        logger.info("Answered in %.2fs using %d passages", latency, len(passages))

        return {
            "question": question,
            "answer": answer_text,
            "citations": [p["citation"] for p in passages],
            "passages": passages,
            "latency_seconds": latency,
        }


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    engine = RAGEngine()
    q = "What are the treatment options for resistant hypertension?"
    result = engine.answer(q)

    print("\n" + "=" * 70)
    print("QUESTION:", result["question"])
    print("=" * 70)
    print("\nANSWER:\n", result["answer"])
    print("\nCITATIONS:")
    for i, c in enumerate(result["citations"], 1):
        print(f"  [{i}] {c}")
    print(f"\nLatency: {result['latency_seconds']}s")