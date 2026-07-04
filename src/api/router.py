"""HAIP query router — decides which engine handles a user question.

Simple, explainable keyword/intent routing (NOT LLM routing).
Scores each engine by keyword matches, picks the highest.
Low confidence -> defaults to RAG (best at open questions).

The router only DECIDES. It returns the decision + reasoning so the
choice is always transparent. Calling the engine is a separate step.
"""

# Keyword signals per engine. Extend freely — this is meant to be readable.
ANALYTICS_KEYWORDS = {
    "summary", "admissions", "count", "how many", "total", "average",
    "trend", "breakdown", "statistics", "stats", "distribution", "volume",
}
PREDICTION_KEYWORDS = {
    "predict", "forecast", "future", "next month", "risk", "expected",
    "projection", "estimate", "upcoming", "plan", "planning",
}
RAG_KEYWORDS = {
    "what is", "explain", "guideline", "recommend", "treatment", "protocol",
    "why", "how do", "definition", "according to", "should",
}

# Strong forecasting verbs. If any appear, the question is a prediction
# regardless of what noun follows ("forecast admissions" = prediction).
STRONG_PREDICTION_VERBS = {"forecast", "predict", "projection", "project"}

CONFIDENCE_THRESHOLD = 0.34  # below this -> fall back to RAG


def _score(question: str, keywords: set[str]) -> tuple[int, list[str]]:
    """Count how many keywords appear in the question. Returns (hits, matched)."""
    q = question.lower()
    matched = [kw for kw in keywords if kw in q]
    return len(matched), matched


def route(question: str) -> dict:
    """Decide which engine should answer. Returns a transparent decision dict.

    Output keys:
      engine     - "analytics" | "prediction" | "rag"
      confidence - 0.0-1.0, share of the winning engine's matches
      matched    - the keywords that triggered the choice
      reason     - plain-English explanation of the decision
    """
    if not question or not question.strip():
        return {
            "engine": "rag",
            "confidence": 0.0,
            "matched": [],
            "reason": "empty question, defaulting to knowledge base",
        }

    scores = {
        "analytics": _score(question, ANALYTICS_KEYWORDS),
        "prediction": _score(question, PREDICTION_KEYWORDS),
        "rag": _score(question, RAG_KEYWORDS),
    }

    total_hits = sum(hits for hits, _ in scores.values())
    winner = max(scores, key=lambda e: scores[e][0])
    win_hits, win_matched = scores[winner]

    # Tie-break: a forecasting verb forces the prediction engine, even if
    # another engine matched more keywords (e.g. "forecast admissions").
    q_lower = question.lower()
    verb_hit = [v for v in STRONG_PREDICTION_VERBS if v in q_lower]
    if verb_hit and winner != "prediction":
        return {
            "engine": "prediction",
            "confidence": 1.0,
            "matched": verb_hit,
            "reason": f"forecasting verb present {verb_hit}, routing to prediction",
        }

    # No keyword matched anything -> honest fallback to RAG.
    if win_hits == 0:
        return {
            "engine": "rag",
            "confidence": 0.0,
            "matched": [],
            "reason": "no keyword match, defaulting to knowledge base",
        }

    confidence = win_hits / total_hits if total_hits else 0.0

    # Weak/ambiguous signal -> fall back to RAG but report the low confidence.
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "engine": "rag",
            "confidence": round(confidence, 2),
            "matched": win_matched,
            "reason": f"low confidence ({confidence:.2f}) for '{winner}', defaulting to knowledge base",
        }

    return {
        "engine": winner,
        "confidence": round(confidence, 2),
        "matched": win_matched,
        "reason": f"matched {win_hits} '{winner}' keyword(s): {win_matched}",
    }
