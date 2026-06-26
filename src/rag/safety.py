"""
HAIP Week 3 - Safety Guard
===========================

A first-line safety net that runs BEFORE retrieval. If a question contains
red-flag emergency language, the system returns an urgent-care redirect
instead of a guideline answer - because a retrieval-grounded summary is the
wrong response to "I have chest pain and can't breathe".

IMPORTANT - honest scope:
    This is a keyword/pattern guard, NOT a clinical triage classifier.
    It deliberately errs toward caution (over-flagging is safer than
    missing an emergency). A production system would use a trained
    intent/triage model. This guard demonstrates responsible-AI design
    for a medical context; it is not a substitute for professional triage.

Two categories are handled distinctly:
    * MEDICAL emergencies  -> direct to emergency services.
    * MENTAL-HEALTH crisis -> direct to crisis support (handled gently,
                              never with a guideline retrieval).
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Red-flag patterns
# --------------------------------------------------------------------------- #
# Patterns are intentionally broad. Word boundaries avoid matching inside
# unrelated words. Phrasing covers first-person urgency ("I can't breathe").

MEDICAL_EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bcan('?t| ?not) breathe\b",
    r"\bdifficulty breathing\b",
    r"\bstruggling to breathe\b",
    r"\bchoking\b",
    r"\bstroke\b",
    r"\bface drooping\b",
    r"\bslurred speech\b",
    r"\bsudden (weakness|numbness)\b",
    r"\bsevere bleeding\b",
    r"\bbleeding (a lot|heavily|uncontrollably)\b",
    r"\banaphylaxis\b",
    r"\bsevere allergic reaction\b",
    r"\bunconscious\b",
    r"\bnot responding\b",
    r"\bseizure\b",
    r"\boverdose\b",
]

MENTAL_HEALTH_CRISIS_PATTERNS = [
    r"\bsuicid(e|al)\b",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bself[- ]harm\b",
    r"\bhurt myself\b",
    r"\bwant to die\b",
]

_MEDICAL_RE = re.compile("|".join(MEDICAL_EMERGENCY_PATTERNS), re.IGNORECASE)
_CRISIS_RE = re.compile("|".join(MENTAL_HEALTH_CRISIS_PATTERNS), re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Redirect messages
# --------------------------------------------------------------------------- #

MEDICAL_REDIRECT = (
    "**This may be a medical emergency.**\n\n"
    "If you or someone else is experiencing these symptoms, contact your "
    "local emergency services immediately (for example, 112 in India/EU, "
    "911 in the US, 999 in the UK) or go to the nearest emergency "
    "department.\n\n"
    "I'm a guideline-information tool and can't assess urgent symptoms - "
    "please seek immediate help from a medical professional."
)

CRISIS_REDIRECT = (
    "**It sounds like you may be going through something very difficult, "
    "and you deserve support right now.**\n\n"
    "Please consider reaching out to a crisis support service or a trusted "
    "person. You can contact your local emergency services, or a mental "
    "health helpline in your region - for example, in India the Tele-MANAS "
    "helpline (14416) offers free 24/7 support.\n\n"
    "I'm a guideline-information tool and not able to provide crisis care, "
    "but people are available to help."
)


# --------------------------------------------------------------------------- #
# Public check
# --------------------------------------------------------------------------- #

def safety_check(question: str) -> dict | None:
    """
    Inspect a question for emergency red flags.

    Returns None if the question is clear to proceed to RAG. Otherwise
    returns a redirect dict in the same shape as RAGEngine.answer(), so the
    caller can return it directly.
    """
    if _CRISIS_RE.search(question):
        return {
            "question": question,
            "answer": CRISIS_REDIRECT,
            "citations": [],
            "passages": [],
            "latency_seconds": 0.0,
            "safety_flag": "mental_health_crisis",
        }

    if _MEDICAL_RE.search(question):
        return {
            "question": question,
            "answer": MEDICAL_REDIRECT,
            "citations": [],
            "passages": [],
            "latency_seconds": 0.0,
            "safety_flag": "medical_emergency",
        }

    return None