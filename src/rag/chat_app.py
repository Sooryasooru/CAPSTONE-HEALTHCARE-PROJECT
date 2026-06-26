"""
HAIP Week 3 - RAG Chatbot (Streamlit)
======================================

An interactive medical-guideline Q&A interface over the RAG engine.
Ask a question in plain English; the system retrieves relevant WHO/CDC/NICE
guideline passages, generates a grounded answer, and shows its sources.

This is the Week 3 demo surface. The same RAGEngine.answer() method will
later be exposed via FastAPI for the Week 4 React frontend - so the engine
is unchanged; this is only a thin presentation layer.

Run from project root:
    streamlit run src/rag/chat_app.py
"""

from __future__ import annotations

import streamlit as st

from src.rag.rag_engine import RAGEngine

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="HAIP Guideline Assistant",
    page_icon="🩺",
    layout="centered",
)


# --------------------------------------------------------------------------- #
# Cached engine (loads models once, not per question)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading RAG engine (first run downloads models)...")
def get_engine() -> RAGEngine:
    """Build the RAG engine once and reuse across reruns."""
    return RAGEngine()


# --------------------------------------------------------------------------- #
# Sidebar - system info (makes the system legible to a reviewer)
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("System")
    st.markdown(
        "**Knowledge base**: WHO / CDC / NICE clinical guidelines  \n"
        "**Chunks indexed**: 6,832  \n"
        "**Retrieval**: bi-encoder (FAISS) + cross-encoder re-rank  \n"
        "**Embeddings**: all-MiniLM-L6-v2  \n"
        "**LLM**: Gemini 2.5 Flash (grounded)"
    )
    st.divider()
    st.caption(
        "Proof-of-concept on public guideline text. "
        "Answers are grounded in retrieved passages and cited. "
        "This is a triage/reference aid, not medical advice."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("🩺 HAIP Guideline Assistant")
st.caption(
    "Ask a clinical question. Answers are retrieved from WHO/CDC/NICE "
    "guidelines and cited to their source."
)


# --------------------------------------------------------------------------- #
# Conversation state
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** {src['citation']}  \n"
                                f"<sub>rerank score: {src['score']:.2f}</sub>",
                                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Chat input + response
# --------------------------------------------------------------------------- #

prompt = st.chat_input("e.g. How is acute heart failure managed?")

if prompt:
    # Show + store the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate the grounded answer.
    with st.chat_message("assistant"):
        with st.spinner("Searching guidelines and composing an answer..."):
            engine = get_engine()
            result = engine.answer(prompt)

        st.markdown(result["answer"])

        # Build a de-duplicated source list with rerank scores.
        sources = []
        seen = set()
        for p in result["passages"]:
            key = p["citation"]
            if key not in seen:
                seen.add(key)
                sources.append({
                    "citation": p["citation"],
                    "score": p.get("rerank_score", 0.0),
                })

        with st.expander("Sources"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**[{i}]** {src['citation']}  \n"
                            f"<sub>rerank score: {src['score']:.2f}</sub>",
                            unsafe_allow_html=True)

        st.caption(f"Answered in {result['latency_seconds']}s")

    # Store assistant message with its sources for replay.
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources,
    })