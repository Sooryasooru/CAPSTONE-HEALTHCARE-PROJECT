"""
HAIP Week 3 - RAG Chatbot (Streamlit)
======================================

Interactive demo surface for the Week 3 RAG system, in two tabs:

  1. Guideline Assistant - Q&A over the public WHO/CDC/NICE knowledge base.
  2. Hospital Documents   - upload an institution's own PDF, which is
                            processed into an ISOLATED per-hospital index,
                            then queried privately (no mixing with the base
                            corpus or with other hospitals).

The same RAGEngine.answer() powers both; only the underlying index differs.
This engine will later be exposed via FastAPI for the Week 4 React frontend.

Run from project root:
    PYTHONPATH=. streamlit run src/rag/chat_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.rag.rag_engine import RAGEngine
from src.rag.retriever import Retriever
from src.rag.doc_uploader import ingest_pdf, list_hospitals, HOSPITALS_ROOT

st.set_page_config(page_title="HAIP Guideline Assistant", page_icon="🩺", layout="centered")


# --------------------------------------------------------------------------- #
# Cached engines (load models once)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading RAG engine (first run downloads models)...")
def get_base_engine() -> RAGEngine:
    """Engine over the public WHO/CDC/NICE knowledge base."""
    return RAGEngine()


@st.cache_resource(show_spinner="Loading hospital index...")
def get_hospital_engine(hospital_id: str) -> RAGEngine:
    """Engine over a single hospital's isolated index."""
    base = Path(HOSPITALS_ROOT) / hospital_id
    retriever = Retriever(str(base / "faiss.index"), str(base / "chunks.pkl"))
    return RAGEngine(retriever=retriever)


def render_sources(passages: list[dict]) -> None:
    """Render a de-duplicated source list inside an expander."""
    seen, sources = set(), []
    for p in passages:
        if p["citation"] not in seen:
            seen.add(p["citation"])
            sources.append((p["citation"], p.get("rerank_score", 0.0)))
    with st.expander("Sources"):
        for i, (cit, score) in enumerate(sources, 1):
            st.markdown(f"**[{i}]** {cit}  \n<sub>rerank score: {score:.2f}</sub>",
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header + tabs
# --------------------------------------------------------------------------- #

st.title("🩺 HAIP Guideline Assistant")
tab_chat, tab_docs = st.tabs(["Guideline Assistant", "Hospital Documents"])


# --------------------------------------------------------------------------- #
# TAB 1 - Guideline chat (base corpus)
# --------------------------------------------------------------------------- #

with tab_chat:
    st.caption("Ask a clinical question. Answers are retrieved from WHO/CDC/NICE "
               "guidelines and cited to their source.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("passages"):
                render_sources(msg["passages"])

    prompt = st.chat_input("e.g. How is delirium managed in hospital?", key="base_chat")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching guidelines..."):
                result = get_base_engine().answer(prompt)
            st.markdown(result["answer"])
            if result["passages"]:
                render_sources(result["passages"])
            st.caption(f"Answered in {result['latency_seconds']}s")
        st.session_state.messages.append({
            "role": "assistant", "content": result["answer"],
            "passages": result["passages"],
        })


# --------------------------------------------------------------------------- #
# TAB 2 - Hospital documents (isolated per-hospital index)
# --------------------------------------------------------------------------- #

with tab_docs:
    st.caption("Upload an institution's own procedure documents. Each hospital's "
               "files are processed into a private, isolated index - never mixed "
               "with the public guidelines or with other hospitals.")

    st.subheader("1. Upload a document")
    hospital_name = st.text_input("Hospital / institution name",
                                  placeholder="e.g. St Mary Hospital")
    pdf_file = st.file_uploader("PDF document", type=["pdf"])

    if st.button("Process document", disabled=not (hospital_name and pdf_file)):
        # Save the uploaded file to a temp path for PyMuPDF.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getbuffer())
            tmp_path = tmp.name
        with st.spinner("Extracting, chunking, and embedding..."):
            n = ingest_pdf(hospital_name, tmp_path, doc_label=pdf_file.name)
        Path(tmp_path).unlink(missing_ok=True)
        if n > 0:
            st.success(f"Added {n} chunks to '{hospital_name}'. "
                       "Clearing cache so the new content is searchable.")
            get_hospital_engine.clear()   # force reload of the updated index
        else:
            st.warning("No usable text was extracted. The PDF may be image-only "
                       "(scanned) rather than text.")

    st.divider()
    st.subheader("2. Ask this hospital's documents")

    hospitals = list_hospitals()
    if not hospitals:
        st.info("No hospital documents uploaded yet. Add one above to begin.")
    else:
        chosen = st.selectbox("Select hospital", hospitals)
        hq = st.chat_input("e.g. What is the sepsis screening procedure?",
                           key="hosp_chat")
        if hq:
            with st.chat_message("user"):
                st.markdown(hq)
            with st.chat_message("assistant"):
                with st.spinner(f"Searching {chosen}'s documents..."):
                    result = get_hospital_engine(chosen).answer(hq)
                st.markdown(result["answer"])
                if result["passages"]:
                    render_sources(result["passages"])
                st.caption(f"Answered in {result['latency_seconds']}s "
                           f"\u00b7 index: {chosen}")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("System")
    st.markdown(
        "**Base knowledge**: WHO / CDC / NICE  \n"
        "**Chunks indexed**: 6,832  \n"
        "**Retrieval**: FAISS bi-encoder + cross-encoder re-rank  \n"
        "**LLM**: Gemini 2.5 Flash (grounded)  \n"
        "**Hospital docs**: PyMuPDF \u2192 isolated index"
    )
    st.divider()
    st.caption("Proof-of-concept on public/synthetic data. Answers are grounded "
               "and cited. A triage/reference aid, not medical advice. Emergency "
               "questions are redirected to urgent care.")