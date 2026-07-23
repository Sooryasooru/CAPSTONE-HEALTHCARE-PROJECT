"""
HAIP Week 3 - RAG Engine (LangChain implementation)
====================================================

A LangChain-native version of the RAG pipeline, built ALONGSIDE the
hand-written engine (rag_engine.py) rather than replacing it.

Why both exist:
    The hand-built engine demonstrates understanding of the internals
    (manual FAISS search, manual re-rank, manual prompt assembly).
    This LangChain version demonstrates fluency with the standard
    industry framework, wiring the same components through LangChain's
    abstractions: an Embeddings wrapper, a FAISS vectorstore + retriever,
    a ChatModel, a prompt template, and an LCEL chain.

It reuses the SAME artifacts already built:
    data/processed/faiss.index   (rebuilt into a LangChain FAISS store)
    data/processed/chunks.pkl    (the chunk texts + metadata)

Run a smoke test from project root:
    python -m src.rag.rag_engine_langchain
"""

from __future__ import annotations

import logging
import os
import pickle
import time
import warnings

# --- Clean handling of two known, harmless framework messages ------------- #
# 1. langchain-community is flagged as "being sunset", but FAISS has no
#    standalone replacement package yet, so this import remains correct.
# 2. faiss logs an INFO line when it falls back from an AVX2-optimized build
#    to the baseline CPU build - the fallback works fine; quiet the chatter.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain_community")
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.rag.safety import safety_check

load_dotenv()

KNOWLEDGE_BASE_PATH = "data/processed/knowledge_base.jsonl"
METADATA_PATH = "data/processed/chunks.pkl"
LC_FAISS_DIR = "data/processed/lc_faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_langchain")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful medical-guideline assistant. Answer the question "
     "using ONLY the numbered context passages. If the answer is not in the "
     "context, say the guidelines do not cover it - do not invent clinical "
     "detail. Cite passage numbers like [1], [2]. Be concise and factual."),
    ("human", "Context passages:\n\n{context}\n\nQuestion: {question}"),
])


class LangChainRAGEngine:
    """RAG pipeline assembled from LangChain components (LCEL)."""

    def __init__(self) -> None:
        logger.info("Loading chunks from %s", METADATA_PATH)
        with open(METADATA_PATH, "rb") as f:
            chunks = pickle.load(f)

        # Convert our chunk dicts into LangChain Documents.
        docs = [
            Document(page_content=c["text"],
                     metadata={"citation": c["citation"], "doc_id": c["doc_id"]})
            for c in chunks
        ]

        logger.info("Loading embeddings: %s", EMBED_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # Build a LangChain FAISS vectorstore from the documents.
        # (Re-embeds via LangChain's interface so the store is LC-native.)
        # Load the pre-built LangChain FAISS store if it exists; only embed
        # all 6,832 chunks when the store is missing (first run / rebuild).
        if os.path.isdir(LC_FAISS_DIR):
            logger.info("Loading saved LangChain FAISS store from %s", LC_FAISS_DIR)
            self.vectorstore = FAISS.load_local(
                LC_FAISS_DIR, self.embeddings,
                allow_dangerous_deserialization=True)
        else:
            logger.info("No saved store - building LangChain FAISS vectorstore (%d docs)...", len(docs))
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.vectorstore.save_local(LC_FAISS_DIR)
            logger.info("Saved LangChain FAISS store -> %s", LC_FAISS_DIR)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        logger.info("Loading Gemini chat model: %s", GEMINI_MODEL)
        if os.getenv("HAIP_RAG_BACKEND") == "groq":
            # Eval backend: Groq's free quota is far larger than Gemini's.
            self.llm = ChatGroq(
                model=os.getenv("HAIP_RAG_MODEL", "llama-3.1-8b-instant"),
                groq_api_key=os.environ["GROQ_API_KEY"],
                temperature=0,
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=os.environ["GEMINI_API_KEY"],
            )

        # LCEL chain: retrieve -> format -> prompt -> LLM -> string.
        self.chain = (
            {"context": self.retriever | self._format_docs,
             "question": RunnablePassthrough()}
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        logger.info("LangChain RAG engine ready.")

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """Number the retrieved passages for the prompt."""
        return "\n\n".join(
            f"[{i}] (source: {d.metadata.get('citation', '?')})\n{d.page_content[:1500]}"
            for i, d in enumerate(docs, 1)
        )

    def answer(self, question: str) -> dict:
        """Run the chain, with the same safety guard as the main engine."""
        flagged = safety_check(question)
        if flagged is not None:
            logger.warning("Safety redirect: %s", flagged["safety_flag"])
            return flagged

        start = time.time()
        retrieved = self.retriever.invoke(question)
        answer_text = self.chain.invoke(question)
        latency = round(time.time() - start, 2)

        return {
            "question": question,
            "answer": answer_text,
            "citations": [d.metadata.get("citation", "?") for d in retrieved],
            "latency_seconds": latency,
        }


if __name__ == "__main__":
    engine = LangChainRAGEngine()
    q = "How is delirium managed in hospital patients?"
    out = engine.answer(q)
    print("\n" + "=" * 70)
    print("QUESTION:", out["question"])
    print("=" * 70)
    print("\nANSWER:\n", out["answer"])
    print("\nCITATIONS:")
    for i, c in enumerate(out["citations"], 1):
        print(f"  [{i}] {c}")
    print(f"\nLatency: {out['latency_seconds']}s")