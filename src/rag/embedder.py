"""
HAIP Week 3 - RAG Embedder
===========================

Turns the chunked knowledge base into a searchable vector index.

Pipeline:
    load knowledge_base.jsonl
        -> encode each chunk with Sentence Transformers (batched)
        -> L2-normalise vectors (so inner product == cosine similarity)
        -> build a FAISS IndexFlatIP
        -> save index + aligned chunk metadata

Outputs:
    data/processed/faiss.index   - the vector index
    data/processed/chunks.pkl    - chunk metadata, row-aligned to the index

Model choice:
    all-MiniLM-L6-v2  - 384-dim, ~80 MB, fast on CPU, RAM-safe.
    A larger model (e.g. all-mpnet-base-v2, 768-dim) would improve recall
    but roughly doubles memory and encode time. Justified in the Week 3
    tool-comparison document.

Run from project root:
    python -m src.rag.embedder
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

KNOWLEDGE_BASE_PATH = "data/processed/knowledge_base.jsonl"
INDEX_PATH = "data/processed/faiss.index"
METADATA_PATH = "data/processed/chunks.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384                # output dimension of the model above
BATCH_SIZE = 64               # chunks encoded per batch (RAM safety)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embedder")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_chunks(path: str) -> list[dict]:
    """Read the JSONL knowledge base into a list of chunk records."""
    chunks: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


def citation_label(chunk: dict) -> str:
    """
    Build a human-readable citation label, falling back gracefully when
    title/url are missing (stored as the literal string 'None' in this corpus).
    """
    title = chunk.get("title")
    source = chunk.get("source", "unknown").upper()
    if title and title != "None":
        return f"{source}: {title}"
    return f"{source} (doc {chunk['doc_id'][:8]})"


# --------------------------------------------------------------------------- #
# Embedding + index build
# --------------------------------------------------------------------------- #

def build_index() -> None:
    """Encode all chunks and persist a FAISS index plus aligned metadata."""
    chunks = load_chunks(KNOWLEDGE_BASE_PATH)
    texts = [c["text"] for c in chunks]

    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    logger.info("Encoding %d chunks (batch size %d)...", len(texts), BATCH_SIZE)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors -> inner product = cosine
    ).astype("float32")

    logger.info("Embeddings shape: %s", embeddings.shape)

    # Build an exact inner-product index (cosine on normalised vectors).
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info("FAISS index built with %d vectors", index.ntotal)

    # Persist index.
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    logger.info("Saved index -> %s", INDEX_PATH)

    # Persist metadata, row-aligned to the index. Add a citation label.
    for c in chunks:
        c["citation"] = citation_label(c)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info("Saved metadata (%d records) -> %s", len(chunks), METADATA_PATH)

    logger.info("Done. Knowledge base is now searchable.")


if __name__ == "__main__":
    build_index()