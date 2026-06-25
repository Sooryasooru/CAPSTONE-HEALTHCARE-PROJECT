"""
- RAG Document Processor
=====================================

Builds the medical knowledge base for the RAG engine.

Pipeline:
    load Arrow dataset
        -> filter to trusted sources (WHO / CDC / NICE)
        -> balance (cap N docs per source)
        -> drop oversized outlier documents (RAM safety)
        -> recursive chunking with overlap
        -> save chunks as JSONL (one chunk per line)

Note on the name "pdf_processor":
    The Week 3 spec names a PDF-processing step. Our corpus
    (epfl-llm/guidelines) arrives pre-extracted as text in the
    `clean_text` field, so no PDF parsing (e.g. PyMuPDF) is required.
    This module performs the equivalent document-processing role:
    cleaning and chunking source text into retrieval-ready passages.

Run from project root:
    python -m src.rag.pdf_processor
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from datasets import load_from_disk

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATASET_PATH = "data/raw_legacy/medical_documents/guidelines_dataset"
OUTPUT_PATH = "data/processed/knowledge_base.jsonl"

TRUSTED_SOURCES = ["who", "cdc", "nice"]
DOCS_PER_SOURCE = 200          # balanced sampling -> ~600 docs total
MAX_DOC_CHARS = 40_000         # skip outlier mega-documents (RAM safety)

CHUNK_SIZE = 2_000             # ~500 tokens per chunk
CHUNK_OVERLAP = 200            # ~50 tokens shared between neighbours
MIN_CHUNK_CHARS = 100          # discard tiny fragments

# Recursive split separators, tried in order (coarse -> fine).
SEPARATORS = ["\n\n", "\n", ". ", " "]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pdf_processor")


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

def clean(text: str) -> str:
    """Normalise whitespace and strip control noise from raw text."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)     # cap blank-line runs
    return text.strip()


# --------------------------------------------------------------------------- #
# Recursive chunking
# --------------------------------------------------------------------------- #

def _split_recursive(text: str, separators: list[str]) -> list[str]:
    """
    Split text into pieces no larger than CHUNK_SIZE, preferring coarse
    boundaries (paragraphs) before falling back to finer ones (sentences,
    words). This keeps semantically related text together instead of
    cutting blindly at a fixed character count.
    """
    if len(text) <= CHUNK_SIZE:
        return [text]

    if not separators:
        # No separators left: hard-split as a last resort.
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    sep, *rest = separators
    parts = text.split(sep)

    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = part if not buffer else buffer + sep + part
        if len(candidate) <= CHUNK_SIZE:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            # The single part itself may still be too big -> recurse finer.
            if len(part) > CHUNK_SIZE:
                chunks.extend(_split_recursive(part, rest))
                buffer = ""
            else:
                buffer = part
    if buffer:
        chunks.append(buffer)
    return chunks


def _add_overlap(chunks: list[str]) -> list[str]:
    """Prepend the tail of each chunk to the next, so context isn't severed."""
    if CHUNK_OVERLAP <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-CHUNK_OVERLAP:]
        out.append(tail + " " + cur)
    return out


def chunk_text(text: str) -> list[str]:
    """Full chunking: recursive split, add overlap, drop tiny fragments."""
    pieces = _split_recursive(text, SEPARATORS)
    pieces = _add_overlap(pieces)
    return [p.strip() for p in pieces if len(p.strip()) >= MIN_CHUNK_CHARS]


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #

def build_knowledge_base() -> None:
    """Load, filter, balance, chunk, and persist the knowledge base."""
    logger.info("Loading dataset from %s", DATASET_PATH)
    ds = load_from_disk(DATASET_PATH)["train"]

    logger.info("Filtering to trusted sources: %s", TRUSTED_SOURCES)
    ds = ds.filter(lambda x: x["source"] in TRUSTED_SOURCES)

    # Balanced sampling + outlier cap, tracked per source.
    kept_per_source: dict[str, int] = {s: 0 for s in TRUSTED_SOURCES}
    selected: list[dict] = []
    skipped_big = 0

    for row in ds:
        src = row["source"]
        if kept_per_source[src] >= DOCS_PER_SOURCE:
            continue
        text = clean(row["clean_text"])
        if len(text) > MAX_DOC_CHARS:
            skipped_big += 1
            continue
        if len(text) < MIN_CHUNK_CHARS:
            continue
        selected.append({
            "doc_id": row["id"],
            "source": src,
            "title": row["title"],
            "url": row["url"],
            "text": text,
        })
        kept_per_source[src] += 1
        if all(kept_per_source[s] >= DOCS_PER_SOURCE for s in TRUSTED_SOURCES):
            break

    logger.info("Selected docs per source: %s", kept_per_source)
    logger.info("Skipped %d oversized docs (> %d chars)", skipped_big, MAX_DOC_CHARS)

    # Chunk every selected document.
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as f:
        for doc in selected:
            for i, chunk in enumerate(chunk_text(doc["text"])):
                record = {
                    "chunk_id": f"{doc['doc_id']}_{i}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "text": chunk,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info("Wrote %d chunks from %d docs -> %s",
                total_chunks, len(selected), OUTPUT_PATH)
    logger.info("Done. Average %.1f chunks/doc",
                total_chunks / max(len(selected), 1))


if __name__ == "__main__":
    build_knowledge_base()
