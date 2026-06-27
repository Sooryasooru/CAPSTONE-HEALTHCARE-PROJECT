"""
HAIP Week 3 - Hospital Document Uploader
=========================================

Lets an institution add its own PDF documents and make them searchable,
WITHOUT mixing them into the public WHO/CDC/NICE knowledge base.

Each hospital gets an isolated index:
    data/hospitals/<hospital_id>/faiss.index
    data/hospitals/<hospital_id>/chunks.pkl

Pipeline (reuses Week 3 logic, but starts from a real PDF):
    PDF -> PyMuPDF text extraction (per page)
        -> recursive chunking with overlap
        -> embed with the same model as the base index
        -> build + save a per-hospital FAISS index

This is where PyMuPDF is actually exercised: unlike the pre-extracted
guidelines corpus, uploaded files are real PDFs that must be parsed.

Programmatic use:
    from src.rag.doc_uploader import ingest_pdf
    n = ingest_pdf("hospital_a", "/path/to/procedures.pdf")
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Reuse the base chunking parameters/logic for consistency.
from src.rag.pdf_processor import chunk_text, clean

HOSPITALS_ROOT = "data/hospitals"
EMBED_MODEL = "all-MiniLM-L6-v2"     # must match embedder.py
EMBED_DIM = 384
BATCH_SIZE = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("doc_uploader")

# Load the embedding model once at module level (lazy-safe singleton).
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _safe_id(name: str) -> str:
    """Turn a hospital name into a filesystem-safe folder id."""
    return re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_") or "unnamed"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract_pdf_text(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, one record per page.

    Returns a list of {page, text} dicts. Empty pages are skipped.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, 1):
        text = clean(page.get_text())
        if len(text) >= 50:            # skip near-empty pages
            pages.append({"page": i, "text": text})
    doc.close()
    logger.info("Extracted %d non-empty pages from %s", len(pages), pdf_path)
    return pages


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #

def ingest_pdf(hospital_name: str, pdf_path: str, doc_label: str | None = None) -> int:
    """
    Process one PDF into the named hospital's isolated index.

    If the hospital already has an index, the new chunks are appended
    (so multiple documents accumulate). Returns the number of new chunks.
    """
    hid = _safe_id(hospital_name)
    out_dir = Path(HOSPITALS_ROOT) / hid
    out_dir.mkdir(parents=True, exist_ok=True)

    label = doc_label or Path(pdf_path).stem

    # 1. Extract + chunk.
    pages = extract_pdf_text(pdf_path)
    new_chunks: list[dict] = []
    for pg in pages:
        for j, chunk in enumerate(chunk_text(pg["text"])):
            new_chunks.append({
                "chunk_id": f"{label}_p{pg['page']}_{j}",
                "doc_id": label,
                "source": hospital_name,
                "title": label,
                "url": "uploaded",
                "page": pg["page"],
                "text": chunk,
                "citation": f"{hospital_name}: {label} (p.{pg['page']})",
            })

    if not new_chunks:
        logger.warning("No usable text extracted from %s", pdf_path)
        return 0

    # 2. Embed.
    model = _get_model()
    texts = [c["text"] for c in new_chunks]
    logger.info("Embedding %d new chunks...", len(texts))
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype("float32")

    # 3. Load existing index/metadata for this hospital, or start fresh.
    index_path = out_dir / "faiss.index"
    meta_path = out_dir / "chunks.pkl"

    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            chunks = pickle.load(f)
        logger.info("Appending to existing index (%d chunks)", len(chunks))
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        chunks = []

    # 4. Add + save.
    index.add(embeddings)
    chunks.extend(new_chunks)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    logger.info("Hospital '%s' index now holds %d chunks (+%d)",
                hospital_name, index.ntotal, len(new_chunks))
    return len(new_chunks)


def list_hospitals() -> list[str]:
    """Return the hospital ids that currently have an index."""
    root = Path(HOSPITALS_ROOT)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "faiss.index").exists())


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.rag.doc_uploader <hospital_name> <pdf_path>")
        sys.exit(1)
    n = ingest_pdf(sys.argv[1], sys.argv[2])
    print(f"Ingested {n} chunks. Hospitals: {list_hospitals()}")