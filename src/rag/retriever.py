"""
HAIP Week 3 - RAG Retriever (retrieve + re-rank)
=================================================

Two-stage retrieval, the core "advanced RAG" component:

    Stage 1 - bi-encoder retrieval (fast, wide):
        embed the query, search FAISS, return top-K candidates.
        The bi-encoder scored query and chunks separately, so it is
        fast but approximate -> cast a wide net (default K=20).

    Stage 2 - cross-encoder re-rank (slow, precise):
        score each (query, candidate) PAIR jointly with a cross-encoder.
        Far more accurate but too slow for the whole corpus, so it runs
        only on the Stage-1 candidates -> keep the best (default 5).

Why two stages:
    bi-encoder alone = fast but misses nuance.
    cross-encoder alone = accurate but too slow on 6,832 chunks.
    Together = FAISS narrows the field, cross-encoder picks the winners.

Models load lazily (on first query) to keep import/startup light.

Run a quick smoke test from project root:
    python -m src.rag.retriever
"""

from __future__ import annotations

import logging
import pickle

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

INDEX_PATH = "data/processed/faiss.index"
METADATA_PATH = "data/processed/chunks.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"               # must match embedder.py
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVE = 20      # Stage 1: wide candidate net
TOP_K_RERANK = 5         # Stage 2: precise final passages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retriever")


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #

class Retriever:
    """Loads the index + metadata + models and serves ranked passages."""

    def __init__(self, index_path: str = INDEX_PATH,
                 metadata_path: str = METADATA_PATH) -> None:
        logger.info("Loading FAISS index from %s", index_path)
        self.index = faiss.read_index(index_path)

        logger.info("Loading chunk metadata from %s", metadata_path)
        with open(metadata_path, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index/metadata mismatch: {self.index.ntotal} vectors "
                f"vs {len(self.chunks)} chunks"
            )

        # Models are loaded on first use (lazy) to keep startup light.
        self._embed_model: SentenceTransformer | None = None
        self._rerank_model: CrossEncoder | None = None

    # -- lazy model accessors ----------------------------------------------- #

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            logger.info("Loading embedding model: %s", EMBED_MODEL)
            self._embed_model = SentenceTransformer(EMBED_MODEL)
        return self._embed_model

    @property
    def rerank_model(self) -> CrossEncoder:
        if self._rerank_model is None:
            logger.info("Loading re-rank model: %s", RERANK_MODEL)
            self._rerank_model = CrossEncoder(RERANK_MODEL)
        return self._rerank_model

    # -- stage 1 ------------------------------------------------------------ #

    def _retrieve(self, query: str, k: int) -> list[dict]:
        """Bi-encoder + FAISS: return the top-k candidate chunks."""
        q_vec = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(q_vec, k)

        candidates: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:                      # FAISS pads with -1 if too few hits
                continue
            chunk = dict(self.chunks[idx])   # copy so we don't mutate the store
            chunk["retrieval_score"] = float(score)
            candidates.append(chunk)
        return candidates

    # -- stage 2 ------------------------------------------------------------ #

    def _rerank(self, query: str, candidates: list[dict], k: int) -> list[dict]:
        """Cross-encoder: jointly score (query, chunk) pairs, keep top-k."""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.rerank_model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:k]

    # -- public API --------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_rerank: int = TOP_K_RERANK,
    ) -> list[dict]:
        """
        Full two-stage retrieval.

        Returns a list of chunk dicts (best first), each containing the
        original text + citation plus `retrieval_score` and `rerank_score`.
        """
        candidates = self._retrieve(query, top_k_retrieve)
        ranked = self._rerank(query, candidates, top_k_rerank)
        logger.info(
            "Query retrieved %d candidates -> kept %d after re-rank",
            len(candidates), len(ranked),
        )
        return ranked


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    retriever = Retriever()
    sample = "What are the recommended treatments for hypertension?"
    print(f"\nQuery: {sample}\n" + "=" * 70)
    for rank, hit in enumerate(retriever.retrieve(sample), 1):
        print(f"\n[{rank}] {hit['citation']}")
        print(f"    retrieval={hit['retrieval_score']:.3f}  "
              f"rerank={hit['rerank_score']:.3f}")
        print(f"    {hit['text'][:200].strip()}...")
        