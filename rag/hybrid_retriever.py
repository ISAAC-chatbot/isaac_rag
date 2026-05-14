"""
Hybrid retriever: combines BM25 keyword search with dense vector similarity.
Works with both local document chunks and Semantic Scholar paper texts.
"""
import logging
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny BM25 implementation (no external deps)
# ---------------------------------------------------------------------------

class _BM25:
    """Okapi BM25 over a list of tokenised documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[List[str]] = []
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0

    def index(self, tokenised_docs: List[List[str]]) -> None:
        self._docs = tokenised_docs
        n = len(tokenised_docs)
        dl_sum = sum(len(d) for d in tokenised_docs)
        self._avgdl = dl_sum / n if n else 1.0

        df: Dict[str, int] = defaultdict(int)
        for doc in tokenised_docs:
            for term in set(doc):
                df[term] += 1

        self._idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc = self._docs[doc_idx]
        dl = len(doc)
        tf_map: Dict[str, int] = defaultdict(int)
        for t in doc:
            tf_map[t] += 1

        score = 0.0
        for term in query_tokens:
            if term not in self._idf:
                continue
            idf = self._idf[term]
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * dl / self._avgdl
            )
            score += idf * (numerator / denominator)
        return score

    def top_k(self, query_tokens: List[str], k: int = 5) -> List[Tuple[int, float]]:
        scores = [
            (idx, self.score(query_tokens, idx))
            for idx in range(len(self._docs))
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def _simple_tokenise(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Vector store (in-memory; swap for FAISS / OpenSearch in production)
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HybridRetriever:
    """
    Retrieves the top-k most relevant text chunks for a query by combining:
      * BM25 keyword score
      * Dense cosine similarity (using any callable embedder)

    Parameters
    ----------
    embed_fn:
        Callable that takes a list[str] and returns list[list[float]].
        If None, only BM25 is used.
    alpha:
        Weight for vector score in the final combined score (0 = BM25 only,
        1 = vector only). Default 0.5.
    """

    def __init__(
        self,
        embed_fn=None,
        alpha: float = 0.5,
    ):
        self.embed_fn = embed_fn
        self.alpha = alpha
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._vectors: List[List[float]] = []
        self._bm25 = _BM25()

    # ------------------------------------------------------------------
    # Build / update index
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]

        self._texts.extend(texts)
        self._metadata.extend(metadatas)

        if self.embed_fn:
            try:
                vecs = self.embed_fn(texts)
                self._vectors.extend(vecs)
            except Exception as exc:
                logger.warning("Embedding failed, falling back to BM25 only: %s", exc)

        tokenised = [_simple_tokenise(t) for t in self._texts]
        self._bm25.index(tokenised)

    def clear(self) -> None:
        self._texts = []
        self._metadata = []
        self._vectors = []
        self._bm25 = _BM25()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns up to top_k results as dicts with keys:
          text, score, bm25_score, vec_score, metadata
        """
        if not self._texts:
            return []

        query_tokens = _simple_tokenise(query)
        n = len(self._texts)

        # BM25 scores (normalised 0-1)
        bm25_raw = [self._bm25.score(query_tokens, i) for i in range(n)]
        max_bm25 = max(bm25_raw) if bm25_raw else 1.0
        if max_bm25 == 0:
            max_bm25 = 1.0
        bm25_norm = [s / max_bm25 for s in bm25_raw]

        # Vector scores (normalised 0-1)
        if self.embed_fn and len(self._vectors) == n:
            try:
                q_vec = self.embed_fn([query])[0]
                vec_raw = [_cosine(q_vec, v) for v in self._vectors]
            except Exception as exc:
                logger.warning("Query embedding failed: %s", exc)
                vec_raw = [0.0] * n
        else:
            vec_raw = [0.0] * n

        max_vec = max(vec_raw) if vec_raw else 1.0
        if max_vec == 0:
            max_vec = 1.0
        vec_norm = [s / max_vec for s in vec_raw]

        # Combined score
        combined = [
            (1 - self.alpha) * bm + self.alpha * vv
            for bm, vv in zip(bm25_norm, vec_norm)
        ]

        ranked = sorted(
            range(n), key=lambda i: combined[i], reverse=True
        )

        results = []
        for idx in ranked[:top_k]:
            meta = self._metadata[idx]
            # User isolation: skip if meta specifies a different user
            if user_id and meta.get("user_id") and meta["user_id"] != user_id:
                continue
            results.append(
                {
                    "text": self._texts[idx],
                    "score": combined[idx],
                    "bm25_score": bm25_norm[idx],
                    "vec_score": vec_norm[idx],
                    "metadata": meta,
                }
            )
        return results
