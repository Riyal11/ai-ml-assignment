"""Hybrid sparse + dense retrieval stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked document from hybrid search."""

    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class HybridRetriever:
    """BM25 + dense embeddings with RRF / weighted fusion (structure only)."""

    def __init__(self, alpha: float = 0.5, rrf_k: int = 60) -> None:
        """Configure fusion parameters.

        Args:
            alpha: Weight for BM25 in linear fusion (ignored when using RRF).
            rrf_k: Reciprocal Rank Fusion constant (typically 60).
        """
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        """Return top-k documents for ``query`` (not implemented).

        Args:
            query: Natural-language search string.
            top_k: Maximum hits to return.

        Raises:
            NotImplementedError: Always — this is a design stub.
        """
        raise NotImplementedError("HybridRetriever.search is a design stub")
