"""
Core semantic search engine for Jira Scout.

Loads the ticket corpus, builds vector representations via the configured
EmbeddingBackend, and answers similarity queries using cosine similarity.
This is the analogue of the resume bullet:

    "Designed an LLM-powered semantic search engine over Jira that
    analyses incoming problem statements and surfaces the most relevant
    existing tickets using embedding models and vector similarity."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .embeddings import EmbeddingBackend, get_backend
from .models import Ticket


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec) + 1e-10
    matrix_norms = np.linalg.norm(matrix, axis=1) + 1e-10
    return (matrix @ query_vec) / (matrix_norms * query_norm)


class TicketSearchEngine:
    def __init__(self, tickets: List[Ticket], backend: EmbeddingBackend | None = None):
        self.tickets = tickets
        self.backend = backend or get_backend()
        self._matrix: np.ndarray | None = None
        self._build_index()

    @classmethod
    def from_json_file(cls, path: str | Path, backend: EmbeddingBackend | None = None) -> "TicketSearchEngine":
        data = json.loads(Path(path).read_text())
        tickets = [Ticket(**item) for item in data]
        return cls(tickets, backend=backend)

    def _build_index(self) -> None:
        corpus = [t.searchable_text for t in self.tickets]
        self.backend.fit(corpus)
        self._matrix = self.backend.encode(corpus)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> List[Tuple[Ticket, float]]:
        """Return the top_k most similar tickets to `query`, ranked by cosine similarity."""
        if self._matrix is None:
            raise RuntimeError("Index not built.")
        query_vec = self.backend.encode([query])[0]
        scores = _cosine_similarity(query_vec, self._matrix)
        ranked_idx = np.argsort(scores)[::-1]

        results: List[Tuple[Ticket, float]] = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append((self.tickets[idx], score))
            if len(results) >= top_k:
                break
        return results

    def duplicate_candidates(self, new_title: str, new_description: str, top_k: int = 3) -> List[Tuple[Ticket, float]]:
        """Convenience wrapper matching the 'incoming ticket' workflow:
        given a freshly-submitted ticket's title + description, find
        existing tickets it might be a duplicate of."""
        query = f"{new_title}. {new_description}"
        return self.search(query, top_k=top_k)
