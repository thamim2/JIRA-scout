"""
Pluggable embedding backends.

Jira Scout is designed so the vector-representation layer is swappable:

- TfidfEmbeddingBackend: zero external dependencies beyond scikit-learn,
  runs fully offline, deterministic. Used as the default so the POC works
  out of the box with no API keys or model downloads.
- SentenceTransformerBackend: drop-in replacement using a real sentence
  embedding model (e.g. all-MiniLM-L6-v2) for higher-quality semantic
  matching. Enable it by setting EMBEDDING_BACKEND=sentence-transformer.

This mirrors how the production version of this system (built on live
Jira data) can start with a lightweight backend and graduate to a
transformer-based embedding model without changing any calling code.

IMPORTANT LIMITATION OF THE DEFAULT (TF-IDF) BACKEND: it matches on
literal word overlap, not meaning. Two known failure modes and how this
module mitigates them without needing a model download:

1. Vocabulary mismatch — a query word that never appears in the ticket
   corpus (e.g. "issue" when tickets only ever say "error"/"bug") scores
   exactly zero for that term, even though a human would treat them as
   near-synonyms. Mitigated here with a small domain synonym map applied
   before vectorization, so "issue"/"problem"/"bug" all normalize to the
   same token as "error", and "signin"/"sign-in"/"log in" normalize to
   "login" — consistent on both the indexed corpus and incoming queries.
2. Typos / unseen word forms — "sigin" (missing a letter) shares zero
   tokens with "sign", so it scores zero. Mitigated here with a fuzzy
   correction pass: any query token not found in the fitted vocabulary is
   snapped to its closest match in the corpus vocabulary (difflib), before
   the synonym map is applied.

Neither of these makes TF-IDF a real semantic model — they patch its two
most visible failure modes on a small, informal ticket corpus. For actual
semantic understanding (synonyms and rewordings the map doesn't anticipate,
typos beyond a character or two, meaning-level similarity), use the
SentenceTransformerBackend instead — see requirements-optional.txt.
"""
from __future__ import annotations

import difflib
import os
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np

# Domain synonym groups: every term in a group is normalized to the first
# (canonical) term. Extend this list as real usage surfaces more synonym
# pairs specific to your ticket vocabulary.
SYNONYM_GROUPS = [
    ["login", "log in", "log-in", "signin", "sign in", "sign-in"],
    ["error", "issue", "problem", "bug", "failure", "fail"],
    ["crash", "crashes", "crashing"],
    ["export", "exporting", "exports"],
    ["timeout", "timed out", "times out"],
]


def _build_synonym_map() -> dict:
    mapping = {}
    for group in SYNONYM_GROUPS:
        canonical = group[0].replace(" ", "").replace("-", "")
        for term in group:
            mapping[term.replace(" ", "").replace("-", "")] = canonical
    return mapping


_SYNONYM_MAP = _build_synonym_map()

# Fuzzy typo correction is restricted to this small, curated set of
# canonical domain terms (rather than the full corpus vocabulary).
# Matching against the *entire* vocabulary is unsafe: a typo like "sigin"
# can end up edit-distance-closer to an unrelated incidental word (e.g.
# "signing", from an unrelated ticket that happens to mention "signing
# up") than to the term you actually meant ("login"). Spelling similarity
# is not the same as topical relevance, so correction candidates are kept
# to the handful of terms we know matter, not everything that appears in
# ticket prose.
_CANONICAL_TERMS = sorted({group[0].replace(" ", "").replace("-", "") for group in SYNONYM_GROUPS})


def normalize_synonyms(text: str) -> str:
    """Lowercase, then replace any word that matches a known synonym
    (ignoring internal spaces/hyphens, so 'sign in' and 'signin' both
    normalize the same way) with its canonical form."""
    text = text.lower()
    # Collapse "sign in" / "sign-in" style two/three-token phrases first,
    # so they match the same canonical form as the single-word "signin".
    for group in SYNONYM_GROUPS:
        canonical = group[0].replace(" ", "").replace("-", "")
        for term in sorted(group, key=len, reverse=True):
            if " " in term or "-" in term:
                text = re.sub(re.escape(term), canonical, text)
    tokens = re.findall(r"[a-z0-9]+", text)
    normalized = [_SYNONYM_MAP.get(tok, tok) for tok in tokens]
    return " ".join(normalized)


class EmbeddingBackend(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Fit/prepare the backend on the full ticket corpus."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) matrix of vector representations."""


class TfidfEmbeddingBackend(EmbeddingBackend):
    """Default backend: TF-IDF vectors + cosine similarity, with synonym
    normalization and fuzzy typo correction applied before vectorizing
    (see module docstring for why)."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self._fitted = False
        self._vocab_terms: List[str] = []

    def fit(self, corpus: List[str]) -> None:
        normalized_corpus = [normalize_synonyms(text) for text in corpus]
        self._vectorizer.fit(normalized_corpus)
        self._vocab_terms = list(self._vectorizer.vocabulary_.keys())
        self._fitted = True

    def _correct_typos(self, text: str) -> str:
        """Snap any token not in the fitted vocabulary to its closest
        match among the curated canonical domain terms (not the full
        corpus vocabulary — see module docstring for why). Only applied
        to query text — the indexed corpus is never altered post-fit."""
        corrected_tokens = []
        for token in text.split():
            if token in self._vocab_terms or len(token) < 4:
                corrected_tokens.append(token)
                continue
            match = difflib.get_close_matches(token, _CANONICAL_TERMS, n=1, cutoff=0.6)
            corrected_tokens.append(match[0] if match else token)
        return " ".join(corrected_tokens)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        normalized = [normalize_synonyms(t) for t in texts]
        normalized = [self._correct_typos(t) for t in normalized]
        matrix = self._vectorizer.transform(normalized)
        return matrix.toarray()


class SentenceTransformerBackend(EmbeddingBackend):
    """Optional higher-quality backend using a real sentence embedding model.
    This is the actual fix for vocabulary mismatch / synonyms / typos —
    a transformer embedding model understands that "signin", "login", and
    "sign in trouble" are semantically related without needing a hand-built
    synonym map. Requires `sentence-transformers` (not installed by
    default — see requirements-optional.txt) and downloads model weights
    on first use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: List[str]) -> None:
        # Sentence transformers are pre-trained; nothing to fit.
        return

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, show_progress_bar=False))


def get_backend() -> EmbeddingBackend:
    """Factory selecting the backend via the EMBEDDING_BACKEND env var."""
    backend_name = os.getenv("EMBEDDING_BACKEND", "tfidf").lower()
    if backend_name == "sentence-transformer":
        return SentenceTransformerBackend()
    return TfidfEmbeddingBackend()

