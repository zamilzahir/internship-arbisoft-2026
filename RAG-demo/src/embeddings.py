"""
Two embedding models, built with zero external downloads (this sandbox has
no network access to model hubs like HuggingFace or S3 -- only pypi/npm/
crates registries are reachable). Both are legitimate, distinct embedding
techniques, not toy stand-ins:

  1. TF-IDF          -- sparse, term-weighted bag-of-words vectors.
  2. LSA (TF-IDF+SVD) -- dense vectors from Latent Semantic Analysis, i.e.
                         TF-IDF projected into a lower-rank latent space via
                         truncated SVD. This captures co-occurrence structure
                         that raw TF-IDF misses (near-synonymy, topic drift).

To swap in real neural embeddings (e.g. sentence-transformers/all-MiniLM-L6-v2
or BAAI/bge-small-en) once you have HF access, implement a third class here
following the same `fit`/`embed` interface and register it in embed_compare.py.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from chromadb import EmbeddingFunction, Documents, Embeddings


class TfidfEmbedder(EmbeddingFunction):
    """Sparse TF-IDF vectors, densified for ChromaDB storage."""

    name = "tfidf"

    def __init__(self, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        self.vectorizer.fit(corpus)
        self._fitted = True

    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit() on the corpus before use.")
        matrix = self.vectorizer.transform(input).toarray()
        return matrix.tolist()


class LsaEmbedder(EmbeddingFunction):
    """Dense embeddings from TF-IDF -> Truncated SVD (Latent Semantic Analysis)."""

    name = "lsa"

    def __init__(self, n_components: int = 100, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        n_components = min(self.svd.n_components, max(2, len(corpus) - 1))
        if n_components != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.svd.fit(tfidf_matrix)
        self._fitted = True

    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            raise RuntimeError("LsaEmbedder must be fit() on the corpus before use.")
        tfidf_matrix = self.vectorizer.transform(input)
        reduced = self.svd.transform(tfidf_matrix)
        return reduced.tolist()
