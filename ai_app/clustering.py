from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN

from ai_app.evaluation import summarize_clusters
from ai_app.retrieval import EmbeddingEncoder


@dataclass(frozen=True)
class FAQClusterDocument:
    post_id: str
    title: str
    keywords: str
    content: str

    @property
    def text(self) -> str:
        return build_clustering_text(self.title, self.keywords, self.content)


@dataclass(frozen=True)
class ClusteringConfig:
    min_cluster_size: int = 4
    min_samples: int = 2
    cluster_selection_method: str = "eom"

    def __post_init__(self) -> None:
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if self.cluster_selection_method not in {"eom", "leaf"}:
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")


@dataclass(frozen=True)
class ClusterResult:
    status: str
    labels: np.ndarray
    embeddings: np.ndarray
    diagnostics: Dict[str, Any]
    config: ClusteringConfig
    reason: str | None = None


def build_clustering_text(title: str, keywords: str, content: str = "") -> str:
    """Build a compact topical representation without appending the raw body."""
    normalized_title = str(title or "").strip()
    normalized_keywords = " ".join(
        token
        for token in re.split(r"[\s,]+", str(keywords or "").strip())
        if token
    )
    parts = [part for part in (normalized_title,) if part]
    if normalized_keywords:
        parts.append(f"핵심 키워드: {normalized_keywords}")
    return "\n".join(parts)


def cluster_documents(
    documents: Sequence[FAQClusterDocument],
    encoder: EmbeddingEncoder,
    config: ClusteringConfig,
    *,
    clusterer_factory: Callable[..., Any] = HDBSCAN,
    text_builder: Callable[[FAQClusterDocument], str] | None = None,
) -> ClusterResult:
    """Cluster one category using normalized document embeddings and cosine distance."""
    document_list = list(documents)
    document_count = len(document_list)
    if document_count < config.min_cluster_size:
        return _skipped_result(document_count, config)

    embeddings = embed_documents(document_list, encoder, text_builder=text_builder)
    return cluster_embeddings(
        embeddings,
        config,
        clusterer_factory=clusterer_factory,
    )


def embed_documents(
    documents: Sequence[FAQClusterDocument],
    encoder: EmbeddingEncoder,
    *,
    text_builder: Callable[[FAQClusterDocument], str] | None = None,
) -> np.ndarray:
    document_list = list(documents)
    build_text = text_builder or _default_text_builder
    texts = [build_text(document) for document in document_list]
    embeddings = _normalize_rows(encoder.encode_documents(texts))
    if embeddings.shape[0] != len(document_list):
        raise ValueError("encoder returned a different number of embeddings than documents")
    return embeddings


def cluster_embeddings(
    embeddings: np.ndarray,
    config: ClusteringConfig,
    *,
    clusterer_factory: Callable[..., Any] = HDBSCAN,
) -> ClusterResult:
    normalized_embeddings = _normalize_rows(embeddings)
    document_count = len(normalized_embeddings)
    if document_count < config.min_cluster_size:
        return _skipped_result(document_count, config, embeddings=normalized_embeddings)

    cosine_distances = np.clip(
        1.0 - normalized_embeddings @ normalized_embeddings.T, 0.0, 2.0
    )
    np.fill_diagonal(cosine_distances, 0.0)
    clusterer = clusterer_factory(
        metric="precomputed",
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method=config.cluster_selection_method,
    )
    labels = np.asarray(clusterer.fit_predict(cosine_distances), dtype=int)
    if len(labels) != document_count:
        raise ValueError("clusterer returned a different number of labels than documents")

    diagnostics = {
        **summarize_clusters(labels),
        "mean_cluster_cohesion": _mean_cluster_cohesion(normalized_embeddings, labels),
    }
    return ClusterResult(
        status="completed",
        labels=labels,
        embeddings=normalized_embeddings,
        diagnostics=diagnostics,
        config=config,
    )


def _normalize_rows(embeddings: Any) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.size == 0:
        return np.empty((0, 0), dtype=float)
    if matrix.ndim != 2:
        raise ValueError("encoder must return a two-dimensional embedding matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def _default_text_builder(document: FAQClusterDocument) -> str:
    return document.text


def _skipped_result(
    document_count: int,
    config: ClusteringConfig,
    *,
    embeddings: np.ndarray | None = None,
) -> ClusterResult:
    labels = np.full(document_count, -1, dtype=int)
    return ClusterResult(
        status="skipped",
        labels=labels,
        embeddings=(
            embeddings
            if embeddings is not None
            else np.empty((document_count, 0), dtype=float)
        ),
        diagnostics=summarize_clusters(labels),
        config=config,
        reason="fewer than min_cluster_size documents",
    )


def _mean_cluster_cohesion(embeddings: np.ndarray, labels: np.ndarray) -> float:
    similarities = []
    for cluster_id in sorted(set(labels) - {-1}):
        cluster_embeddings = embeddings[labels == cluster_id]
        if len(cluster_embeddings) < 2:
            continue
        matrix = cluster_embeddings @ cluster_embeddings.T
        upper_indices = np.triu_indices(len(cluster_embeddings), k=1)
        similarities.extend(matrix[upper_indices].tolist())
    return float(np.mean(similarities)) if similarities else 0.0
