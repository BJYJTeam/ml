import inspect

import numpy as np

from ai_app.clustering import (
    ClusteringConfig,
    FAQClusterDocument,
    build_clustering_text,
    cluster_documents,
)


class FixedLabelsClusterer:
    labels = np.array([], dtype=int)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_predict(self, distances):
        assert distances.shape == (len(self.labels), len(self.labels))
        return self.labels


class FixedEncoder:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=float)
        self.received_texts = []

    def encode_documents(self, texts):
        self.received_texts = list(texts)
        return self.vectors

    def encode_queries(self, texts):
        raise AssertionError("Clustering must use document embeddings")


def documents(count):
    return [
        FAQClusterDocument(
            post_id=str(index),
            title=f"질문 {index}",
            keywords="허리 통증",
            content="본문은 클러스터링 표현에 직접 포함되지 않아야 합니다.",
        )
        for index in range(count)
    ]


def test_clustering_text_uses_title_and_keywords_without_raw_body():
    text = build_clustering_text(
        title="걸을 때 허리가 아파요",
        keywords="허리 통증, 보행 통증",
        content="이 문장은 긴 본문이며 클러스터링 입력에 포함되면 안 됩니다.",
    )

    assert text == "걸을 때 허리가 아파요\n핵심 키워드: 허리 통증 보행 통증"
    assert "긴 본문" not in text


def test_cluster_documents_has_no_target_cluster_count_parameter():
    parameter_names = inspect.signature(cluster_documents).parameters

    assert "target_clusters" not in parameter_names
    assert "target_clusters_per_category" not in parameter_names


def test_cluster_documents_reports_all_noise_diagnostics():
    FixedLabelsClusterer.labels = np.array([-1, -1, -1])
    encoder = FixedEncoder([[1, 0], [0, 1], [1, 1]])

    result = cluster_documents(
        documents(3),
        encoder,
        ClusteringConfig(min_cluster_size=2, min_samples=1),
        clusterer_factory=FixedLabelsClusterer,
    )

    assert result.status == "completed"
    assert result.diagnostics["n_clusters"] == 0
    assert result.diagnostics["noise_ratio"] == 1.0
    assert result.diagnostics["largest_cluster_ratio"] == 0.0


def test_cluster_documents_reports_single_cluster_and_normalized_embeddings():
    FixedLabelsClusterer.labels = np.array([0, 0, 0])
    encoder = FixedEncoder([[3, 4], [6, 8], [0, 5]])

    result = cluster_documents(
        documents(3),
        encoder,
        ClusteringConfig(
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_method="leaf",
        ),
        clusterer_factory=FixedLabelsClusterer,
    )

    assert result.status == "completed"
    assert result.diagnostics["n_clusters"] == 1
    assert result.diagnostics["largest_cluster_ratio"] == 1.0
    assert np.allclose(np.linalg.norm(result.embeddings, axis=1), [1.0, 1.0, 1.0])
    assert encoder.received_texts == [document.text for document in documents(3)]


def test_cluster_documents_skips_category_smaller_than_minimum_cluster_size():
    encoder = FixedEncoder([[1, 0], [0, 1]])

    result = cluster_documents(
        documents(2),
        encoder,
        ClusteringConfig(min_cluster_size=3, min_samples=1),
        clusterer_factory=FixedLabelsClusterer,
    )

    assert result.status == "skipped"
    assert result.reason == "fewer than min_cluster_size documents"
    assert result.diagnostics["n_samples"] == 2
    assert result.diagnostics["noise_ratio"] == 1.0
