import numpy as np
import pandas as pd
import pytest

from ai_app.application.clustering_benchmark import (
    ClusteringCandidate,
    run_clustering_benchmark,
    select_best_cluster_candidate,
)
from ai_app.clustering import ClusteringConfig


class FixedClusterer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_predict(self, distances):
        return np.zeros(len(distances), dtype=int)


class FakeEncoder:
    def __init__(self):
        self.texts = []

    def encode_documents(self, texts):
        self.texts = list(texts)
        return np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])

    def encode_queries(self, texts):
        raise AssertionError("clustering does not encode queries")


def questions():
    return pd.DataFrame(
        [
            {"id": "1", "tag": "[증상] 허리 통증", "title": "허리가 아파요", "content": "긴 본문 1"},
            {"id": "2", "tag": "[증상] 허리 통증", "title": "허리 통증 치료", "content": "긴 본문 2"},
            {"id": "3", "tag": "[증상] 허리 통증", "title": "허리 통증 문의", "content": "긴 본문 3"},
        ]
    )


def test_run_clustering_benchmark_records_each_representation_and_samples():
    encoders = []
    candidates = [
        ClusteringCandidate(
            name=representation,
            model_name="fake",
            representation=representation,
            config=ClusteringConfig(min_cluster_size=2, min_samples=1),
        )
        for representation in ("title_keywords", "title_only", "legacy_keyword_content")
    ]

    reports = run_clustering_benchmark(
        questions(),
        candidates,
        lambda candidate: encoders.append(FakeEncoder()) or encoders[-1],
        clusterer_factory=FixedClusterer,
    )

    assert [report["name"] for report in reports] == [
        "title_keywords",
        "title_only",
        "legacy_keyword_content",
    ]
    assert reports[0]["categories"]["증상"]["sampled_questions"]["0"][0]["post_id"] == "1"
    assert "긴 본문" not in encoders[0].texts[0]
    assert encoders[1].texts[0] == "허리가 아파요"
    assert "긴 본문 1" in encoders[2].texts[0]


def test_run_clustering_benchmark_rejects_unknown_representation():
    candidate = ClusteringCandidate(
        name="unknown",
        model_name="fake",
        representation="unknown",
        config=ClusteringConfig(min_cluster_size=2, min_samples=1),
    )

    with pytest.raises(ValueError, match="unknown clustering representation"):
        run_clustering_benchmark(
            questions(),
            [candidate],
            lambda candidate: FakeEncoder(),
            clusterer_factory=FixedClusterer,
        )


def test_select_best_cluster_candidate_rejects_an_empty_report_set():
    with pytest.raises(ValueError, match="at least one"):
        select_best_cluster_candidate([])
