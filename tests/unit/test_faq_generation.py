import numpy as np
import pandas as pd

from ai_app.application.faq_generation import generate_faq_draft_run
from ai_app.clustering import ClusteringConfig
from ai_app.llm import LLMProviderError


class FixedClusterer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_predict(self, distances):
        return np.zeros(len(distances), dtype=int)


class FakeEncoder:
    def encode_documents(self, texts):
        return np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])

    def encode_queries(self, texts):
        raise AssertionError("FAQ generation only encodes clustered documents")


def questions():
    return pd.DataFrame(
        [
            {"id": "1", "tag": "[증상] 허리 통증", "title": "허리 통증", "content": "본문 1"},
            {"id": "2", "tag": "[증상] 허리 통증", "title": "허리 치료", "content": "본문 2"},
            {"id": "3", "tag": "[증상] 허리 통증", "title": "허리 저림", "content": "본문 3"},
        ]
    )


def test_generate_faq_draft_run_uses_multiple_aligned_pairs_and_marks_draft_pending_review():
    prompts = []
    run = generate_faq_draft_run(
        questions(),
        doctor_answers_by_post_id={"1": ["의사 답변 1"], "3": ["의사 답변 3"]},
        encoder=FakeEncoder(),
        llm_generate=lambda prompt: prompts.append(prompt) or "Q: 허리 FAQ\nA: 검토용 답변",
        config=ClusteringConfig(min_cluster_size=2, min_samples=1),
        clusterer_factory=FixedClusterer,
    )

    assert run["drafts"] == [
        {
            "content": "허리 FAQ",
            "answer": "검토용 답변",
            "category": "증상",
            "source_post_ids": ["1", "3"],
            "status": "pending_review",
        }
    ]
    assert "의사 답변 1" in prompts[0]
    assert "의사 답변 3" in prompts[0]


def test_generate_faq_draft_run_keeps_malformed_llm_output_out_of_stored_drafts():
    run = generate_faq_draft_run(
        questions(),
        doctor_answers_by_post_id={"1": ["의사 답변 1"]},
        encoder=FakeEncoder(),
        llm_generate=lambda prompt: "출력 형식 오류",
        config=ClusteringConfig(min_cluster_size=2, min_samples=1),
        clusterer_factory=FixedClusterer,
    )

    assert run["drafts"] == []
    assert run["errors"] == [
        {"category": "증상", "cluster_id": 0, "reason": "FAQ output must contain exactly one Q: line and one A: line"}
    ]


def test_generate_faq_draft_run_records_llm_failure_without_publishing_a_draft():
    run = generate_faq_draft_run(
        questions(),
        doctor_answers_by_post_id={"1": ["의사 답변 1"]},
        encoder=FakeEncoder(),
        llm_generate=lambda prompt: (_ for _ in ()).throw(LLMProviderError("unavailable")),
        config=ClusteringConfig(min_cluster_size=2, min_samples=1),
        clusterer_factory=FixedClusterer,
    )

    assert run["drafts"] == []
    assert run["errors"] == [{"category": "증상", "cluster_id": 0, "reason": "unavailable"}]
