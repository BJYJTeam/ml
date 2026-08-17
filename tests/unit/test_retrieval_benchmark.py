import pytest

from ai_app.application.retrieval_benchmark import (
    EncoderCandidate,
    benchmark_candidates,
    evaluate_retriever,
    select_best_candidate,
)
from ai_app.retrieval import QuestionDocument, SearchResult


class FakeRetriever:
    def search(self, title, content, *, exclude_post_id, top_k, min_score):
        results_by_title = {
            "관련 질문": [SearchResult("relevant", 0.9)],
            "관련 없음": [SearchResult("unrelated", 0.8)],
        }
        return [
            result
            for result in results_by_title[title]
            if min_score is None or result.score >= min_score
        ][:top_k]


def test_evaluate_retriever_selects_a_threshold_that_keeps_relevant_results_and_rejects_noise():
    queries = [
        {
            "query_id": "relevant-query",
            "title": "관련 질문",
            "content": "본문",
            "exclude_post_id": None,
            "relevant_post_ids": ["relevant"],
        },
        {
            "query_id": "unrelated-query",
            "title": "관련 없음",
            "content": "본문",
            "exclude_post_id": None,
            "relevant_post_ids": [],
        },
    ]

    report = evaluate_retriever(FakeRetriever(), queries, thresholds=[0.0, 0.85])

    assert report["metrics"] == {
        "n_queries": 2,
        "n_queries_with_relevance": 1,
        "top_1_recall": 1.0,
        "top_3_recall": 1.0,
    }
    assert report["selected_threshold"] == 0.85
    assert report["threshold_metrics"]["0.85"]["balanced_score"] == 1.0


class BenchmarkEncoder:
    def encode_documents(self, texts):
        return [[1.0, 0.0], [0.0, 1.0]]

    def encode_queries(self, texts):
        return [[1.0, 0.0]]


def test_benchmark_candidates_records_initialization_and_selects_the_best_report():
    candidate = EncoderCandidate("fake", "fake-model")
    reports = benchmark_candidates(
        [QuestionDocument("relevant", "관련 질문", "본문"), QuestionDocument("other", "다른 질문", "본문")],
        [
            {
                "query_id": "query",
                "title": "관련 질문",
                "content": "본문",
                "exclude_post_id": None,
                "relevant_post_ids": ["relevant"],
            }
        ],
        [candidate],
        lambda _: BenchmarkEncoder(),
        thresholds=[0.0],
    )

    assert reports[0]["name"] == "fake"
    assert reports[0]["initialization_and_index_seconds"] >= 0.0
    assert select_best_candidate(reports) == reports[0]


def test_select_best_candidate_rejects_an_empty_report_set():
    with pytest.raises(ValueError, match="At least one"):
        select_best_candidate([])
