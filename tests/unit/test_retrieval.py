import numpy as np
import pandas as pd

from ai_app.evaluation import calculate_retrieval_metrics, resolve_validation_queries
from ai_app.retrieval import QuestionDocument, SemanticRetriever, build_question_text


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode_documents(self, texts):
        return np.array([self.vectors[text] for text in texts], dtype=float)

    def encode_queries(self, texts):
        return np.array([self.vectors[text] for text in texts], dtype=float)


def test_build_question_text_uses_the_same_canonical_format_for_corpus_and_query():
    assert build_question_text(" 허리 통증 ", " 걸을 때 아파요 ") == "허리 통증\n걸을 때 아파요"
    assert build_question_text("제목만", "") == "제목만"
    assert build_question_text("", "본문만") == "본문만"


def test_semantic_retriever_normalizes_vectors_excludes_self_and_sorts_by_score():
    documents = [
        QuestionDocument("same", "허리 통증", "걸을 때 아파요"),
        QuestionDocument("relevant", "허리 통증", "걷기 힘들어요"),
        QuestionDocument("unrelated", "두통", "머리가 아파요"),
    ]
    encoder = FakeEncoder(
        {
            build_question_text("허리 통증", "걸을 때 아파요"): [4.0, 0.0],
            build_question_text("허리 통증", "걷기 힘들어요"): [2.0, 0.0],
            build_question_text("두통", "머리가 아파요"): [0.0, 3.0],
        }
    )
    retriever = SemanticRetriever(documents, encoder)

    results = retriever.search(
        "허리 통증", "걸을 때 아파요", exclude_post_id="same", top_k=3
    )

    assert [result.post_id for result in results] == ["relevant", "unrelated"]
    assert results[0].score == 1.0
    assert results[1].score == 0.0


def test_semantic_retriever_applies_a_threshold_and_handles_a_short_corpus():
    documents = [QuestionDocument("post-1", "허리", "통증")]
    text = build_question_text("허리", "통증")
    retriever = SemanticRetriever(documents, FakeEncoder({text: [1.0, 0.0]}))

    assert retriever.search("허리", "통증", top_k=3, min_score=0.9)[0].post_id == "post-1"
    assert retriever.search("허리", "통증", top_k=3, min_score=1.01) == []
    assert retriever.search("", "", top_k=3) == []


def test_retrieval_metrics_report_top_1_and_top_3_relevance():
    rankings = {
        "query-1": ["post-2", "post-1"],
        "query-2": ["post-3", "post-4", "post-5"],
    }
    judgments = [
        {"query_id": "query-1", "relevant_post_ids": ["post-1"]},
        {"query_id": "query-2", "relevant_post_ids": ["post-3"]},
    ]

    assert calculate_retrieval_metrics(rankings, judgments) == {
        "n_queries": 2,
        "n_queries_with_relevance": 2,
        "top_1_recall": 0.5,
        "top_3_recall": 1.0,
    }


def test_validation_queries_can_reference_a_historical_source_or_embed_text():
    questions = pd.DataFrame(
        [
            {
                "id": "source-post",
                "tag": "[증상 및 진단 문의], 허리 통증",
                "title": "허리 통증",
                "content": "걷기 힘듭니다.",
            }
        ]
    )
    judgments = [
        {
            "query_id": "from-source",
            "source_post_id": "source-post",
            "relevant_post_ids": ["related-post"],
        },
        {
            "query_id": "inline",
            "title": "두통",
            "content": "머리가 아픕니다.",
            "relevant_post_ids": [],
        },
    ]

    resolved = resolve_validation_queries(judgments, questions)

    assert resolved[0]["title"] == "허리 통증"
    assert resolved[0]["exclude_post_id"] == "source-post"
    assert resolved[1]["content"] == "머리가 아픕니다."
