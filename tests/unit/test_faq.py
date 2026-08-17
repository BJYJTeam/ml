import json

import numpy as np
import pytest

from ai_app.clustering import FAQClusterDocument
from ai_app.faq import (
    FAQDraftError,
    filter_duplicate_drafts,
    load_approved_faqs,
    parse_faq_draft,
    select_representative_qa_pairs,
)


def test_representative_pairs_keep_each_question_aligned_with_its_own_doctor_answer():
    documents = [
        FAQClusterDocument("1", "허리 통증", "허리 통증", "본문 1"),
        FAQClusterDocument("2", "허리 치료", "허리 치료", "본문 2"),
        FAQClusterDocument("3", "허리 저림", "다리 저림", "본문 3"),
    ]
    pairs_by_cluster = select_representative_qa_pairs(
        documents,
        labels=np.array([0, 0, 0]),
        embeddings=np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        doctor_answers_by_post_id={"1": ["답변 1"], "3": ["답변 3"]},
        top_n=3,
    )

    pairs = pairs_by_cluster[0]
    assert [(pair.post_id, pair.answer) for pair in pairs] == [
        ("1", "답변 1"),
        ("3", "답변 3"),
    ]
    assert all(pair.answer != "(의사 답변 없음)" for pair in pairs)


@pytest.mark.parametrize("output", ["", "Q: 질문만 있습니다", "A: 답변만 있습니다"])
def test_parse_faq_draft_rejects_empty_or_malformed_llm_output(output):
    with pytest.raises(FAQDraftError):
        parse_faq_draft(output, category="증상", source_post_ids=["1"])


def test_parse_faq_draft_marks_a_valid_result_pending_review():
    draft = parse_faq_draft(
        "Q: 허리 통증은 언제 진료를 받아야 하나요?\nA: 증상이 지속되면 진료가 필요합니다.",
        category="증상",
        source_post_ids=["1", "3"],
    )

    assert draft["status"] == "pending_review"
    assert draft["source_post_ids"] == ["1", "3"]
    assert draft["content"] == "허리 통증은 언제 진료를 받아야 하나요?"


def test_filter_duplicate_drafts_keeps_the_first_normalized_question():
    drafts = [
        {"content": "허리 통증은 치료가 가능한가요?", "answer": "답변 1"},
        {"content": "허리 통증은 치료가 가능한가요? ", "answer": "답변 2"},
    ]

    assert filter_duplicate_drafts(drafts) == [drafts[0]]


def test_load_approved_faqs_reads_only_approved_drafts(tmp_path):
    path = tmp_path / "faq-drafts.json"
    path.write_text(
        json.dumps(
            {
                "drafts": [
                    {"content": "승인 질문", "answer": "승인 답변", "status": "approved"},
                    {"content": "대기 질문", "answer": "대기 답변", "status": "pending_review"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert load_approved_faqs(path) == [{"content": "승인 질문", "answer": "승인 답변"}]
