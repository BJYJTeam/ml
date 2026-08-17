import pandas as pd
import pytest

from ai_app.evaluation import (
    DataIntegrityError,
    build_data_integrity_report,
    load_qa_dataframe,
)


def test_data_integrity_report_excludes_unusable_records_and_counts_answers():
    questions = pd.DataFrame(
        [
            {
                "id": "post-1",
                "tag": "[증상 및 진단 문의], 허리 통증",
                "title": "허리가 아파요",
                "content": "걷기 어려울 정도로 허리가 아픕니다.",
            },
            {
                "id": "post-2",
                "tag": "[치료 및 시술 문의], 도수치료",
                "title": "도수치료 문의",
                "content": "Content not found.",
            },
            {
                "id": "post-3",
                "tag": "[증상 및 진단 문의], 두통",
                "title": "두통 문의",
                "content": "",
            },
        ]
    )
    comments = [
        {"post_id": "post-1", "author": "DOCTOR", "content": "진료가 필요합니다."},
        {"post_id": "post-3", "author": "USER", "content": "추가 질문입니다."},
    ]

    report = build_data_integrity_report(questions, comments)

    assert report["total_records"] == 3
    assert report["usable_records"] == 2
    assert report["doctor_answered_records"] == 1
    assert report["doctor_answer_coverage"] == 0.5
    assert report["category_counts"] == {"증상 및 진단 문의": 2}


def test_data_integrity_report_requires_expected_columns():
    questions = pd.DataFrame([{"id": "post-1", "title": "제목", "content": "본문"}])

    with pytest.raises(DataIntegrityError, match="tag"):
        build_data_integrity_report(questions, [])


def test_load_qa_dataframe_filters_unusable_records_from_a_csv(tmp_path):
    data_path = tmp_path / "questions.csv"
    pd.DataFrame(
        [
            {
                "id": "post-1",
                "tag": "[증상 및 진단 문의], 허리 통증",
                "title": "허리 통증",
                "content": "통증이 있습니다.",
            },
            {
                "id": "post-2",
                "tag": "[증상 및 진단 문의], 두통",
                "title": "두통",
                "content": "content not found",
            },
        ]
    ).to_csv(data_path, index=False)

    loaded = load_qa_dataframe(data_path)

    assert loaded["id"].tolist() == ["post-1"]
