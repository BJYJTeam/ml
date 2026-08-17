import json
from unittest.mock import patch

import pandas as pd
from django.core.management import call_command


def test_generate_faq_drafts_command_writes_reviewable_versioned_run_without_live_models(tmp_path):
    output_path = tmp_path / "faq-drafts-v1.json"
    questions = pd.DataFrame(
        [
            {
                "id": "1",
                "tag": "[증상] 허리 통증",
                "title": "허리 통증",
                "content": "본문",
            }
        ]
    )
    generated_run = {
        "drafts": [
            {
                "content": "허리 FAQ",
                "answer": "검토용 답변",
                "category": "증상",
                "source_post_ids": ["1"],
                "status": "pending_review",
            }
        ],
        "errors": [],
        "clustering": {"증상": {"status": "completed", "diagnostics": {}}},
    }

    with patch(
        "ai_app.management.commands.generate_faq_drafts.load_qa_dataframe",
        return_value=questions,
    ), patch(
        "ai_app.management.commands.generate_faq_drafts.load_doctor_comments",
        return_value=[{"post_id": "1", "author": "doctor", "content": "의사 답변"}],
    ), patch(
        "ai_app.management.commands.generate_faq_drafts.SentenceTransformerEncoder"
    ), patch(
        "ai_app.management.commands.generate_faq_drafts.generate_faq_draft_run",
        return_value=generated_run,
    ):
        call_command("generate_faq_drafts", "--output", str(output_path))

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["status"] == "pending_review"
    assert record["drafts"] == generated_run["drafts"]
    assert record["clustering"] == generated_run["clustering"]
