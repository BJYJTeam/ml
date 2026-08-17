from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from django.core.management.base import BaseCommand

from ai_app.application.faq_generation import generate_faq_draft_run
from ai_app.evaluation import load_doctor_comments, load_qa_dataframe, write_report
from ai_app.faq import DEFAULT_FAQ_CLUSTERING_CONFIG
from ai_app.infrastructure.embeddings import SentenceTransformerEncoder
from ai_app.llm import GEMMA_MODEL_NAME, generate_gemma_answer


FAQ_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


class Command(BaseCommand):
    help = "Generate reviewable FAQ drafts from clustered historical Q/A pairs."

    def add_arguments(self, parser):
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--representatives-per-cluster", default=3, type=int)

    def handle(self, *args, **options):
        representatives_per_cluster = options["representatives_per_cluster"]
        if representatives_per_cluster < 1:
            raise ValueError("representatives-per-cluster must be at least 1")

        questions = load_qa_dataframe()
        doctor_answers_by_post_id = _doctor_answers_by_post_id(load_doctor_comments())
        encoder = SentenceTransformerEncoder(
            FAQ_EMBEDDING_MODEL_NAME,
            document_prefix="passage: ",
        )
        generated_run = generate_faq_draft_run(
            questions,
            doctor_answers_by_post_id,
            encoder,
            generate_gemma_answer,
            DEFAULT_FAQ_CLUSTERING_CONFIG,
            representatives_per_cluster=representatives_per_cluster,
        )
        record = {
            "run_id": options["output"].stem,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_review",
            "embedding_model": {
                "name": FAQ_EMBEDDING_MODEL_NAME,
                "document_prefix": "passage: ",
            },
            "llm_model": GEMMA_MODEL_NAME,
            "clustering_config": {
                "min_cluster_size": DEFAULT_FAQ_CLUSTERING_CONFIG.min_cluster_size,
                "min_samples": DEFAULT_FAQ_CLUSTERING_CONFIG.min_samples,
                "cluster_selection_method": DEFAULT_FAQ_CLUSTERING_CONFIG.cluster_selection_method,
            },
            "representatives_per_cluster": representatives_per_cluster,
            **generated_run,
        }
        write_report(record, options["output"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Stored {len(generated_run['drafts'])} pending-review FAQ drafts at {options['output']}"
            )
        )


def _doctor_answers_by_post_id(comments: List[Dict[str, object]]) -> Dict[str, List[str]]:
    answers: Dict[str, List[str]] = {}
    for comment in comments:
        if str(comment.get("author", "")).lower() != "doctor":
            continue
        answer = str(comment.get("content", "")).strip()
        if answer:
            answers.setdefault(str(comment.get("post_id")), []).append(answer)
    return answers
