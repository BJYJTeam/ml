from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ai_app.application.retrieval_benchmark import (
    DEFAULT_ENCODER_CANDIDATES,
    benchmark_candidates,
    select_best_candidate,
)
from ai_app.evaluation import (
    DEFAULT_QA_DATA_PATH,
    DEFAULT_RETRIEVAL_VALIDATION_PATH,
    filter_usable_questions,
    load_raw_qa_dataframe,
    load_validation_judgments,
    resolve_validation_queries,
    write_report,
)
from ai_app.infrastructure.embeddings import SentenceTransformerEncoder
from ai_app.retrieval import QuestionDocument


class Command(BaseCommand):
    help = "Benchmark configured retrieval encoders against the reviewed relevance dataset."

    def add_arguments(self, parser):
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument(
            "--validation-path", type=Path, default=DEFAULT_RETRIEVAL_VALIDATION_PATH
        )
        parser.add_argument(
            "--candidate",
            action="append",
            choices=[candidate.name for candidate in DEFAULT_ENCODER_CANDIDATES],
        )

    def handle(self, *args, **options):
        requested_candidates = options["candidate"]
        candidates = tuple(
            candidate
            for candidate in DEFAULT_ENCODER_CANDIDATES
            if not requested_candidates or candidate.name in requested_candidates
        )
        if not candidates:
            raise CommandError("No encoder candidates were selected")

        questions = load_raw_qa_dataframe(DEFAULT_QA_DATA_PATH)
        usable_questions = filter_usable_questions(questions)
        documents = [
            QuestionDocument(str(row.id), row.title, row.content)
            for row in usable_questions.itertuples(index=False)
        ]
        judgments = load_validation_judgments(options["validation_path"])
        queries = resolve_validation_queries(judgments, usable_questions)

        reports = benchmark_candidates(
            documents,
            queries,
            candidates,
            lambda candidate: SentenceTransformerEncoder(
                candidate.model_name,
                query_prefix=candidate.query_prefix,
                document_prefix=candidate.document_prefix,
            ),
        )
        selected = select_best_candidate(reports)
        report = {
            "validation_path": str(options["validation_path"]),
            "n_corpus_documents": len(documents),
            "n_validation_queries": len(queries),
            "reports": reports,
            "selected": selected,
        }
        write_report(report, options["output"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Selected {selected['name']} with threshold {selected['selected_threshold']}"
            )
        )
