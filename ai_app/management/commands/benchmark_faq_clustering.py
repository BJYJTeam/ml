from datetime import datetime, timezone
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ai_app.application.clustering_benchmark import (
    ClusteringCandidate,
    run_clustering_benchmark,
    select_best_cluster_candidate,
)
from ai_app.clustering import ClusteringConfig
from ai_app.evaluation import (
    DEFAULT_QA_DATA_PATH,
    filter_usable_questions,
    load_raw_qa_dataframe,
    write_report,
)
from ai_app.infrastructure.embeddings import SentenceTransformerEncoder


def _build_default_candidates() -> tuple[ClusteringCandidate, ...]:
    candidates = [
        ClusteringCandidate(
            name="legacy-all-minilm-keyword-content-eom-mcs4-ms2",
            model_name="all-MiniLM-L6-v2",
            representation="legacy_keyword_content",
            config=ClusteringConfig(
                min_cluster_size=4,
                min_samples=2,
                cluster_selection_method="eom",
            ),
        )
    ]
    model_options = (
        ("all-minilm", "all-MiniLM-L6-v2", ""),
        ("kr-sbert-v40k", "snunlp/KR-SBERT-V40K-klueNLI-augSTS", ""),
        ("multilingual-e5-small", "intfloat/multilingual-e5-small", "passage: "),
    )
    for model_label, model_name, document_prefix in model_options:
        for representation in ("title_only", "title_keywords"):
            for min_cluster_size in (3, 4, 5):
                for min_samples in (1, 2, 3):
                    for selection_method in ("eom", "leaf"):
                        candidates.append(
                            ClusteringCandidate(
                                name=(
                                    f"{model_label}-{representation}-{selection_method}"
                                    f"-mcs{min_cluster_size}-ms{min_samples}"
                                ),
                                model_name=model_name,
                                representation=representation,
                                document_prefix=document_prefix,
                                config=ClusteringConfig(
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_method=selection_method,
                                ),
                            )
                        )
    return tuple(candidates)


DEFAULT_CLUSTERING_CANDIDATES = _build_default_candidates()
SELECTED_PRODUCTION_CANDIDATE = "multilingual-e5-small-title_keywords-leaf-mcs3-ms2"


class Command(BaseCommand):
    help = "Benchmark FAQ clustering representations and HDBSCAN configurations."

    def add_arguments(self, parser):
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument(
            "--candidate",
            action="append",
            choices=[candidate.name for candidate in DEFAULT_CLUSTERING_CANDIDATES],
        )

    def handle(self, *args, **options):
        requested_candidates = options["candidate"]
        candidates = tuple(
            candidate
            for candidate in DEFAULT_CLUSTERING_CANDIDATES
            if not requested_candidates or candidate.name in requested_candidates
        )
        if not candidates:
            raise CommandError("No clustering candidates were selected")

        questions = filter_usable_questions(load_raw_qa_dataframe(DEFAULT_QA_DATA_PATH))
        encoders = {}

        def encoder_factory(candidate):
            cache_key = (candidate.model_name, candidate.document_prefix)
            if cache_key not in encoders:
                encoders[cache_key] = SentenceTransformerEncoder(
                    candidate.model_name,
                    document_prefix=candidate.document_prefix,
                )
            return encoders[cache_key]

        reports = run_clustering_benchmark(questions, candidates, encoder_factory)
        automated_best = select_best_cluster_candidate(reports)
        reports_by_name = {report["name"]: report for report in reports}
        selected = reports_by_name.get(SELECTED_PRODUCTION_CANDIDATE, automated_best)
        report = {
            "benchmark_id": "faq-clustering-benchmark-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_path": str(DEFAULT_QA_DATA_PATH),
            "n_usable_documents": len(questions),
            "random_seed": None,
            "umap": {
                "used": False,
                "reason": "The direct-embedding baseline was evaluated before introducing dimensionality reduction.",
            },
            "selection_criterion": (
                "Maximize the sample-weighted mean of category scores. A category score is "
                "0.3 * mean cluster cohesion + 0.3 * mean keyword Jaccard purity + "
                "0.2 * non-noise coverage + 0.2 * non-dominance (1 - largest cluster "
                "ratio). Cluster count is not used as a score."
            ),
            "selection_decision": {
                "automated_best_candidate": automated_best["name"],
                "production_candidate": selected["name"],
                "reason": (
                    "The production candidate is restricted to the title_keywords "
                    "representation. The title_only candidate with the highest automatic "
                    "score was retained only as a diagnostic comparison because it omits "
                    "the extracted topical context. Within the selected representation, "
                    "leaf was chosen to avoid the oversized EOM cluster; min_samples=2 "
                    "was selected because it produced the same result as 1 while requiring "
                    "at least two neighboring points."
                ),
            },
            "review": {
                "status": "repository-content-review",
                "scope": "semantic topic coherence only; not clinical validation",
                "findings": [
                    "The legacy symptom cluster is overly dominant and unsuitable for FAQ drafting.",
                    "The title_only comparison omits extracted topical context and is not eligible for production.",
                    "The selected leaf configuration retains dense subtopics and leaves sparse treatment questions as noise.",
                ],
            },
            "reports": reports,
            "selected": selected,
        }
        write_report(report, options["output"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Selected {selected['name']} with score {selected['selection_score']:.4f}"
            )
        )
