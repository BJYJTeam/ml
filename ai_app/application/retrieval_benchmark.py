from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Sequence

from ai_app.evaluation import calculate_retrieval_metrics
from ai_app.retrieval import EmbeddingEncoder, QuestionDocument, SemanticRetriever


@dataclass(frozen=True)
class EncoderCandidate:
    name: str
    model_name: str
    query_prefix: str = ""
    document_prefix: str = ""


DEFAULT_ENCODER_CANDIDATES = (
    EncoderCandidate("all-minilm-l6-v2", "all-MiniLM-L6-v2"),
    EncoderCandidate("kr-sbert-v40k", "snunlp/KR-SBERT-V40K-klueNLI-augSTS"),
    EncoderCandidate(
        "multilingual-e5-small",
        "intfloat/multilingual-e5-small",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
)
DEFAULT_THRESHOLDS = tuple(round(value, 2) for value in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))


def evaluate_retriever(
    retriever: SemanticRetriever,
    queries: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    baseline_rankings = _rank_queries(retriever, queries, min_score=None)
    metrics = calculate_retrieval_metrics(baseline_rankings, queries)
    threshold_metrics: Dict[str, Dict[str, float]] = {}

    for threshold in thresholds:
        rankings = _rank_queries(retriever, queries, min_score=threshold)
        threshold_retrieval_metrics = calculate_retrieval_metrics(rankings, queries)
        negative_queries = [query for query in queries if not query["relevant_post_ids"]]
        specificity = (
            sum(not rankings[query["query_id"]] for query in negative_queries)
            / len(negative_queries)
            if negative_queries
            else 1.0
        )
        balanced_score = (threshold_retrieval_metrics["top_3_recall"] + specificity) / 2
        threshold_metrics[str(threshold)] = {
            "top_1_recall": threshold_retrieval_metrics["top_1_recall"],
            "top_3_recall": threshold_retrieval_metrics["top_3_recall"],
            "no_match_specificity": specificity,
            "balanced_score": balanced_score,
        }

    selected_threshold = max(
        thresholds,
        key=lambda threshold: (
            threshold_metrics[str(threshold)]["balanced_score"],
            threshold_metrics[str(threshold)]["top_3_recall"],
            threshold_metrics[str(threshold)]["no_match_specificity"],
            threshold,
        ),
    )
    return {
        "metrics": metrics,
        "selected_threshold": selected_threshold,
        "threshold_metrics": threshold_metrics,
    }


def benchmark_candidates(
    documents: Sequence[QuestionDocument],
    queries: Sequence[Mapping[str, Any]],
    candidates: Sequence[EncoderCandidate],
    encoder_factory: Callable[[EncoderCandidate], EmbeddingEncoder],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> List[Dict[str, Any]]:
    reports = []
    for candidate in candidates:
        started_at = perf_counter()
        encoder = encoder_factory(candidate)
        retriever = SemanticRetriever(documents, encoder)
        report = evaluate_retriever(retriever, queries, thresholds=thresholds)
        reports.append(
            {
                "name": candidate.name,
                "model_name": candidate.model_name,
                "query_prefix": candidate.query_prefix,
                "document_prefix": candidate.document_prefix,
                "initialization_and_index_seconds": perf_counter() - started_at,
                **report,
            }
        )
    return reports


def select_best_candidate(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not reports:
        raise ValueError("At least one encoder benchmark report is required")

    return max(
        reports,
        key=lambda report: (
            report["threshold_metrics"][str(report["selected_threshold"])]["balanced_score"],
            report["metrics"]["top_3_recall"],
            -report["initialization_and_index_seconds"],
        ),
    )


def _rank_queries(
    retriever: SemanticRetriever,
    queries: Sequence[Mapping[str, Any]],
    *,
    min_score: float | None,
) -> Dict[str, List[str]]:
    rankings: Dict[str, List[str]] = {}
    for query in queries:
        results = retriever.search(
            query["title"],
            query["content"],
            exclude_post_id=query["exclude_post_id"],
            top_k=3,
            min_score=min_score,
        )
        rankings[query["query_id"]] = [result.post_id for result in results]
    return rankings
