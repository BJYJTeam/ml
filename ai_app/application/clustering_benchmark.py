from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import pandas as pd
from sklearn.cluster import HDBSCAN

from ai_app.clustering import (
    ClusteringConfig,
    FAQClusterDocument,
    build_clustering_text,
    cluster_embeddings,
    embed_documents,
)
from ai_app.retrieval import EmbeddingEncoder


@dataclass(frozen=True)
class ClusteringCandidate:
    name: str
    model_name: str
    representation: str
    config: ClusteringConfig
    document_prefix: str = ""


def score_cluster_diagnostics(diagnostics: Mapping[str, Any]) -> float:
    """Score density quality without using cluster count as an objective."""
    if not diagnostics.get("n_samples") or diagnostics.get("n_clusters", 0) == 0:
        return 0.0

    cohesion = max(0.0, min(1.0, float(diagnostics["mean_cluster_cohesion"])))
    keyword_purity = max(
        0.0, min(1.0, float(diagnostics.get("mean_keyword_jaccard", 0.0)))
    )
    coverage = 1.0 - max(0.0, min(1.0, float(diagnostics["noise_ratio"])))
    non_dominance = 1.0 - max(
        0.0, min(1.0, float(diagnostics["largest_cluster_ratio"]))
    )
    return (
        0.3 * cohesion
        + 0.3 * keyword_purity
        + 0.2 * coverage
        + 0.2 * non_dominance
    )


def select_best_cluster_candidate(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not reports:
        raise ValueError("at least one clustering candidate report is required")
    return max(reports, key=lambda report: float(report["selection_score"]))


def run_clustering_benchmark(
    questions: pd.DataFrame,
    candidates: Sequence[ClusteringCandidate],
    encoder_factory: Callable[[ClusteringCandidate], EmbeddingEncoder],
    *,
    clusterer_factory: Callable[..., Any] = HDBSCAN,
) -> list[Dict[str, Any]]:
    categorized_documents = _documents_by_category(questions)
    reports: list[Dict[str, Any]] = []
    embedding_cache: Dict[tuple[str, str, str, str], Any] = {}

    for candidate in candidates:
        encoder = encoder_factory(candidate)
        category_reports: Dict[str, Dict[str, Any]] = {}
        weighted_score = 0.0
        weighted_samples = 0
        text_builder = _text_builder_for(candidate.representation)

        for category, documents in categorized_documents.items():
            cache_key = (
                candidate.model_name,
                candidate.document_prefix,
                candidate.representation,
                category,
            )
            embeddings = embedding_cache.get(cache_key)
            if embeddings is None:
                embeddings = embed_documents(documents, encoder, text_builder=text_builder)
                embedding_cache[cache_key] = embeddings
            result = cluster_embeddings(
                embeddings,
                candidate.config,
                clusterer_factory=clusterer_factory,
            )
            diagnostics = {
                **result.diagnostics,
                "mean_keyword_jaccard": _mean_keyword_jaccard(documents, result.labels),
            }
            category_score = score_cluster_diagnostics(diagnostics)
            document_count = len(documents)
            if result.status == "completed":
                weighted_score += category_score * document_count
                weighted_samples += document_count

            category_reports[category] = {
                "status": result.status,
                "reason": result.reason,
                "diagnostics": diagnostics,
                "sampled_questions": _sample_questions(documents, result.labels),
                "selection_score": category_score,
            }

        reports.append(
            {
                "name": candidate.name,
                "model_name": candidate.model_name,
                "document_prefix": candidate.document_prefix,
                "representation": candidate.representation,
                "config": {
                    "min_cluster_size": candidate.config.min_cluster_size,
                    "min_samples": candidate.config.min_samples,
                    "cluster_selection_method": candidate.config.cluster_selection_method,
                },
                "categories": category_reports,
                "selection_score": (
                    weighted_score / weighted_samples if weighted_samples else 0.0
                ),
            }
        )
    return reports


def _documents_by_category(questions: pd.DataFrame) -> Dict[str, list[FAQClusterDocument]]:
    working_questions = questions.copy()
    working_questions["category"] = working_questions["tag"].str.extract(r"\[(.*?)\]")
    working_questions["keywords"] = working_questions["tag"].str.replace(
        r"\[.*?\]", "", regex=True
    ).str.strip()

    documents_by_category: Dict[str, list[FAQClusterDocument]] = {}
    for category, category_questions in working_questions.groupby("category", dropna=False):
        documents_by_category[str(category)] = [
            FAQClusterDocument(
                post_id=str(row.id),
                title=str(row.title or ""),
                keywords=str(row.keywords or ""),
                content=str(row.content or ""),
            )
            for row in category_questions.itertuples(index=False)
        ]
    return documents_by_category


def _text_builder_for(representation: str) -> Callable[[FAQClusterDocument], str]:
    if representation == "title_keywords":
        return lambda document: build_clustering_text(
            document.title, document.keywords, document.content
        )
    if representation == "title_only":
        return lambda document: document.title.strip()
    if representation == "legacy_keyword_content":
        return lambda document: " ".join(
            part for part in (document.keywords.strip(), document.content.strip()) if part
        )
    raise ValueError(f"unknown clustering representation: {representation}")


def _sample_questions(
    documents: Sequence[FAQClusterDocument], labels: Sequence[int], sample_size: int = 3
) -> Dict[str, list[Dict[str, str]]]:
    samples: Dict[str, list[Dict[str, str]]] = {}
    for document, label in zip(documents, labels):
        if label == -1:
            continue
        cluster_samples = samples.setdefault(str(label), [])
        if len(cluster_samples) < sample_size:
            cluster_samples.append(
                {
                    "post_id": document.post_id,
                    "title": document.title,
                    "keywords": document.keywords,
                }
            )
    return samples


def _mean_keyword_jaccard(
    documents: Sequence[FAQClusterDocument], labels: Sequence[int]
) -> float:
    similarities = []
    for cluster_id in sorted(set(labels) - {-1}):
        token_sets = [
            set(document.keywords.lower().replace(",", " ").split())
            for document, label in zip(documents, labels)
            if label == cluster_id
        ]
        for left_index, left_tokens in enumerate(token_sets):
            for right_tokens in token_sets[left_index + 1 :]:
                union = left_tokens | right_tokens
                similarities.append(len(left_tokens & right_tokens) / len(union) if union else 0.0)
    return sum(similarities) / len(similarities) if similarities else 0.0
