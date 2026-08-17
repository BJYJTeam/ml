from typing import Any, Callable, Dict, Mapping, Sequence

import pandas as pd
from sklearn.cluster import HDBSCAN

from ai_app.clustering import ClusteringConfig, FAQClusterDocument, cluster_documents
from ai_app.faq import (
    FAQDraftError,
    build_faq_prompt,
    filter_duplicate_drafts,
    parse_faq_draft,
    select_representative_qa_pairs,
)
from ai_app.llm import LLMError
from ai_app.retrieval import EmbeddingEncoder


def generate_faq_draft_run(
    questions: pd.DataFrame,
    doctor_answers_by_post_id: Mapping[str, Sequence[str]],
    encoder: EmbeddingEncoder,
    llm_generate: Callable[[str], str],
    config: ClusteringConfig,
    *,
    representatives_per_cluster: int = 3,
    clusterer_factory: Callable[..., Any] = HDBSCAN,
) -> Dict[str, object]:
    drafts = []
    errors = []
    clustering_reports: Dict[str, Dict[str, object]] = {}

    for category, documents in _documents_by_category(questions).items():
        result = cluster_documents(
            documents,
            encoder,
            config,
            clusterer_factory=clusterer_factory,
        )
        clustering_reports[category] = {
            "status": result.status,
            "reason": result.reason,
            "diagnostics": result.diagnostics,
        }
        if result.status != "completed":
            continue

        pairs_by_cluster = select_representative_qa_pairs(
            documents,
            result.labels,
            result.embeddings,
            doctor_answers_by_post_id,
            top_n=representatives_per_cluster,
        )
        for cluster_id, pairs in pairs_by_cluster.items():
            try:
                output = llm_generate(build_faq_prompt(category, pairs))
                drafts.append(
                    parse_faq_draft(
                        output,
                        category=category,
                        source_post_ids=[pair.post_id for pair in pairs],
                    )
                )
            except (FAQDraftError, LLMError) as error:
                errors.append(
                    {
                        "category": category,
                        "cluster_id": cluster_id,
                        "reason": str(error),
                    }
                )

    return {
        "drafts": filter_duplicate_drafts(drafts),
        "errors": errors,
        "clustering": clustering_reports,
    }


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
