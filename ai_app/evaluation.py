import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QA_DATA_PATH = BASE_DIR / "docs" / "qa_DB_tag_json.csv"
DEFAULT_DOCTOR_COMMENTS_PATH = BASE_DIR / "docs" / "post_comments.json"
DEFAULT_RETRIEVAL_VALIDATION_PATH = (
    BASE_DIR / "docs" / "validation" / "retrieval-relevance-v1.json"
)
REQUIRED_QA_COLUMNS = {"id", "tag", "title", "content"}


class DataIntegrityError(ValueError):
    """Raised when the historical Q&A data cannot support ML evaluation."""


def validate_qa_columns(questions: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_QA_COLUMNS.difference(questions.columns))
    if missing_columns:
        raise DataIntegrityError(
            f"Q&A data is missing required columns: {', '.join(missing_columns)}"
        )


def filter_usable_questions(questions: pd.DataFrame) -> pd.DataFrame:
    validate_qa_columns(questions)
    filtered = questions[
        questions["content"].notna()
        & ~questions["content"].str.lower().str.contains("content not found")
    ].copy()
    filtered["id"] = filtered["id"].astype(str)
    filtered["title"] = filtered["title"].fillna("")
    return filtered


def load_qa_dataframe(path: Path = DEFAULT_QA_DATA_PATH) -> pd.DataFrame:
    return filter_usable_questions(load_raw_qa_dataframe(path))


def load_raw_qa_dataframe(path: Path = DEFAULT_QA_DATA_PATH) -> pd.DataFrame:
    questions = pd.read_csv(path)
    validate_qa_columns(questions)
    return questions


def load_doctor_comments(path: Path = DEFAULT_DOCTOR_COMMENTS_PATH) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_validation_judgments(path: Path = DEFAULT_RETRIEVAL_VALIDATION_PATH) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        judgments = json.load(file)
    if not isinstance(judgments, list):
        raise DataIntegrityError("Validation judgments must be a JSON list")
    return judgments


def build_data_integrity_report(
    questions: pd.DataFrame, comments: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    validate_qa_columns(questions)
    usable_questions = filter_usable_questions(questions)
    categories = usable_questions["tag"].str.extract(r"\[(.*?)\]")[0].fillna("<missing>")
    doctor_post_ids = {
        str(comment.get("post_id"))
        for comment in comments
        if str(comment.get("author", "")).lower() == "doctor"
    }
    usable_post_ids = usable_questions["id"].tolist()
    doctor_answered_records = sum(post_id in doctor_post_ids for post_id in usable_post_ids)
    usable_records = len(usable_questions)

    return {
        "total_records": len(questions),
        "usable_records": usable_records,
        "empty_title_records": int((usable_questions["title"].str.strip() == "").sum()),
        "duplicate_id_records": int(usable_questions["id"].duplicated().sum()),
        "doctor_answered_records": doctor_answered_records,
        "doctor_answer_coverage": (
            doctor_answered_records / usable_records if usable_records else 0.0
        ),
        "category_counts": {
            str(category): int(count)
            for category, count in categories.value_counts().sort_index().items()
        },
    }


def summarize_clusters(labels: Iterable[int]) -> Dict[str, Any]:
    normalized_labels = [int(label) for label in labels]
    total = len(normalized_labels)
    noise_count = sum(label == -1 for label in normalized_labels)
    cluster_sizes = Counter(label for label in normalized_labels if label != -1)

    return {
        "n_samples": total,
        "n_clusters": len(cluster_sizes),
        "noise_count": noise_count,
        "noise_ratio": noise_count / total if total else 0.0,
        "cluster_sizes": {
            str(label): count for label, count in sorted(cluster_sizes.items())
        },
        "largest_cluster_ratio": max(cluster_sizes.values(), default=0) / total if total else 0.0,
    }


def calculate_retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    judgments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not judgments:
        return {
            "n_queries": 0,
            "n_queries_with_relevance": 0,
            "top_1_recall": 0.0,
            "top_3_recall": 0.0,
        }

    top_1_hits = 0
    top_3_hits = 0
    queries_with_relevance = 0
    for judgment in judgments:
        ranking = rankings.get(str(judgment["query_id"]), [])
        relevant_post_ids = set(judgment.get("relevant_post_ids", []))
        if not relevant_post_ids:
            continue
        queries_with_relevance += 1
        top_1_hits += bool(set(ranking[:1]).intersection(relevant_post_ids))
        top_3_hits += bool(set(ranking[:3]).intersection(relevant_post_ids))

    query_count = len(judgments)
    return {
        "n_queries": query_count,
        "n_queries_with_relevance": queries_with_relevance,
        "top_1_recall": top_1_hits / queries_with_relevance if queries_with_relevance else 0.0,
        "top_3_recall": top_3_hits / queries_with_relevance if queries_with_relevance else 0.0,
    }


def resolve_validation_queries(
    judgments: Sequence[Mapping[str, Any]], questions: pd.DataFrame
) -> List[Dict[str, Any]]:
    validate_qa_columns(questions)
    questions_by_id = questions.set_index(questions["id"].astype(str), drop=False)
    resolved_queries: List[Dict[str, Any]] = []

    for judgment in judgments:
        source_post_id = judgment.get("source_post_id")
        if source_post_id:
            source_post_id = str(source_post_id)
            if source_post_id not in questions_by_id.index:
                raise DataIntegrityError(
                    f"Validation source post does not exist: {source_post_id}"
                )
            source = questions_by_id.loc[source_post_id]
            title = str(source["title"] or "")
            content = str(source["content"] or "")
            exclude_post_id = source_post_id
        else:
            title = str(judgment.get("title", "") or "")
            content = str(judgment.get("content", "") or "")
            exclude_post_id = None

        if not title.strip() and not content.strip():
            raise DataIntegrityError(
                f"Validation query has no title or content: {judgment.get('query_id')}"
            )

        resolved_queries.append(
            {
                "query_id": str(judgment["query_id"]),
                "title": title,
                "content": content,
                "exclude_post_id": exclude_post_id,
                "relevant_post_ids": list(judgment.get("relevant_post_ids", [])),
            }
        )

    return resolved_queries


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
