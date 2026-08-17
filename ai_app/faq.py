from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence

import numpy as np

from ai_app.clustering import ClusteringConfig, FAQClusterDocument


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FAQ_DRAFT_PATH = BASE_DIR / "docs" / "faq_drafts" / "faq-drafts-v1.json"
DEFAULT_FAQ_CLUSTERING_CONFIG = ClusteringConfig(
    min_cluster_size=3,
    min_samples=2,
    cluster_selection_method="leaf",
)


class FAQDraftError(ValueError):
    """Raised when an FAQ draft cannot be safely stored or published."""


@dataclass(frozen=True)
class FAQContextPair:
    post_id: str
    title: str
    content: str
    answer: str


def select_representative_qa_pairs(
    documents: Sequence[FAQClusterDocument],
    labels: np.ndarray,
    embeddings: np.ndarray,
    doctor_answers_by_post_id: Mapping[str, Sequence[str]],
    *,
    top_n: int = 3,
) -> Dict[int, List[FAQContextPair]]:
    if top_n <= 0:
        return {}

    document_list = list(documents)
    normalized_labels = np.asarray(labels, dtype=int)
    normalized_embeddings = np.asarray(embeddings, dtype=float)
    if len(document_list) != len(normalized_labels) or len(document_list) != len(normalized_embeddings):
        raise ValueError("documents, labels, and embeddings must have the same length")

    pairs_by_cluster: Dict[int, List[FAQContextPair]] = {}
    for cluster_id in sorted(set(normalized_labels) - {-1}):
        cluster_indices = np.flatnonzero(normalized_labels == cluster_id)
        ranked_indices = _rank_by_centrality(normalized_embeddings, cluster_indices)
        pairs: List[FAQContextPair] = []
        for index in ranked_indices:
            document = document_list[index]
            answers = doctor_answers_by_post_id.get(document.post_id, [])
            if not answers:
                continue
            answer = str(answers[0]).strip()
            if not answer:
                continue
            pairs.append(
                FAQContextPair(
                    post_id=document.post_id,
                    title=document.title,
                    content=document.content,
                    answer=answer,
                )
            )
            if len(pairs) == top_n:
                break
        if pairs:
            pairs_by_cluster[int(cluster_id)] = pairs
    return pairs_by_cluster


def build_faq_prompt(category: str, pairs: Sequence[FAQContextPair]) -> str:
    if not pairs:
        raise FAQDraftError("at least one historical Q/A pair is required")

    examples = "\n\n".join(
        "\n".join(
            [
                f"[상담 사례 {index}]",
                f"질문 제목: {pair.title}",
                f"질문 내용: {pair.content}",
                f"의료진 답변: {pair.answer}",
            ]
        )
        for index, pair in enumerate(pairs, start=1)
    )
    return f"""[FAQ 초안 생성]
다음은 \"{category}\" 카테고리의 유사 상담 사례입니다. 각 질문과 그 질문에 대응하는 의료진 답변을 함께 참고하세요.

{examples}

공통적으로 도움이 되는 FAQ 초안을 작성하세요. 확정적 진단이나 개별 치료 지시는 피하고, 의료진 상담이 필요한 경우를 안내하세요.

출력 형식은 반드시 아래 두 줄로 작성하세요.
Q: 질문
A: 답변"""


def parse_faq_draft(
    output: str, *, category: str, source_post_ids: Sequence[str]
) -> Dict[str, object]:
    lines = [line.strip() for line in str(output or "").strip().splitlines() if line.strip()]
    question_lines = [index for index, line in enumerate(lines) if line.startswith("Q:")]
    answer_lines = [index for index, line in enumerate(lines) if line.startswith("A:")]
    if len(question_lines) != 1 or len(answer_lines) != 1:
        raise FAQDraftError("FAQ output must contain exactly one Q: line and one A: line")

    question_index = question_lines[0]
    answer_index = answer_lines[0]
    if question_index != 0 or answer_index <= question_index:
        raise FAQDraftError("FAQ output must start with Q: followed by A:")

    content = lines[question_index].removeprefix("Q:").strip()
    answer = "\n".join(
        [lines[answer_index].removeprefix("A:").strip(), *lines[answer_index + 1 :]]
    ).strip()
    if not content or not answer:
        raise FAQDraftError("FAQ question and answer must not be empty")

    return {
        "content": content,
        "answer": answer,
        "category": category,
        "source_post_ids": list(dict.fromkeys(str(post_id) for post_id in source_post_ids)),
        "status": "pending_review",
    }


def filter_duplicate_drafts(drafts: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    unique_drafts: List[Dict[str, object]] = []
    seen_questions = set()
    for draft in drafts:
        normalized_question = re.sub(r"\s+", " ", str(draft.get("content", "")).strip()).lower()
        if not normalized_question or normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        unique_drafts.append(dict(draft))
    return unique_drafts


def load_approved_faqs(path: Path = DEFAULT_FAQ_DRAFT_PATH) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        record = json.load(file)
    drafts = record.get("drafts", [])
    if not isinstance(drafts, list):
        raise FAQDraftError("FAQ draft record must contain a drafts list")
    return [
        {"content": str(draft["content"]), "answer": str(draft["answer"])}
        for draft in drafts
        if draft.get("status") == "approved"
        and str(draft.get("content", "")).strip()
        and str(draft.get("answer", "")).strip()
    ]


def _rank_by_centrality(embeddings: np.ndarray, cluster_indices: np.ndarray) -> List[int]:
    cluster_embeddings = embeddings[cluster_indices]
    if len(cluster_embeddings) <= 1:
        return cluster_indices.tolist()
    similarities = cluster_embeddings @ cluster_embeddings.T
    average_similarities = (similarities.sum(axis=1) - 1.0) / (len(cluster_embeddings) - 1)
    ranked_positions = np.argsort(average_similarities)[::-1]
    return cluster_indices[ranked_positions].tolist()
