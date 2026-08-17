from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence

import numpy as np


class EmbeddingEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> Any:
        ...

    def encode_queries(self, texts: Sequence[str]) -> Any:
        ...


@dataclass(frozen=True)
class QuestionDocument:
    post_id: str
    title: str
    content: str

    @property
    def text(self) -> str:
        return build_question_text(self.title, self.content)


@dataclass(frozen=True)
class SearchResult:
    post_id: str
    score: float


def build_question_text(title: str, content: str) -> str:
    return "\n".join(part for part in (title.strip(), content.strip()) if part)


class SemanticRetriever:
    def __init__(self, documents: Sequence[QuestionDocument], encoder: EmbeddingEncoder):
        self._documents = list(documents)
        self._encoder = encoder
        document_texts = [document.text for document in self._documents]
        embeddings = encoder.encode_documents(document_texts) if document_texts else []
        self._document_embeddings = _normalize_rows(embeddings)

    def search(
        self,
        title: str,
        content: str,
        *,
        exclude_post_id: str | None = None,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> List[SearchResult]:
        query_text = build_question_text(title, content)
        if not query_text or not self._documents or top_k <= 0:
            return []

        query_embedding = _normalize_rows(self._encoder.encode_queries([query_text]))
        scores = self._document_embeddings @ query_embedding[0]
        candidates = [
            SearchResult(document.post_id, float(score))
            for document, score in zip(self._documents, scores)
            if document.post_id != exclude_post_id
            and (min_score is None or float(score) >= min_score)
        ]
        return sorted(candidates, key=lambda result: result.score, reverse=True)[:top_k]


def _normalize_rows(embeddings: Any) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.size == 0:
        return np.empty((0, 0), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
