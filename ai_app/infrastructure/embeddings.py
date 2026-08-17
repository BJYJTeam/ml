from typing import Any, Sequence

from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder:
    """Adapter around SentenceTransformer for retrieval and evaluation."""

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    def encode(self, texts: Sequence[str]) -> Any:
        return self._model.encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> Any:
        return self.encode([f"{self._document_prefix}{text}" for text in texts])

    def encode_queries(self, texts: Sequence[str]) -> Any:
        return self.encode([f"{self._query_prefix}{text}" for text in texts])
