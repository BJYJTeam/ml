from unittest.mock import Mock

from ai_app.infrastructure.embeddings import SentenceTransformerEncoder


def test_sentence_transformer_encoder_applies_query_and_document_prefixes(monkeypatch):
    model = Mock()
    model.encode.side_effect = lambda texts: list(texts)
    monkeypatch.setattr("ai_app.infrastructure.embeddings.SentenceTransformer", Mock(return_value=model))

    encoder = SentenceTransformerEncoder(
        "fake-model",
        query_prefix="query: ",
        document_prefix="passage: ",
    )

    assert encoder.encode_documents(["문서"]) == ["passage: 문서"]
    assert encoder.encode_queries(["질의"]) == ["query: 질의"]
    assert encoder.encode(["원문"]) == ["원문"]
