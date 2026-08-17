import importlib
import sys
from unittest.mock import Mock

import numpy as np


class FakeModel:
    def encode(self, texts, convert_to_tensor=False):
        values = np.ones((len(texts), 2))
        return FakeTensor(values) if convert_to_tensor else values


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self.values


def test_inference_import_defers_embedding_model_initialization(monkeypatch):
    model_factory = Mock(return_value=FakeModel())
    import ai_app.infrastructure.embeddings as embeddings
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", model_factory)
    monkeypatch.setattr(embeddings, "SentenceTransformer", model_factory)
    sys.modules.pop("ai_app.inference", None)

    module = importlib.import_module("ai_app.inference")
    monkeypatch.setitem(sys.modules, "ai_app.inference", module)

    assert model_factory.call_count == 0
