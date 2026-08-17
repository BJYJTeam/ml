from unittest.mock import Mock

import pytest
import requests

from ai_app.llm import (
    GemmaClient,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
)


def test_gemma_client_requires_url_and_token_before_a_network_call():
    client = GemmaClient(api_url="", api_token="")

    with pytest.raises(LLMConfigurationError):
        client.generate("질문")


def test_gemma_client_retries_timeout_then_returns_controlled_provider_error(monkeypatch):
    post = Mock(side_effect=requests.Timeout("slow"))
    monkeypatch.setattr("ai_app.llm.requests.post", post)
    client = GemmaClient(api_url="https://example.test", api_token="token", max_retries=1)

    with pytest.raises(LLMProviderError, match="unavailable"):
        client.generate("질문")

    assert post.call_count == 2
    assert post.call_args.kwargs["timeout"] == 15.0


def test_gemma_client_rejects_non_success_and_malformed_json(monkeypatch):
    failed_response = Mock(ok=False, status_code=503)
    malformed_response = Mock(ok=True)
    malformed_response.json.side_effect = ValueError("bad json")
    post = Mock(side_effect=[failed_response, malformed_response])
    monkeypatch.setattr("ai_app.llm.requests.post", post)
    client = GemmaClient(api_url="https://example.test", api_token="token", max_retries=0)

    with pytest.raises(LLMProviderError):
        client.generate("질문")
    with pytest.raises(LLMResponseError):
        client.generate("질문")


def test_gemma_client_rejects_an_empty_model_response(monkeypatch):
    response = Mock(ok=True)
    response.json.return_value = {"response": "  "}
    monkeypatch.setattr("ai_app.llm.requests.post", Mock(return_value=response))
    client = GemmaClient(api_url="https://example.test", api_token="token", max_retries=0)

    with pytest.raises(LLMResponseError):
        client.generate("질문")


def test_gemma_client_returns_a_nonempty_successful_response(monkeypatch):
    response = Mock(ok=True)
    response.json.return_value = {"response": "생성된 답변"}
    monkeypatch.setattr("ai_app.llm.requests.post", Mock(return_value=response))
    client = GemmaClient(api_url="https://example.test", api_token="token", max_retries=0)

    assert client.generate("질문") == "생성된 답변"
