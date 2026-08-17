import os

import requests


GEMMA_MODEL_NAME = "gemma3:12b"


class LLMError(RuntimeError):
    """Base error for controlled LLM integration failures."""


class LLMConfigurationError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


class GemmaClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_token: str,
        model_name: str = GEMMA_MODEL_NAME,
        timeout_seconds: float = 15.0,
        max_retries: int = 1,
    ):
        self._api_url = api_url.strip()
        self._api_token = api_token.strip()
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    @classmethod
    def from_environment(cls) -> "GemmaClient":
        return cls(
            api_url=os.getenv("API_URL", ""),
            api_token=os.getenv("GEMMA_API_TOKEN", ""),
            timeout_seconds=float(os.getenv("GEMMA_TIMEOUT_SECONDS", "15")),
            max_retries=int(os.getenv("GEMMA_MAX_RETRIES", "1")),
        )

    def generate(self, prompt: str) -> str:
        if not self._api_url or not self._api_token:
            raise LLMConfigurationError("Gemma API URL and token must be configured")
        if self._timeout_seconds <= 0 or self._max_retries < 0:
            raise LLMConfigurationError("Gemma timeout and retry settings must be non-negative")

        for attempt in range(self._max_retries + 1):
            try:
                response = requests.post(
                    self._api_url,
                    headers={
                        "Authorization": self._api_token,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model_name,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=self._timeout_seconds,
                )
            except requests.RequestException as error:
                if attempt == self._max_retries:
                    raise LLMProviderError("Gemma service is temporarily unavailable") from error
                continue

            if not response.ok:
                raise LLMProviderError("Gemma service returned an unsuccessful response")
            try:
                payload = response.json()
            except ValueError as error:
                raise LLMResponseError("Gemma service returned malformed JSON") from error

            content = payload.get("response") if isinstance(payload, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise LLMResponseError("Gemma service returned an empty response")
            return content

        raise LLMProviderError("Gemma service is temporarily unavailable")


def generate_gemma_answer(prompt: str) -> str:
    return GemmaClient.from_environment().generate(prompt)
