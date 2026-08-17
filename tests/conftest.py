import os

import pytest
import requests

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_server.settings")


@pytest.fixture(autouse=True)
def block_external_http_requests(monkeypatch):
    def fail_request(*args, **kwargs):
        raise AssertionError("unit tests must not make external HTTP requests")

    monkeypatch.setattr(requests.sessions.Session, "request", fail_request)
