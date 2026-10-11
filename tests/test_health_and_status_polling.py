"""Health endpoint contents and rate limits of status polling.

Self-contained: Redis and Celery are replaced with in-memory fakes, no broker needed.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("REDIS_URL", "redis://:s3cret@redis.example:6379/0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.storage as storage  # noqa: E402


class _FakeRepository:
    def __init__(self):
        self.items = {}

    def save(self, status):
        self.items[status.request_id] = status

    def get_by_id(self, request_id):
        return self.items.get(request_id)

    def get_all(self, job_type=None):
        return [s for s in self.items.values()
                if job_type is None or getattr(s, "job_type", None) == job_type]

    def ping(self):
        return True

    def close(self):
        pass


# app.celery_app connects to Redis at import time: patch before app.main is imported.
_REPOSITORY = _FakeRepository()
storage.get_redis_repository = lambda redis_url=None: _REPOSITORY

HEADERS = {"X-API-KEY": "test-api-key"}
STATUS_PATHS = [
    "/inference/status",
    "/inference/status/some-id",
    "/evaluation/status",
    "/evaluation/status/some-id",
    "/xml-merge/status",
    "/xml-merge/status/some-id",
]


@pytest.fixture
def client():
    from app import main

    _REPOSITORY.items.clear()
    main.limiter.enabled = True
    main.limiter.reset()
    tasks = {n: MagicMock() for n in ("inference_task", "evaluation_task", "xml_merge_task")}
    with patch.multiple(main, **tasks):
        with TestClient(main.app) as c:
            yield c


def test_health_does_not_expose_redis_url(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "redis_url" not in r.json()
    assert "s3cret" not in r.text


@pytest.mark.parametrize("path", STATUS_PATHS)
def test_status_polling_is_not_throttled_at_the_default_limit(client, path):
    # The default limit is 4/minute; an n8n workflow polls twice a minute per job.
    codes = [client.get(path, headers=HEADERS).status_code for _ in range(20)]
    assert 429 not in codes


def test_status_polling_still_has_an_upper_bound(client):
    codes = [client.get("/inference/status/x", headers=HEADERS).status_code for _ in range(70)]
    assert codes[-1] == 429


def test_job_start_keeps_the_strict_default_limit(client):
    body = {"huggingface_repo_name": "org/raw-xml", "upload_repo_name": "org/out"}
    codes = [client.post("/inference/start", headers=HEADERS, json=body).status_code
             for _ in range(5)]
    assert codes[:4] == [201] * 4
    assert codes[4] == 429
