"""Smoke tests for the inference/evaluation/xml-merge FastAPI service.

Redis and Celery are replaced with in-memory fakes, so no broker is needed.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

HEADERS = {"X-API-KEY": "test-api-key"}

JOBS = {
    "inference": (
        "/inference/start",
        "/inference/status",
        "inference_task",
        {"huggingface_repo_name": "org/raw-xml", "upload_repo_name": "org/out"},
    ),
    "evaluation": (
        "/evaluation/hf/start",
        "/evaluation/status",
        "evaluation_task",
        {"huggingface_repo_name": "org/inference"},
    ),
    "xml_merge": (
        "/xml-merge/start",
        "/xml-merge/status",
        "xml_merge_task",
        {"raw_xml_repo": "org/raw-xml", "inference_repo": "org/inference"},
    ),
}


@pytest.fixture
def client():
    from app import main
    from conftest import FAKE_REPOSITORY

    FAKE_REPOSITORY.items.clear()
    main.limiter.enabled = False  # default limit is 4/minute per client
    tasks = {name: MagicMock() for name in ("inference_task", "evaluation_task", "xml_merge_task")}
    with patch.multiple(main, **tasks):
        with TestClient(main.app) as c:
            c.tasks = tasks
            yield c


def test_health_is_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["service"] == "service-trocr-inference"
    assert body["repository"] == "initialized"


@pytest.mark.parametrize("job", JOBS)
def test_endpoints_require_api_key(client, job):
    start, status_path, _, body = JOBS[job]
    assert client.get(status_path).status_code == 401
    assert client.post(start, json=body).status_code == 401
    assert client.get(status_path, headers={"X-API-KEY": "wrong"}).status_code == 401


@pytest.mark.parametrize("job", JOBS)
def test_empty_status_list_is_404(client, job):
    # Current behaviour: an empty list is reported as 404, not [].
    _, status_path, _, _ = JOBS[job]
    assert client.get(status_path, headers=HEADERS).status_code == 404


@pytest.mark.parametrize("job", JOBS)
def test_job_is_created_dispatched_and_queryable(client, job):
    start, status_path, task_name, body = JOBS[job]
    r = client.post(start, headers=HEADERS, json={**body, "huggingface_token": "hf_secret"})
    assert r.status_code == 201, r.text
    created = r.json()
    assert "hf_secret" not in r.text

    task = client.tasks[task_name]
    task.apply_async.assert_called_once()
    call = task.apply_async.call_args.kwargs
    assert call["task_id"] == created["request_id"]
    assert call["kwargs"]["huggingface_token"] == "hf_secret"

    listing = client.get(status_path, headers=HEADERS)
    assert listing.status_code == 200
    assert [j["request_id"] for j in listing.json()] == [created["request_id"]]

    one = client.get(f"{status_path}/{created['request_id']}", headers=HEADERS)
    assert one.status_code == 200
    assert one.json()["request_id"] == created["request_id"]


def test_status_by_id_is_scoped_to_job_type(client):
    start, _, _, body = JOBS["inference"]
    job_id = client.post(start, headers=HEADERS, json=body).json()["request_id"]
    assert client.get(f"/evaluation/status/{job_id}", headers=HEADERS).status_code == 404
    assert client.get(f"/xml-merge/status/{job_id}", headers=HEADERS).status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/inference/status/nope", headers=HEADERS).status_code == 404


def test_invalid_body_is_422(client):
    r = client.post("/inference/start", headers=HEADERS, json={})
    assert r.status_code == 422
