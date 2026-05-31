import json
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from loguru import logger
from pydantic import BaseModel

from app.models.shared_models import StateEnum


def _status_target_repo(status: BaseModel) -> str | None:
    """
    Decide which Hugging Face repo should receive the status JSON.
    """

    job_type = getattr(status, "job_type", None)

    if job_type == "inference":
        # Only upload inference status if the inference result is pushed to HF.
        if not getattr(status, "push_to_hub", True):
            return None

        return getattr(status, "upload_repo_name", None) or getattr(status, "source", None)

    if job_type == "evaluation":
        return getattr(status, "source", None)

    if job_type == "xml_merge":
        return getattr(status, "output_repo_name", None)

    return None


def _status_path_in_repo(status: BaseModel) -> str:
    job_type = getattr(status, "job_type", "job")
    request_id = getattr(status, "request_id", "unknown-request")

    return f"status/{job_type}/{request_id}.json"


def upload_status_to_huggingface(
    status: BaseModel,
    huggingface_token: str | None,
    result_updates: dict[str, Any] | None = None,
    started_at: datetime | None = None,
) -> bool:
    """
    Upload final job status JSON to the relevant Hugging Face dataset repo.

    The token is intentionally passed directly from the Celery task and is not
    stored in Redis.
    """

    target_repo = _status_target_repo(status)

    if not target_repo:
        logger.info(
            "Skipping status upload | request_id={} | job_type={} | no target repo",
            getattr(status, "request_id", None),
            getattr(status, "job_type", None),
        )
        return False

    if not huggingface_token:
        logger.warning(
            "Skipping status upload | request_id={} | target_repo={} | no Hugging Face token",
            getattr(status, "request_id", None),
            target_repo,
        )
        return False

    try:
        ended_at = datetime.now(UTC)

        status_payload = status.model_dump(
            by_alias=True,
            mode="json",
            exclude={"huggingface_token"},
        )

        if result_updates:
            status_payload.update(result_updates)

        status_payload["state"] = StateEnum.COMPLETED.value
        status_payload["ended_at"] = ended_at.isoformat()

        if started_at is not None:
            status_payload["runtime"] = (ended_at - started_at).total_seconds()

        path_in_repo = _status_path_in_repo(status)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / Path(path_in_repo).name

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    status_payload,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(temp_path),
                path_in_repo=path_in_repo,
                repo_id=target_repo,
                repo_type="dataset",
                token=huggingface_token,
                commit_message=(
                    f"Add {getattr(status, 'job_type', 'job')} status "
                    f"for job {getattr(status, 'request_id', '')[:8]}"
                ),
            )

        logger.info(
            "Uploaded status JSON | request_id={} | target_repo={} | path={}",
            getattr(status, "request_id", None),
            target_repo,
            path_in_repo,
        )

        return True

    except Exception as e:
        logger.warning(
            "Failed to upload status JSON | request_id={} | target_repo={} | error={}",
            getattr(status, "request_id", None),
            target_repo,
            e,
        )
        return False