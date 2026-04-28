"""
Redis storage layer for all inference-service job statuses.

Supports:
- inference
- xml_merge
- evaluation
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Type
import redis
from loguru import logger
from pydantic import BaseModel

from app.models.inference_models import InferenceResponseModel
from app.models.raw_xml_upload_models import RawXMLMergeResponseModel
from app.models.evaluation_models import EvaluationResponseModel


_REDIS_HASH_KEY = "inference:statuses"


JOB_MODELS: dict[str, Type[BaseModel]] = {
    "inference": InferenceResponseModel,
    "xml_merge": RawXMLMergeResponseModel,
    "evaluation": EvaluationResponseModel,
}


class RedisStatusRepository:
    """
    Status repository backed by a Redis Hash.

    Shared by:
    - FastAPI process
    - Celery worker process
    - Celery signal handlers
    """

    def __init__(self, redis_url: str):
        self._redis: redis.Redis[str] = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )
        self._redis.ping()

        count = self._redis.hlen(_REDIS_HASH_KEY)
        logger.info(
            f"Initialized RedisStatusRepository "
            f"({count} existing records) at {redis_url}"
        )

    def save(self, status: BaseModel) -> None:
        """Save or update a job status in Redis."""
        self._redis.hset(
            _REDIS_HASH_KEY,
            key=status.request_id,
            value=status.model_dump_json(by_alias=True),
        )

        logger.info(
            "Saved job status | request_id={} | job_type={} | state={}",
            status.request_id,
            status.job_type,
            status.state.value,
        )

    def get_by_id(self, request_id: str) -> BaseModel | None:
        """Retrieve a single status by request ID."""
        data = self._redis.hget(_REDIS_HASH_KEY, request_id)

        if not data:
            logger.warning(f"No status found for request {request_id}")
            return None

        raw = json.loads(data)
        job_type = raw.get("job_type")

        model = JOB_MODELS.get(job_type)
        if not model:
            raise ValueError(f"Unknown job_type in Redis: {job_type}")

        return model.model_validate(raw)

    def get_all(self, job_type: str | None = None) -> list[BaseModel]:
        """Retrieve all statuses, optionally filtered by job_type."""
        all_values = self._redis.hvals(_REDIS_HASH_KEY)

        if not all_values:
            logger.warning("No statuses found")
            return []

        statuses = []

        for value in all_values:
            raw = json.loads(value)

            if job_type and raw.get("job_type") != job_type:
                continue

            model = JOB_MODELS.get(raw.get("job_type"))
            if not model:
                raise ValueError(f"Unknown job_type in Redis: {raw.get('job_type')}")

            statuses.append(model.model_validate(raw))

        return sorted(
            statuses,
            key=lambda s: s.created_at,
            reverse=True,
        )

    def export_to_json(
        self,
        output_path: Path,
        job_type: str | None = None,
    ) -> None:
        """Export all or filtered statuses to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        statuses = self.get_all(job_type=job_type)
        payload = [
            status.model_dump(by_alias=True, mode="json")
            for status in statuses
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Exported {len(payload)} statuses to {output_path}")

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return bool(self._redis.ping())
        except redis.ConnectionError:
            return False

    def close(self) -> None:
        """Close Redis connection."""
        self._redis.close()
        logger.info("Redis connection closed")


@lru_cache(maxsize=1)
def get_redis_repository(redis_url: str) -> RedisStatusRepository:
    return RedisStatusRepository(redis_url=redis_url)