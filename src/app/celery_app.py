"""Celery application configuration for the FLOW Inference Service."""

from datetime import datetime, UTC, timedelta

from celery import Celery
from celery.signals import (
    task_prerun,
    task_postrun,
    task_failure,
    before_task_publish,
    task_retry,
)
from loguru import logger

from app.settings import Settings
from app.models.shared_models import StateEnum
from app.storage import get_redis_repository

settings = Settings()

celery_app = Celery(
    "service_trocr_inference",
    broker=settings.REDIS_URL,
    include=[
        "app.tasks.inference_task",
        "app.tasks.evaluation_task",
        "app.tasks.xml_merge_task",
    ],
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=timedelta(hours=48) if settings.is_production else timedelta(hours=1),
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_default_queue="default",
    task_routes={
        # only route to "gpu" when this deployment's image actually has a CUDA
        # torch build (docker-compose.gpu.yml is the only worker that consumes that queue);
        # otherwise CPU-only deployments would publish to a queue nobody listens to.
        "app.tasks.inference_task": {
            "queue": "gpu" if settings.TORCH_VARIANT != "cpu" else "default",
        },
        "app.tasks.evaluation_task": {
            "queue": "gpu" if settings.TORCH_VARIANT != "cpu" else "default",
        }
    },
)

redis_repository = get_redis_repository(redis_url=settings.REDIS_URL)


@before_task_publish.connect
def before_task_publish_handler(headers=None, body=None, **kwargs):
    """Fire before task is sent to the broker."""
    info = headers or {}
    task_id = info.get("id") or info.get("task_id")
    task_name = info.get("task")

    logger.info(f"Publishing task {task_id} to broker | task={task_name}")

    # In this service, main.py should already create and save the initial status.
    # So only warn if it does not exist.
    if task_id and redis_repository.get_by_id(task_id):
        return

    logger.warning(f"Task {task_id} was published but no status exists in Redis yet")


@task_prerun.connect
def task_prerun_handler(task_id=None, **kwargs):
    """Fire when task begins executing."""
    logger.info(f"Task {task_id} prerun")

    task = redis_repository.get_by_id(task_id)
    if task is None:
        logger.warning(f"Task {task_id} not found in repository during prerun")
        return

    task.state = StateEnum.IN_PROGRESS
    task.started_at = datetime.now(UTC)
    redis_repository.save(task)


@task_postrun.connect
def task_postrun_handler(task_id=None, state=None, retval=None, **kwargs):
    """Fire when task finishes successfully."""
    logger.info(f"Task {task_id} postrun | celery_state={state}")

    if state != "SUCCESS":
        return

    task = redis_repository.get_by_id(task_id)
    if task is None:
        logger.warning(f"Task {task_id} not found in repository during postrun")
        return

    task.state = StateEnum.COMPLETED
    task.ended_at = datetime.now(UTC)

    if task.started_at:
        task.runtime_seconds = (task.ended_at - task.started_at).total_seconds()

    if isinstance(retval, dict):
        for key, value in retval.items():
            if hasattr(task, key):
                setattr(task, key, value)

    redis_repository.save(task)


@task_failure.connect
def task_failure_handler(task_id=None, exception=None, sender=None, **kwargs):
    """Fire when task fails permanently."""
    logger.info(f"Task {task_id} failure")

    if sender and sender.request.retries < sender.max_retries:
        logger.info(f"Task {task_id} will retry, skipping final failure update")
        return

    task = redis_repository.get_by_id(task_id)
    if task is None:
        logger.warning(f"Task {task_id} not found in repository during failure")
        return

    task.state = StateEnum.FAILED
    task.ended_at = datetime.now(UTC)
    task.error_message = str(exception)

    if task.started_at:
        task.runtime_seconds = (task.ended_at - task.started_at).total_seconds()

    redis_repository.save(task)


@task_retry.connect
def task_retry_handler(request=None, reason=None, einfo=None, sender=None, **kwargs):
    """Fire when task retries."""
    task_id = getattr(request, "id", None)

    if task_id is None and sender is not None:
        task_id = getattr(getattr(sender, "request", None), "id", None)

    if task_id is None:
        logger.warning(
            "Task retry signal received without task id; skipping status update. "
            f"request={request!r}, sender={sender!r}, reason={reason!r}"
        )
        return

    logger.info(f"Task {task_id} retry")

    task = redis_repository.get_by_id(task_id)
    if task is None:
        logger.warning(f"Task {task_id} not found in repository during retry")
        return

    task.state = StateEnum.IN_PROGRESS
    task.retry_count += 1
    task.error_message = str(reason)

    redis_repository.save(task)
