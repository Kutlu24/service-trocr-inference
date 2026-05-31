from loguru import logger
from app.celery_app import celery_app
from app.models.evaluation_models import EvaluationResponseModel
from flow_inference.evaluation import Evaluation

from app.tasks.retry_utils import is_retryable_error
from datetime import datetime, UTC
from app.tasks.status_upload import upload_status_to_huggingface


@celery_app.task(
    bind=True,
    name="app.tasks.evaluation_task",
    max_retries=3,
    retry_backoff=30,
    retry_backoff_max=120,
    retry_jitter=True,
    time_limit=3600,
    soft_time_limit=3300,
)
def evaluation_task(
    self,
    status_dict: dict,
    huggingface_token: str | None = None,
    splits: list[str] | None = None,
) -> dict:
    try:
        return _run_evaluation(
            status_dict=status_dict,
            huggingface_token=huggingface_token,
            splits=splits,
        )

    except MemoryError:
        logger.exception("Memory error, do not retry")
        raise

    except Exception as e:
        logger.exception(f"Evaluation failed for request: {e}")

        if "Cannot allocate memory" in str(e):
            logger.error("OOM error in task, will not retry")
            raise

        if is_retryable_error(e):
            logger.warning("Transient evaluation error detected; retrying task.")
            raise self.retry(exc=e)

        logger.error("Non-retryable evaluation error; failing task without retry.")
        raise

def _run_evaluation(
    status_dict: dict,
    huggingface_token: str | None = None,
    splits: list[str] | None = None,
) -> dict:
    started_at = datetime.now(UTC)

    status = EvaluationResponseModel.model_validate(status_dict)

    worker = Evaluation(
        evaluation_repo_name=status.source,
        hf_token=huggingface_token,
        splits=splits,
    )

    result = worker.perform_evaluation()

    if isinstance(result, dict):
        result_updates = {
            "cer": result.get("cer"),
            "gt_lines": result.get("gt_lines"),
            "hypothesis_lines": result.get("hypothesis_lines"),
        }
    else:
        result_updates = {}

    upload_status_to_huggingface(
        status=status,
        huggingface_token=huggingface_token,
        result_updates=result_updates,
        started_at=started_at,
    )

    return result_updates