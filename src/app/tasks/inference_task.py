import os
import threading
from loguru import logger
from app.celery_app import celery_app
from app.models.inference_models import InferenceResponseModel
from flow_inference.inference import Inference

@celery_app.task(
    bind=True,
    name="app.tasks.inference_task",
    max_retries=3,
    retry_backoff=30,
    retry_backoff_max=120,
    retry_jitter=True,
    time_limit=3600,
    soft_time_limit=3300,
)
def inference_task(
    self,
    status_dict: dict,
    huggingface_token: str | None = None,
) -> dict:
    try:
        return _run_inference(status_dict, huggingface_token)

    except MemoryError:
        logger.exception("Memory error, do not retry")
        raise

    except Exception as e:
        logger.exception(f"Inference failed for request: {e}")

        if "Cannot allocate memory" in str(e):
            logger.error("OOM error in task, will not retry")
            raise

        raise self.retry(exc=e)

def _run_inference(
    status_dict: dict,
    huggingface_token: str | None = None,
) -> dict:
    status = InferenceResponseModel.model_validate(status_dict)

    logger.info(
        "Inference task started | request_id={} | pid={} | thread={}",
        status.request_id,
        os.getpid(),
        threading.current_thread().name,
    )

    params = status.model_dump(
        by_alias=True,
        include={
            "upload_repo_name",
            "private_repo",
            "trocr_model",
            "target_image_size",
            "stop_on_fail",
            "splits",
            "push_to_hub",
            "upload_mode",
            "allow_source_repo_update",
        },
    )

    worker = Inference(
        download_repo_name=status.source,
        hf_token=huggingface_token,
        **params,
    )

    result = worker.perform_inference()

    total_records = 0
    if isinstance(result, dict):
        total_records = sum(len(df) for df in result.values())

    return {"total_records": total_records}