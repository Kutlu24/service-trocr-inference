from loguru import logger
from app.celery_app import celery_app
from app.models.raw_xml_upload_models import RawXMLMergeResponseModel
from flow_inference.write_inference_to_raw_xml import InferenceToRawXMLWriter

from app.tasks.retry_utils import is_retryable_error


@celery_app.task(
    bind=True,
    name="app.tasks.xml_merge_task",
    max_retries=3,
    retry_backoff=30,
    retry_backoff_max=120,
    retry_jitter=True,
    time_limit=3600,
    soft_time_limit=3300,
)
def xml_merge_task(
    self,
    status_dict: dict,
    huggingface_token: str | None = None,
) -> dict:
    try:
        return _run_xml_merge(
            status_dict=status_dict,
            huggingface_token=huggingface_token,
        )

    except MemoryError:
        logger.exception("Memory error, do not retry")
        raise

    except Exception as e:
        logger.exception(f"XML merge failed for request: {e}")

        if "Cannot allocate memory" in str(e):
            logger.error("OOM error in task, will not retry")
            raise

        if is_retryable_error(e):
            logger.warning("Transient XML merge error detected; retrying task.")
            raise self.retry(exc=e)

        logger.error("Non-retryable XML merge error; failing task without retry.")
        raise


def _run_xml_merge(
    status_dict: dict,
    huggingface_token: str | None = None,
) -> dict:
    status = RawXMLMergeResponseModel.model_validate(status_dict)

    logger.info(
        "XML merge task started | request_id={} | raw={} | inference={}",
        status.request_id,
        status.source_raw_repo,
        status.source_inference_repo,
    )

    worker = InferenceToRawXMLWriter(
        raw_xml_repo=status.source_raw_repo,
        inference_repo=status.source_inference_repo,
        token=huggingface_token,
        allow_source_repo_update=status.allow_source_repo_update,
    )

    result = worker.process_and_upload(
        output_repo=status.output_repo_name,
        upload_mode=status.upload_mode,
        private=status.private_repo,
    )

    if isinstance(result, dict):
        return {
            "updated_records": result.get("updated_records", 0),
        }

    return {}