import time
from flow_inference.write_inference_to_raw_xml import InferenceToRawXMLWriter
from app.models.raw_xml_upload_models import RawXMLMergeResponseModel, StateEnum
from app.storage import StatusRepository
from loguru import logger


async def xml_merge_task(
    repository: StatusRepository,
    huggingface_token: str | None,
    created_status: RawXMLMergeResponseModel,
):

    logger.info(
        "XML merge started | request_id=%s | raw=%s | inference=%s",
        created_status.request_id,
        created_status.source_raw_repo,
        created_status.source_inference_repo,
    )

    try:
        worker = InferenceToRawXMLWriter(
            raw_xml_repo=created_status.source_raw_repo,
            inference_repo=created_status.source_inference_repo,
            token=huggingface_token,
        )
        logger.info("XML merge worker created")
    except Exception:
        logger.exception("Failed to create XML merge worker")
        created_status.state = StateEnum.FAILED
        await repository.save(created_status)
        raise

    start = time.time()

    try:
        logger.info("Starting XML merge")
        result = worker.process_and_upload(
            output_repo=created_status.output_repo_name
        )
        elapsed = time.time() - start

        if isinstance(result, dict):
            created_status.updated_records = result.get("updated_records", 0)

        created_status.state = StateEnum.DONE
        created_status.runtime_seconds = elapsed

        logger.info(
            "XML merge finished in %.2fs | updated_records=%s",
            elapsed,
            created_status.updated_records,
        )

    except Exception as e:
        elapsed = time.time() - start
        created_status.state = StateEnum.FAILED
        created_status.runtime_seconds = elapsed

        logger.exception(
            "XML merge failed after %.2fs | error=%s",
            elapsed,
            str(e),
        )

        await repository.save(created_status)
        return

    await repository.save(created_status)

    logger.info(
        "Final XML merge status saved | request_id=%s | state=%s",
        created_status.request_id,
        created_status.state,
    )
