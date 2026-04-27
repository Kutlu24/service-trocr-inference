import os
import threading
import time
from flow_inference.inference import Inference
from app.models.inference_models import InferenceResponseModel, StateEnum
from app.storage import StatusRepository
from loguru import logger


async def inference_task(
    repository: StatusRepository,
    huggingface_token: str | None,
    created_status: InferenceResponseModel,
):

    logger.info(
        "Inference task started | request_id=%s | pid=%s | thread=%s",
        created_status.request_id,
        os.getpid(),
        threading.current_thread().name,
    )

    params = created_status.model_dump(
        by_alias=True,
        include={
            "upload_repo_name",
            "private_repo",
            "trocr_model",
            "target_image_size",
            "stop_on_fail",
            "splits",
            "push_to_hub",
        }
    )

    logger.info("Inference params: %s", params)
    logger.info(
        "HF source=%s | upload repo=%s | push_to_hub=%s | token_present=%s",
        created_status.source,
        params.get("upload_repo_name"),
        params.get("push_to_hub"),
        bool(huggingface_token),
    )

    try:
        worker = Inference(
            download_repo_name=created_status.source,
            hf_token=huggingface_token,
            **params,
        )
        logger.info("Inference object created")
    except Exception:
        logger.exception("Failed to construct Inference object")
        created_status.state = StateEnum.FAILED
        await repository.save(created_status)
        raise

    start = time.time()

    try:
        logger.info("Starting inference")
        result = worker.perform_inference()
        elapsed = time.time() - start

        # If your worker returns statistics, attach them here
        if hasattr(result, "total_records"):
            created_status.total_records = result.total_records

        created_status.state = StateEnum.DONE
        created_status.runtime_seconds = elapsed

        logger.info(
            "Inference finished successfully in %.2fs | records=%s",
            elapsed,
            created_status.total_records,
        )

    except Exception as e:
        elapsed = time.time() - start
        created_status.state = StateEnum.FAILED
        created_status.runtime_seconds = elapsed

        logger.exception(
            "Inference failed after %.2fs | error=%s",
            elapsed,
            str(e),
        )

        await repository.save(created_status)

        if created_status.stop_on_fail:
            raise
        return

    await repository.save(created_status)

    logger.info(
        "Final inference status saved | request_id=%s | state=%s",
        created_status.request_id,
        created_status.state,
    )
