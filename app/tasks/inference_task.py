import logging
import os
import threading
import time

from flow_inference.inference import Inference
from app.models.inference_models import InferenceResponseModel


async def inference_task(
        huggingface_token: str,
        created_status: InferenceResponseModel
):
    print("Inference started")
    logger = logging.getLogger("inference.task")

    logger.info(
        "Inference task started | pid=%s | thread=%s",
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
            "push_to_hub"
        }
    )

    logger.info("Inference params: %s", params)
    logger.info(
        "HF source repo=%s | upload repo=%s | push_to_hub=%s | token_present=%s",
        created_status.source,
        params.get("upload_repo_name"),
        params.get("push_to_hub"),
        bool(huggingface_token),
    )

    try:
        logger.info("Constructing Inference object...")
        worker = Inference(
            download_repo_name=created_status.source,
            hf_token=huggingface_token,
            **params
        )
        logger.info("Inference object constructed successfully")
    except Exception:
        logger.exception("Failed during Inference() construction")
        raise

    start = time.time()
    logger.info("Starting perform_inference()")

    try:
        result = worker.perform_inference()
        elapsed = time.time() - start
        logger.info(
            "perform_inference() completed in %.2fs | result=%s",
            elapsed,
            type(result).__name__,
        )
        state = "completed"
        logger.info(
            "Inference task finished | state=%s | pid=%s",
            state,
            os.getpid(),
        )
    except Exception:
        elapsed = time.time() - start
        logger.exception(
            "perform_inference() failed after %.2fs",
            elapsed,
        )
        state = "failed"
        logger.info(
            "Inference task finished with error | state=%s | pid=%s",
            state,
            os.getpid(),
        )

    # write to status file...
