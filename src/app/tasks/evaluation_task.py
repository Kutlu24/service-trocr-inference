import time
from flow_inference.evaluation import Evaluation
from app.models.evaluation_models import EvaluationResponseModel, StateEnum
from app.storage import StatusRepository
from loguru import logger


async def evaluation_task(
    repository: StatusRepository,
    huggingface_token: str | None,
    created_status: EvaluationResponseModel,
    splits: list[str] | None = None,
):
    logger.info(
        "Evaluation started | request_id=%s | source=%s",
        created_status.request_id,
        created_status.source,
    )

    try:
        worker = Evaluation(
            download_repo_name=created_status.source,
            hf_token=huggingface_token,
            splits=splits,
        )
        logger.info("Evaluation object created")
    except Exception:
        logger.exception("Failed to construct Evaluation object")
        created_status.state = StateEnum.FAILED
        await repository.save(created_status)
        raise

    start = time.time()

    try:
        logger.info("Starting evaluation")
        result = worker.perform_evaluation()
        elapsed = time.time() - start

        # If Evaluation returns metrics, attach them
        if isinstance(result, dict):
            created_status.cer = result.get("cer")
            created_status.gt_lines = result.get("gt_lines")
            created_status.hypothesis_lines = result.get("hypothesis_lines")

        created_status.state = StateEnum.DONE
        created_status.runtime_seconds = elapsed

        logger.info(
            "Evaluation finished in %.2fs | CER=%s",
            elapsed,
            created_status.cer,
        )

    except Exception as e:
        elapsed = time.time() - start
        created_status.state = StateEnum.FAILED
        created_status.runtime_seconds = elapsed

        logger.exception(
            "Evaluation failed after %.2fs | error=%s",
            elapsed,
            str(e),
        )

        await repository.save(created_status)
        return

    await repository.save(created_status)

    logger.info(
        "Final evaluation status saved | request_id=%s | state=%s",
        created_status.request_id,
        created_status.state,
    )