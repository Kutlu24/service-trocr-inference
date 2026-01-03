from flow_inference.evaluation import Evaluation
from app.models.evaluation_models import EvaluationResponseModel


async def evaluation_task(
    huggingface_token: str,
    created_status: EvaluationResponseModel,
    splits: list[str] | None = None,
):
    print("Evaluation started")

    worker = Evaluation(
        download_repo_name=created_status.source,
        hf_token=huggingface_token,
        splits=splits,
    )

    try:
        files = worker.perform_evaluation()
        state = "completed"
    except Exception as e:
        print(f"Evaluation failed: {e}")
        state = "failed"

