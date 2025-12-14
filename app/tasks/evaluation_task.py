from flow_inference.evaluation import Evaluation
from app.models.evaluation_models import EvaluationResponseModel


async def evaluation_task(
        huggingface_token: str,
        created_status: EvaluationResponseModel
):
    print("Evaluation started")

    params = created_status.model_dump(
        by_alias=True,
        include={"splits"}
    )

    worker = Evaluation(
        download_repo_name=created_status.source,
        hf_token=huggingface_token,
        **params
    )

    try:
        files = worker.perform_evaluation()
        state = "completed"
    except Exception as e:
        print(f"Evaluation failed: {e}")
        state = "failed"

