from flow_inference.inference import Inference
from app.models.inference_models import InferenceResponseModel


async def inference_task(
        huggingface_token: str,
        created_status: InferenceResponseModel
):
    print("Inference started")

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

    worker = Inference(
        download_repo_name=created_status.source,
        hf_token=huggingface_token,
        **params
    )

    try:
        result = worker.perform_inference()
        state = "completed"
    except Exception as e:
        print(f"Inference failed: {e}")
        state = "failed"

    # write to status file...
