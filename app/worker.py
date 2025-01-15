from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models import InferenceDBModel
from flow_inference.inference import Inference


async def update_progress(status: InferenceDBModel, db: AsyncIOMotorDatabase, first: False) -> None:
    """
    Update the progress of an inference job.

    :param status: The status of the inference job to update.
    :param db: to connect to the MongoDB.
    :param first:
    """
    if isinstance(status.id, str):
        status.id = ObjectId(status.id)

    if first:
        status_dict = status.model_dump(by_alias=True)
    else:
        status_dict = status.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=True)

    del status_dict["_id"]

    await db.inference_status.update_one(
        {"_id": status.id},
        {"$set": status_dict},
    )


async def inference_task(
        github_access_token: str,
        created_status: InferenceDBModel,
        db: AsyncIOMotorDatabase
) -> None:
    await update_progress(created_status, db, first=True)

    async def callback(progress_update_status: dict) -> None:
        progress_update_status["_id"] = progress_update_status["process_id"]
        del progress_update_status["process_id"]
        status_in_db = InferenceDBModel(**progress_update_status)
        try:
            await update_progress(status_in_db, db, first=False)
        except Exception as e:
            print(f"Failed to update inference status: {e}")

    created_status_dict = created_status.model_dump(by_alias=True, exclude={"password"})
    created_status_dict["process_id"] = str(created_status_dict["_id"])
    del created_status_dict["_id"]

    inference = Inference(
        **created_status_dict,
        github_access_token=github_access_token,
        callback_inference=callback,
    )
    await inference.perform_inference()
