import io
import logging
import os.path
import shutil
import zipfile
from asyncio import gather
from contextlib import asynccontextmanager

from motor.motor_asyncio import AsyncIOMotorDatabase

from bson import ObjectId
from fastapi import (
    FastAPI,
    Body,
    Depends,
    HTTPException,
    status,
    BackgroundTasks
)
from fastapi.encoders import ENCODERS_BY_TYPE
from fastapi.responses import StreamingResponse

from app.database import mongo_database
from app.db_connection import (
    ping_mongo_db_server,
)

from app.models import InferenceRequestModel, InferenceResponseModel, InferenceDBModel, PyObjectId
from app.worker import inference_task

logger = logging.getLogger('uvicorn')

ENCODERS_BY_TYPE[ObjectId] = str
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for FastAPI application lifespan events.

    :param: app: The FastAPI application instance.

    :return: None: This function does not yield any value but ensures database tables are created on startup.
    """
    await gather(
        ping_mongo_db_server(),
    )

    db = mongo_database()
    collection_list = await db.list_collection_names()
    if "inference_status" not in collection_list:
        await db.create_collection("inference_status")

    yield


app = FastAPI(lifespan=lifespan)


async def get_inference_status_or_404(
        inference_id: str,
        db: AsyncIOMotorDatabase = Depends(mongo_database),
) -> InferenceDBModel:
    """
    Retrieve an inference status by ObjectID or raise HTTP 404 if not found.

    :param: inference_id: The ObjectID of the inference status to retrieve.
    :param: db: The MongoDB database connection object, injected by FastAPI.

    :Return: The found InferenceStatus object.
    """
    if not ObjectId.is_valid(inference_id):
        raise HTTPException(status_code=400, detail="Invalid ObjectID format")
    inference_status = await db.inference_status.find_one(
        {"_id": ObjectId(inference_id)}
    )

    if inference_status is None:
        raise HTTPException(status_code=404, detail="Inference status not found")

    return InferenceDBModel(**inference_status)


async def check_password(
        password: str,
        inference_status: InferenceDBModel,
) -> bool:
    """

    :param password:
    :param inference_status:
    :return:
    """
    if password is None and inference_status.password == '':
        return True
    if password is None and inference_status.password is not None:
        raise HTTPException(status_code=401, detail="Password required")
    if inference_status.password != password:
        raise HTTPException(status_code=401, detail="Invalid password")
    return True


@app.get(
    "/status/{inference_id}",
    response_description="Retrieve an inference status by ObjectID.",
    response_model=InferenceResponseModel,
    response_model_by_alias=False,
)
async def get_inference_status(
        password: str = None,
        inference_status: InferenceDBModel = Depends(get_inference_status_or_404)
) -> InferenceResponseModel:
    """
    Retrieve an inference status by ObjectID.

    :param password:
    :param inference_status:
    """
    new_status = inference_status
    password_val = await check_password(password, new_status)
    if password_val:
        return InferenceResponseModel(**new_status.model_dump(by_alias=True, exclude={"password"}))
    else:
        raise HTTPException(status_code=401, detail="Invalid password")


@app.post(
    "/run_inference",
    response_description="Start a new inference job.",
    response_model=InferenceResponseModel,
    response_model_by_alias=False,
    status_code=status.HTTP_201_CREATED,
)
async def create_inference_status(
        background_tasks: BackgroundTasks,
        inference_parameters: InferenceRequestModel = Body(...),
        db: AsyncIOMotorDatabase = Depends(mongo_database),
) -> dict[str, str]:
    """
    Create a new inference status.

    :param: inference_parameters: The inference status to create.
    :param: db: The MongoDB database connection object, injected by FastAPI.
    :param: background_tasks: The background tasks object, injected by FastAPI.

    :return: The created InferenceStatus object.
    """
    result = await db.inference_status.insert_one(
        inference_parameters.model_dump(exclude={"github_access_token"})
    )
    created_inference_status = await db.inference_status.find_one({"_id": result.inserted_id})

    if created_inference_status:
        created_inference_status = InferenceDBModel(
            id=PyObjectId(str(created_inference_status["_id"])),
            **created_inference_status
        )

        # Start the inference job
        background_tasks.add_task(
            inference_task,
            github_access_token=inference_parameters.github_access_token,
            created_status=created_inference_status,
            db=db,
        )
        return {
            "id": str(created_inference_status.id),
            "state": created_inference_status.state,
            "repo_name": created_inference_status.repo_name,
            "repo_folder": created_inference_status.repo_folder,
            "directory": created_inference_status.directory,
            "in_path": created_inference_status.in_path,
            "out_path": created_inference_status.out_path,
            "password": created_inference_status.password,
            "created_at": created_inference_status.created_at,
            "trocr_model": created_inference_status.trocr_model,
            "trocr_processor": created_inference_status.trocr_processor,
            "output_txt": created_inference_status.output_txt,
            "output_xml": created_inference_status.output_xml,
            "target_image_size": created_inference_status.target_image_size
        }
    else:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/status",
    response_description="Retrieve all inference statuses.",
    response_model=list[InferenceResponseModel],
    response_model_by_alias=False,
)
async def get_all_inference_statuses(
        db: AsyncIOMotorDatabase = Depends(mongo_database),
) -> list:
    """
    Retrieve all inference statuses.

    :param: db: The MongoDB database connection object, injected by FastAPI.

    :return: A list of all found InferenceStatus objects without password.
    """
    query = {"password": {"$eq": None, "$exists": True}}
    inference_statuses = await db.inference_status.find(query).to_list(length=None)
    query2 = {"password": {"$eq": "", "$exists": True}}
    inference_statuses2 = await db.inference_status.find(query2).to_list(length=None)

    inference_statuses.extend(inference_statuses2)

    responses = [
        InferenceResponseModel(
            id=PyObjectId(inference_status["_id"]),
            **inference_status
        ).model_dump(by_alias=True)
        for inference_status in inference_statuses
    ]
    return responses


@app.delete(
    "/status/{inference_id}",
    response_description="Delete an inference status by ObjectID.",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_inference_status(
        password: str = None,
        inference_status: InferenceDBModel = Depends(get_inference_status_or_404),
        db: AsyncIOMotorDatabase = Depends(mongo_database),
) -> None:
    """
    Delete a inference status by ObjectID.

    Args:
        db (AsyncIOMotorDatabase): The MongoDB database connection object, injected by FastAPI.
        :param password:
        :param db:
        :param inference_status:
    """
    password_val = await check_password(password, inference_status)

    if password_val:
        folder_path_out = os.path.join(inference_status.directory,
                                       inference_status.out_path,
                                       str(inference_status.id)
                                       )
        folder_path_in = os.path.join(inference_status.directory,
                                      inference_status.in_path,
                                      str(inference_status.id)
                                      )
        if os.path.exists(folder_path_out):
            shutil.rmtree(folder_path_out)
        if os.path.exists(folder_path_in):
            shutil.rmtree(folder_path_in)

        if os.path.exists('logs'):
            for file in os.listdir('logs'):
                if file.startswith(str(inference_status.id)):
                    os.remove(os.path.join('logs', file))

        await db.inference_status.delete_one({"_id": inference_status.id})
    else:
        raise HTTPException(status_code=401, detail="Invalid password")


@app.get(
    "/files/{inference_id}",
    response_description="Retrieve all files of a inference job.",
)
async def get_inference_files(
        password: str = None,
        inference_status: InferenceDBModel = Depends(get_inference_status_or_404),
) -> StreamingResponse:
    """
    Retrieve all files of an inference job.

    :param password:
    :param inference_status:
    :return: A dictionary containing the file names and their content.
    """
    password_val = await check_password(password, inference_status)

    if password_val:
        if not os.path.exists(inference_status.out_path):
            raise HTTPException(status_code=404, detail="Files not found")

        zip_bytes_io = io.BytesIO()
        with zipfile.ZipFile(zip_bytes_io, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for root, _, files in os.walk(inference_status.out_path):
                for filename in files:
                    full_file_path = os.path.join(root, filename)
                    zipped.write(full_file_path, filename)

        zip_bytes_io.seek(0)

        response = StreamingResponse(
            zip_bytes_io,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=inference_{str(inference_status.id)}.zip"}
        )
        return response
    else:
        raise HTTPException(status_code=401, detail="Invalid password")
