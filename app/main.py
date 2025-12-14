"""
Main script FLOW Inference Service
"""
import json
import logging
import os
from typing import Any, List
from contextlib import asynccontextmanager

import aiofiles
from fastapi import (
    FastAPI,
    Body,
    Depends,
    HTTPException,
    status,
    Security,
    BackgroundTasks
)
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.models.evaluation_models import (
    HFEvaluationRequestModel,
    EvaluationResponseModel
)

from app.models.inference_models import (
    InferenceRequestModel,
    InferenceResponseModel
)

from app.models.raw_xml_upload_models import (
    RawXMLMergeRequestModel,
    RawXMLMergeResponseModel
)

from app.tasks import inference_task
from app.tasks import evaluation_task
from app.tasks import xml_merge_task

logging.getLogger("fastapi").setLevel(logging.DEBUG)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def check_api_key(api_key: str | None = Security(api_key_header)) -> None:
    env_api_key = os.getenv("API_KEY")
    if env_api_key and api_key != env_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

INFERENCE_STATUS_FILE = os.getenv("INFERENCE_STATUS_FILE", "inference-status.json")
EVALUATION_STATUS_FILE = os.getenv("EVALUATION_STATUS_FILE", "evaluation-status.json")
XML_MERGE_STATUS_FILE = os.getenv("XML_MERGE_STATUS_FILE", "xml-merge-status.json")

async def load_json(file_path: str, model_cls):
    if os.path.exists(file_path):
        async with aiofiles.open(file_path, encoding="utf-8") as file:
            try:
                data = json.loads(await file.read())
                return [model_cls(**item) for item in data]
            except json.JSONDecodeError:
                return []
    return []

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield

app = FastAPI(title="FLOW-Inference-Microservice", lifespan=lifespan)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per day", "4/minute"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.post(
    "/inference/start",
    response_model=InferenceResponseModel,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_api_key)],
)
def start_inference(
    background_tasks: BackgroundTasks,
    params: InferenceRequestModel = Body(...)
) -> Any:

    status_obj = InferenceResponseModel(
        **params.model_dump(by_alias=True, exclude={"huggingface_token"})
    )

    background_tasks.add_task(
        inference_task,
        huggingface_token=params.huggingface_token,
        created_status=status_obj,
    )

    return {"message": "Inference job started", **status_obj.dict()}

@app.get("/inference/status", response_model=List[InferenceResponseModel])
async def get_inference_status():
    data = await load_json(INFERENCE_STATUS_FILE, InferenceResponseModel)
    if not data:
        raise HTTPException(404, "No inference jobs found")
    return data

@app.get("/inference/status/{uuid}", response_model=InferenceResponseModel)
async def get_inference_status_by_id(uuid: str):
    data = await load_json(INFERENCE_STATUS_FILE, InferenceResponseModel)
    for item in data:
        if item.request_id == uuid:
            return item
    raise HTTPException(404, "Inference job not found")

@app.post(
    "/evaluation/hf/start",
    response_model=EvaluationResponseModel,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_api_key)],
)
def start_hf_evaluation(
    background_tasks: BackgroundTasks,
    params: HFEvaluationRequestModel = Body(...)
):
    status_obj = EvaluationResponseModel(
        **params.model_dump(by_alias=True, exclude={"huggingface_token"})
    )

    background_tasks.add_task(
        evaluation_task,
        huggingface_token=params.huggingface_token,
        created_status=status_obj,
        mode="hf",  # 👈 explicit
    )

    return {"message": "HF evaluation job started", **status_obj.dict()}

@app.get(
    "/evaluation/status",
    response_description="Retrieve all evaluation statuses.",
    response_model=List[EvaluationResponseModel],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_api_key)],
)
async def get_all_evaluation_statuses_or_404() -> List[EvaluationResponseModel]:
    """
    Retrieve all evaluation job statuses.
    """
    data = await load_json(EVALUATION_STATUS_FILE, EvaluationResponseModel)
    if len(data) == 0:
        raise HTTPException(status_code=404, detail="No evaluation jobs found")
    return data

@app.get(
    "/evaluation/status/{uuid}",
    response_description="Retrieve an evaluation status by UUID.",
    response_model=EvaluationResponseModel,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_api_key)],
)
async def get_evaluation_status_or_404(uuid: str) -> EvaluationResponseModel:
    """
    Retrieve a single evaluation job status by its UUID.
    """
    data = await load_json(EVALUATION_STATUS_FILE, EvaluationResponseModel)
    for item in data:
        if item.request_id == uuid:
            return item
    raise HTTPException(status_code=404, detail="Evaluation job not found")

@app.post(
    "/xml-merge/start",
    response_model=RawXMLMergeResponseModel,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_api_key)],
)
def start_xml_merge(
    background_tasks: BackgroundTasks,
    params: RawXMLMergeRequestModel = Body(...)
) -> Any:

    status_obj = RawXMLMergeResponseModel(
        **params.model_dump(by_alias=True, exclude={"huggingface_token"})
    )

    background_tasks.add_task(
        xml_merge_task,
        huggingface_token=params.huggingface_token,
        created_status=status_obj,
    )

    return {"message": "XML merge job started", **status_obj.dict()}

@app.get(
    "/xml-merge/status",
    response_description="Retrieve all XML merge statuses.",
    response_model=List[RawXMLMergeResponseModel],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_api_key)],
)
async def get_all_xml_merge_statuses_or_404() -> List[RawXMLMergeResponseModel]:
    """
    Retrieve all XML merge job statuses.
    """
    data = await load_json(XML_MERGE_STATUS_FILE, RawXMLMergeResponseModel)
    if len(data) == 0:
        raise HTTPException(status_code=404, detail="No XML merge jobs found")
    return data

@app.get(
    "/xml-merge/status/{uuid}",
    response_description="Retrieve an XML merge status by UUID.",
    response_model=RawXMLMergeResponseModel,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_api_key)],
)
async def get_xml_merge_status_or_404(uuid: str) -> RawXMLMergeResponseModel:
    """
    Retrieve a single XML merge job status by its UUID.
    """
    data = await load_json(XML_MERGE_STATUS_FILE, RawXMLMergeResponseModel)
    for item in data:
        if item.request_id == uuid:
            return item
    raise HTTPException(status_code=404, detail="XML merge job not found")
