"""
Main script FLOW Inference Service
"""
from pathlib import Path
import logging
import os
from typing import Any, List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    Body,
    Depends,
    HTTPException,
    status,
    Security,
    BackgroundTasks,
    Request
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

from app.storage import create_repository, StatusRepository
from app.tasks.inference_task import inference_task
from app.tasks.evaluation_task import evaluation_task
from app.tasks.xml_merge_task import xml_merge_task

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.getLogger("fastapi").setLevel(logging.DEBUG)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------
# Security
# ---------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def check_api_key(api_key: str | None = Security(api_key_header)) -> None:
    env_api_key = os.getenv("API_KEY")
    if env_api_key and api_key != env_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# ---------------------------------------------------------
# Status export locations
# ---------------------------------------------------------

INFERENCE_STATUS_JSON = Path(os.getenv("INFERENCE_STATUS_FILE", "inference-status.json"))
EVALUATION_STATUS_JSON = Path(os.getenv("EVALUATION_STATUS_FILE", "evaluation-status.json"))
XML_MERGE_STATUS_JSON = Path(os.getenv("XML_MERGE_STATUS_FILE", "xml-merge-status.json"))

async def export_all(repo: StatusRepository):
    await repo.export_to_json(INFERENCE_STATUS_JSON, job_type="inference")
    await repo.export_to_json(EVALUATION_STATUS_JSON, job_type="evaluation")
    await repo.export_to_json(XML_MERGE_STATUS_JSON, job_type="xml_merge")

# ---------------------------------------------------------
# Repository lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    storage_type = os.getenv("STATUS_STORAGE", "sqlite")
    path = os.getenv("STATUS_PATH")

    repo = create_repository(storage_type, Path(path) if path else None)
    app.state.repository = repo

    # Export once at startup
    await export_all(repo)

    try:
        yield
    finally:
        await export_all(repo)
        await repo.close()

def get_repository(request: Request) -> StatusRepository:
    return request.app.state.repository

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(title="FLOW-Inference-Microservice", lifespan=lifespan)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per day", "4/minute"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ---------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------

@app.post(
    "/inference/start",
    response_model=InferenceResponseModel,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_api_key)],
)
async def start_inference(
    background_tasks: BackgroundTasks,
    params: InferenceRequestModel = Body(...),
    repository: StatusRepository = Depends(get_repository),
) -> Any:
    status_obj = InferenceResponseModel(
        **params.model_dump(by_alias=True, exclude={"huggingface_token"})
    )

    await repository.save(status_obj)
    await export_all(repository)

    background_tasks.add_task(
        inference_task,
        repository=repository,
        huggingface_token=params.huggingface_token,
        created_status=status_obj,
    )

    return {"message": "Inference job started", **status_obj.model_dump(by_alias=True)}

@app.get("/inference/status", response_model=List[InferenceResponseModel])
async def get_inference_status(repository: StatusRepository = Depends(get_repository)):
    data = await repository.get_all("inference")
    if not data:
        raise HTTPException(404, "No inference jobs found")
    return data

@app.get("/inference/status/{uuid}", response_model=InferenceResponseModel)
async def get_inference_status_by_id(uuid: str, repository: StatusRepository = Depends(get_repository)):
    status = await repository.get_by_id(uuid)
    if not status or status.job_type != "inference":
        raise HTTPException(404, "Inference job not found")
    return status

# ---------------------------------------------------------
# Evaluation endpoints
# ---------------------------------------------------------

@app.post(
    "/evaluation/hf/start",
    response_model=EvaluationResponseModel,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_api_key)],
)
async def start_hf_evaluation(
    background_tasks: BackgroundTasks,
    params: HFEvaluationRequestModel = Body(...),
    repository: StatusRepository = Depends(get_repository),
):
    status_obj = EvaluationResponseModel(source=params.huggingface_repo_name)

    await repository.save(status_obj)
    await export_all(repository)

    background_tasks.add_task(
        evaluation_task,
        repository=repository,
        huggingface_token=params.huggingface_token,
        created_status=status_obj,
    )

    return {"message": "HF evaluation job started", **status_obj.model_dump(by_alias=True)}

@app.get("/evaluation/status", response_model=List[EvaluationResponseModel])
async def get_all_evaluation_statuses(repository: StatusRepository = Depends(get_repository)):
    data = await repository.get_all("evaluation")
    if not data:
        raise HTTPException(404, "No evaluation jobs found")
    return data

@app.get("/evaluation/status/{uuid}", response_model=EvaluationResponseModel)
async def get_evaluation_status(uuid: str, repository: StatusRepository = Depends(get_repository)):
    status = await repository.get_by_id(uuid)
    if not status or status.job_type != "evaluation":
        raise HTTPException(404, "Evaluation job not found")
    return status

# ---------------------------------------------------------
# XML merge endpoints
# ---------------------------------------------------------

@app.post(
    "/xml-merge/start",
    response_model=RawXMLMergeResponseModel,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_api_key)],
)
async def start_xml_merge(
    background_tasks: BackgroundTasks,
    params: RawXMLMergeRequestModel = Body(...),
    repository: StatusRepository = Depends(get_repository),
):
    status_obj = RawXMLMergeResponseModel(
        source_raw_repo=params.raw_xml_repo,
        source_inference_repo=params.inference_repo,
        output_repo_name=params.output_repo_name or params.raw_xml_repo,
    )

    await repository.save(status_obj)
    await export_all(repository)

    background_tasks.add_task(
        xml_merge_task,
        repository=repository,
        huggingface_token=params.huggingface_token,
        created_status=status_obj,
    )

    return {"message": "XML merge job started", **status_obj.model_dump(by_alias=True)}

@app.get("/xml-merge/status", response_model=List[RawXMLMergeResponseModel])
async def get_all_xml_merge_statuses(repository: StatusRepository = Depends(get_repository)):
    data = await repository.get_all("xml_merge")
    if not data:
        raise HTTPException(404, "No xml merge jobs found")
    return data

@app.get("/xml-merge/status/{uuid}", response_model=RawXMLMergeResponseModel)
async def get_xml_merge_status(uuid: str, repository: StatusRepository = Depends(get_repository)):
    status = await repository.get_by_id(uuid)
    if not status or status.job_type != "xml_merge":
        raise HTTPException(404, "XML merge job not found")
    return status
