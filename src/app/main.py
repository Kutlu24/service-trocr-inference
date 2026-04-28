from contextlib import asynccontextmanager
import secrets
from typing import cast
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Body, Depends, HTTPException, status, Security, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from app.settings import Settings
from app.storage import get_redis_repository, RedisStatusRepository

from app.models.inference_models import (
    InferenceRequestModel,
    InferenceResponseModel,
)

from app.models.evaluation_models import (
    HFEvaluationRequestModel,
    EvaluationResponseModel,
)

from app.models.raw_xml_upload_models import (
    RawXMLMergeRequestModel,
    RawXMLMergeResponseModel,
)

from app.tasks.inference_task import inference_task
from app.tasks.evaluation_task import evaluation_task
from app.tasks.xml_merge_task import xml_merge_task


settings = Settings()

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def rate_limit_handler(
    _request: Request,
    _exc: RateLimitExceeded,
):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
    )

async def check_api_key(api_key: str | None = Security(api_key_header)):
    env_api_key = settings.API_KEY

    if env_api_key and (
        not api_key or not secrets.compare_digest(api_key, env_api_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    fastapi_app.state.repository = get_redis_repository(settings.REDIS_URL)
    yield
    fastapi_app.state.repository.close()


app = FastAPI(
    title="FLOW-Inference-Microservice",
    lifespan=lifespan,
)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per day", "4/minute"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)


def get_repository(request: Request) -> RedisStatusRepository:
    fastapi_app = cast(FastAPI, request.app)
    return getattr(fastapi_app.state, "repository")

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
    params: InferenceRequestModel = Body(...),
    repository: RedisStatusRepository = Depends(get_repository),
) -> InferenceResponseModel:
    huggingface_token = params.huggingface_token

    status_obj = InferenceResponseModel(
        **params.model_dump(
            by_alias=True,
            exclude={"huggingface_token"},
        )
    )

    repository.save(status_obj)

    inference_task.apply_async(
        kwargs={
            "status_dict": status_obj.model_dump(
                by_alias=True,
                mode="json",
            ),
            "huggingface_token": huggingface_token,
        },
        task_id=status_obj.request_id,
    )

    return status_obj


@app.get(
    "/inference/status",
    response_model=list[InferenceResponseModel],
    dependencies=[Depends(check_api_key)],
)
async def get_inference_status(
    repository: RedisStatusRepository = Depends(get_repository),
) -> list[InferenceResponseModel]:
    data = repository.get_all("inference")
    if not data:
        raise HTTPException(404, "No inference jobs found")
    return cast(list[InferenceResponseModel], data)


@app.get(
    "/inference/status/{uuid}",
    response_model=InferenceResponseModel,
    dependencies=[Depends(check_api_key)],
)
async def get_inference_status_by_id(
    uuid: str,
    repository: RedisStatusRepository = Depends(get_repository),
) -> InferenceResponseModel:
    status_obj = repository.get_by_id(uuid)

    if not status_obj or status_obj.job_type != "inference":
        raise HTTPException(404, "Inference job not found")

    return cast(InferenceResponseModel, status_obj)

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
    params: HFEvaluationRequestModel = Body(...),
    repository: RedisStatusRepository = Depends(get_repository),
) -> EvaluationResponseModel:
    status_obj = EvaluationResponseModel(
        source=params.huggingface_repo_name,
    )

    repository.save(status_obj)

    evaluation_task.apply_async(
        kwargs={
            "status_dict": status_obj.model_dump(
                by_alias=True,
                mode="json",
            ),
            "huggingface_token": params.huggingface_token,
            "splits": params.splits,
        },
        task_id=status_obj.request_id,
    )

    return status_obj


@app.get(
    "/evaluation/status",
    response_model=list[EvaluationResponseModel],
    dependencies=[Depends(check_api_key)],
)
async def get_all_evaluation_statuses(
    repository: RedisStatusRepository = Depends(get_repository),
) -> list[EvaluationResponseModel]:
    data = repository.get_all("evaluation")
    if not data:
        raise HTTPException(404, "No evaluation jobs found")
    return cast(list[EvaluationResponseModel], data)


@app.get(
    "/evaluation/status/{uuid}",
    response_model=EvaluationResponseModel,
    dependencies=[Depends(check_api_key)],
)
async def get_evaluation_status(
    uuid: str,
    repository: RedisStatusRepository = Depends(get_repository),
) -> EvaluationResponseModel:
    status_obj = repository.get_by_id(uuid)

    if not status_obj or status_obj.job_type != "evaluation":
        raise HTTPException(404, "Evaluation job not found")

    return cast(EvaluationResponseModel, status_obj)

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
    params: RawXMLMergeRequestModel = Body(...),
    repository: RedisStatusRepository = Depends(get_repository),
) -> RawXMLMergeResponseModel:
    status_obj = RawXMLMergeResponseModel(
        source=params.raw_xml_repo,
        source_raw_repo=params.raw_xml_repo,
        source_inference_repo=params.inference_repo,
        output_repo_name=params.output_repo_name or params.raw_xml_repo,
    )

    repository.save(status_obj)

    xml_merge_task.apply_async(
        kwargs={
            "status_dict": status_obj.model_dump(
                by_alias=True,
                mode="json",
            ),
            "huggingface_token": params.huggingface_token,
        },
        task_id=status_obj.request_id,
    )

    return status_obj


@app.get(
    "/xml-merge/status",
    response_model=list[RawXMLMergeResponseModel],
    dependencies=[Depends(check_api_key)],
)
async def get_all_xml_merge_statuses(
    repository: RedisStatusRepository = Depends(get_repository),
) -> list[RawXMLMergeResponseModel]:
    data = repository.get_all("xml_merge")
    if not data:
        raise HTTPException(404, "No xml merge jobs found")
    return cast(list[RawXMLMergeResponseModel], data)


@app.get(
    "/xml-merge/status/{uuid}",
    response_model=RawXMLMergeResponseModel,
    dependencies=[Depends(check_api_key)],
)
async def get_xml_merge_status(
    uuid: str,
    repository: RedisStatusRepository = Depends(get_repository),
) -> RawXMLMergeResponseModel:
    status_obj = repository.get_by_id(uuid)

    if not status_obj or status_obj.job_type != "xml_merge":
        raise HTTPException(404, "XML merge job not found")

    return cast(RawXMLMergeResponseModel, status_obj)
