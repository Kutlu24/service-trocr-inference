from datetime import datetime
from enum import Enum
from typing import Annotated, List, Tuple, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    PositiveInt,
)


# -----------------------------
# Shared job state
# -----------------------------

class StateEnum(str, Enum):
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    DONE = "completed"


class JobStatusBase(BaseModel):
    request_id: Annotated[str, Field(
        default_factory=lambda: str(uuid4()),
        alias="request_id",
    )]

    job_type: Annotated[str, Field(
        alias="job_type",
        description="Which pipeline this job belongs to (inference, xml_merge, evaluation)"
    )]

    source: Annotated[str, Field(
        alias="source",
        description="Primary dataset or repository this job runs on"
    )]

    created_at: Annotated[datetime, Field(
        default_factory=datetime.now,
        alias="created_at",
    )]

    runtime_seconds: Annotated[float, Field(
        default=0.0,
        alias="runtime",
    )]

    state: Annotated[StateEnum, Field(
        default=StateEnum.IN_PROGRESS,
        alias="state",
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# -----------------------------
# Inference config (user input)
# -----------------------------

class InferenceBaseModel(BaseModel):
    upload_repo_name: Annotated[str | None, Field(
        default=None,
        alias="upload_repo_name",
    )]

    private_repo: Annotated[bool, Field(
        default=True,
        alias="private_repo",
    )]

    trocr_model: Annotated[str | None, Field(
        default="microsoft/trocr-small-handwritten",
        alias="trocr_model",
    )]

    target_image_size: Annotated[Tuple[PositiveInt, PositiveInt] | None, Field(
        default=None,
        alias="target_image_size",
    )]

    stop_on_fail: Annotated[bool | None, Field(
        default=False,
        alias="stop_on_fail",
    )]

    splits: Annotated[List[str] | None, Field(
        default=None,
        alias="splits",
    )]

    push_to_hub: Annotated[bool | None, Field(
        default=True,
        alias="push_to_hub",
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )


# -----------------------------
# Request
# -----------------------------

class InferenceRequestModel(InferenceBaseModel):
    huggingface_token: Annotated[str | None, Field(
        default=None,
        alias="huggingface_token",
    )]

    huggingface_repo_name: Annotated[str, Field(
        alias="huggingface_repo_name",
        serialization_alias="source",
    )]


# -----------------------------
# Status / Response
# -----------------------------

class InferenceResponseModel(JobStatusBase, InferenceBaseModel):
    """
    Persistent status model for inference jobs.
    Stored in DB, exported to JSON, returned by API.
    """

    job_type: Literal["inference"] = "inference"

    total_records: Annotated[int, Field(
        default=0,
        alias="total_records",
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )