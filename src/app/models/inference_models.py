from typing import Annotated, List, Tuple, Literal
from app.models.shared_models import JobStatusBase

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    PositiveInt,
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