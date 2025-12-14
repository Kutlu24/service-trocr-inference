from datetime import datetime
from enum import Enum
from typing import Annotated, List, Tuple
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    PositiveInt
)


class StateEnum(str, Enum):
    """
    Enum class for the state of the inference process.
    """
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    DONE = "completed"


class InferenceBaseModel(BaseModel):
    """
    Base model for inference requests and responses.
    Contains all user-configurable options.
    """
    upload_repo_name: Annotated[str | None, Field(
        default=None,
        alias="upload_repo_name",
        description="Name of the Hugging Face repository to upload the dataset with inference results to.",
        title="Upload-Repo-Name",
        examples=["username/my-dataset-with-inference"],
    )]

    private_repo: Annotated[bool, Field(
        default=True,
        alias="private_repo",
        description="Whether the upload repository should be private (if push_to_hub=True).",
        title="Private-Repo",
    )]

    trocr_model: Annotated[str | None, Field(
        default="microsoft/trocr-small-handwritten",
        alias="trocr_model",
        description="Name of the Hugging Face TrOCR model to use for inference.",
        title="TrOCR-Model",
        examples=["microsoft/trocr-small-handwritten"],
    )]

    target_image_size: Annotated[Tuple[PositiveInt, PositiveInt] | None, Field(
        default=None,
        alias="target_image_size",
        description="Target (width, height) to resize images to before inference.",
        title="Target-Image-Size",
        examples=[(384, 384)],
    )]

    stop_on_fail: Annotated[bool | None, Field(
        default=False,
        alias="stop_on_fail",
        description="Whether to stop processing when an error occurs during inference.",
        title="Stop-On-Fail",
    )]

    splits: Annotated[List[str] | None, Field(
        default=None,
        alias="splits",
        description="List of dataset splits to run inference on. "
                    "If not provided, defaults to ['train'].",
        title="Splits",
        examples=[["train", "validation"]],
    )]

    push_to_hub: Annotated[bool | None, Field(
        default=True,
        alias="push_to_hub",
        description="Whether to push the updated dataset with inference results to the Hugging Face Hub.",
        title="Push-To-Hub",
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )


class InferenceRequestModel(InferenceBaseModel):
    """
    Request model for starting an inference job on a Hugging Face dataset.
    """
    huggingface_token: Annotated[str | None, Field(
        default=None,
        alias="huggingface_token",
        description="Hugging Face token to authenticate with the Hugging Face Hub. "
                    "Required if the dataset is private or if uploading results.",
        title="HuggingFace-Token",
        examples=["hf_1234567890"],
    )]

    huggingface_repo_name: Annotated[str, Field(
        alias="huggingface_repo_name",
        description="Name of the Hugging Face dataset repository to run inference on.",
        title="HuggingFace-Repo-Name",
        serialization_alias="source",
        examples=["username/my-dataset"],
    )]


class InferenceResponseModel(InferenceBaseModel):
    """
    Response model representing the status of an inference job.
    """
    request_id: Annotated[str, Field(
        default_factory=lambda: str(uuid4()),
        alias="request_id",
        description="Unique identifier for the inference request.",
        title="Request-ID",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )]

    source: Annotated[str, Field(
        alias="source",
        description="Source Hugging Face repository name on which inference is performed.",
        title="Source",
    )]

    created_at: Annotated[datetime, Field(
        default_factory=datetime.now,
        alias="created_at",
        description="Timestamp of the inference status creation.",
        title="Created-At",
    )]

    runtime_seconds: Annotated[float, Field(
        default=0.0,
        alias="runtime",
        description="Runtime of the inference job in seconds.",
        title="Runtime",
    )]

    state: Annotated[StateEnum, Field(
        default=StateEnum.IN_PROGRESS,
        alias="state",
        description="Current state of the inference status. "
                    "Can be 'in_progress', 'failed', or 'completed'.",
        title="State",
    )]

    total_records: Annotated[int, Field(
        default=0,
        alias="total_records",
        description="Total number of data records to infer.",
        title="Total-Records",
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )



