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
        description=(
            "Output Hugging Face dataset repository where the dataset with added "
            "inference_* columns will be uploaded. This is the target repository. "
            "If omitted, the input repository is used as the output target, which "
            "requires allow_source_repo_update=True."
        ),
        json_schema_extra={"example": "my-org/target-dataset-for-inference-output"},
    )]

    private_repo: Annotated[bool, Field(
        default=True,
        alias="private_repo",
        description="Whether the output Hugging Face repository should be private.",
    )]

    trocr_model: Annotated[str, Field(
        default="microsoft/trocr-small-handwritten",
        alias="trocr_model",
        description="Hugging Face model id used for text-line inference.",
        json_schema_extra={"example": "microsoft/trocr-small-handwritten"},
    )]

    target_image_size: Annotated[Tuple[PositiveInt, PositiveInt] | None, Field(
        default=None,
        alias="target_image_size",
        description=(
            "Optional image resize target as [width, height]. "
            "If omitted or null, the image handler/model processor default behavior is used."
        ),
        json_schema_extra={"example": [384, 384]},
    )]

    stop_on_fail: Annotated[bool, Field(
        default=False,
        alias="stop_on_fail",
        description="Whether inference should stop on the first failed record.",
    )]

    splits: Annotated[List[str] | None, Field(
        default=None,
        alias="splits",
        description=(
            "Dataset splits to process. If omitted, the inference package default is used. "
            "Recommended value: ['train']."
        ),
        json_schema_extra={"example": ["train"]},
    )]

    push_to_hub: Annotated[bool, Field(
        default=True,
        alias="push_to_hub",
        description="Whether to upload the updated dataset back to Hugging Face Hub.",
    )]

    upload_mode: Annotated[Literal["new_repo", "replace", "update"], Field(
        default="new_repo",
        alias="upload_mode",
        description=(
            "Upload behavior for the output repository. "
            "'new_repo' creates a new repo, 'replace' overwrites existing data, "
            "and 'update' updates existing data."
        ),
        json_schema_extra={"example": "new_repo"},
    )]

    allow_source_repo_update: Annotated[bool, Field(
        default=False,
        alias="allow_source_repo_update",
        description=(
            "Safety flag allowing the source repository itself to be updated. "
            "Defaults to False to avoid accidentally modifying the input repo."
        ),
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
        description=(
            "Optional Hugging Face token. Required for private repositories "
            "or when pushing results to the Hub."
        ),
        json_schema_extra={"example": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
    )]

    huggingface_repo_name: Annotated[str, Field(
        alias="source_repo_name",
        serialization_alias="source",
        description=(
            "Input/source Hugging Face dataset repository to download and run inference on."
        ),
        json_schema_extra={"example": "my-org/my-source-dataset"},
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
        description="Number of rows in the returned inference output dataframes.",
        json_schema_extra={"example": 1200},
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )