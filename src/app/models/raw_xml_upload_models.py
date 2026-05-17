from typing import Annotated, Literal
from app.models.shared_models import JobStatusBase

from pydantic import BaseModel, Field, ConfigDict


UploadMode = Literal["new_repo", "replace", "update"]


# -----------------------------
# Request
# -----------------------------

class RawXMLMergeRequestModel(BaseModel):
    raw_xml_repo: Annotated[str, Field(
        alias="raw_xml_repo",
        description=(
            "Source Hugging Face dataset repository containing raw XML parquet files. "
            "This repository is read as the raw XML base."
        ),
        json_schema_extra={"example": "my-org/my-raw-xml-dataset"},
    )]

    inference_repo: Annotated[str, Field(
        alias="inference_repo",
        description=(
            "Source Hugging Face dataset repository containing inference results. "
            "The latest inference_* column is written back into the raw XML."
        ),
        json_schema_extra={"example": "my-org/my-dataset-with-inference"},
    )]

    huggingface_token: Annotated[str | None, Field(
        default=None,
        alias="huggingface_token",
        description=(
            "Optional Hugging Face token. Required for private repositories "
            "or when uploading the merged output dataset."
        ),
        json_schema_extra={"example": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
    )]

    output_repo_name: Annotated[str | None, Field(
        default=None,
        alias="output_repo_name",
        description=(
            "Target Hugging Face dataset repository for the merged raw XML output. "
            "If omitted, the raw_xml_repo is used as the target. Updating the source "
            "repo requires allow_source_repo_update=True."
        ),
        json_schema_extra={"example": "my-org/my-raw-xml-with-inference"},
    )]

    private_repo: Annotated[bool, Field(
        default=True,
        alias="private_repo",
        description="Whether a newly created output repository should be private.",
    )]

    upload_mode: Annotated[UploadMode, Field(
        default="new_repo",
        alias="upload_mode",
        description=(
            "Upload behavior for the output repository. "
            "'new_repo' creates a new repository and fails if it already exists. "
            "'update' preserves existing inference_xml_* columns in the target repo. "
            "'replace' replaces the target dataset contents."
        ),
        json_schema_extra={"example": "new_repo"},
    )]

    allow_source_repo_update: Annotated[bool, Field(
        default=False,
        alias="allow_source_repo_update",
        description=(
            "Safety flag allowing the raw_xml_repo itself to be used as the output target. "
            "Defaults to False to avoid accidentally modifying the source raw XML dataset."
        ),
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# -----------------------------
# Status
# -----------------------------

class RawXMLMergeResponseModel(JobStatusBase):
    """
    Persistent status model for raw XML merge jobs.
    Stored in DB, exported to JSON, returned by the API, and updated by workers.
    """

    job_type: Literal["xml_merge"] = "xml_merge"

    source_raw_repo: Annotated[str, Field(
        alias="source_raw_repo",
        description="Source repository containing the raw XML parquet files.",
        json_schema_extra={"example": "my-org/my-raw-xml-dataset"},
    )]

    source_inference_repo: Annotated[str, Field(
        alias="source_inference_repo",
        description="Source repository containing inference results.",
        json_schema_extra={"example": "my-org/my-dataset-with-inference"},
    )]

    output_repo_name: Annotated[str, Field(
        alias="output_repo_name",
        description="Target repository where merged raw XML output is uploaded.",
        json_schema_extra={"example": "my-org/my-raw-xml-with-inference"},
    )]

    private_repo: Annotated[bool, Field(
        default=True,
        alias="private_repo",
        description="Whether a newly created output repository should be private.",
    )]

    upload_mode: Annotated[UploadMode, Field(
        default="new_repo",
        alias="upload_mode",
        description=(
            "Upload behavior used for the output repository: "
            "'new_repo', 'update', or 'replace'."
        ),
        json_schema_extra={"example": "new_repo"},
    )]

    allow_source_repo_update: Annotated[bool, Field(
        default=False,
        alias="allow_source_repo_update",
        description=(
            "Whether this job is allowed to write back into the original raw XML source repo."
        ),
    )]

    updated_records: Annotated[int, Field(
        default=0,
        alias="updated_records",
        description="Number of raw XML records updated by the merge job.",
        json_schema_extra={"example": 1200},
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
    )