from typing import Annotated, Literal
from app.models.shared_models import JobStatusBase

from pydantic import BaseModel, Field, ConfigDict

# -----------------------------
# Request
# -----------------------------

class RawXMLMergeRequestModel(BaseModel):
    raw_xml_repo: Annotated[str, Field(
        alias="raw_xml_repo",
        description="Hugging Face dataset containing raw XML files.",
    )]

    inference_repo: Annotated[str, Field(
        alias="inference_repo",
        description="Dataset containing inference results.",
    )]

    huggingface_token: Annotated[str | None, Field(
        default=None,
        alias="huggingface_token",
    )]

    output_repo_name: Annotated[str | None, Field(
        default=None,
        alias="output_repo_name",
        description="If None, raw_xml_repo is overwritten.",
    )]

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


# -----------------------------
# Status
# -----------------------------

class RawXMLMergeResponseModel(JobStatusBase):
    """
    Persistent status model for Raw XML merge jobs.
    Stored in DB, exported to JSON, updated by workers.
    """

    job_type: Literal["xml_merge"] = "xml_merge"

    # We store BOTH datasets, but choose one as primary `source`
    source_raw_repo: Annotated[str, Field(
        alias="source_raw_repo"
    )]

    source_inference_repo: Annotated[str, Field(
        alias="source_inference_repo"
    )]

    output_repo_name: Annotated[str, Field(
        alias="output_repo_name"
    )]

    updated_records: Annotated[int, Field(
        default=0,
        alias="updated_records"
    )]

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)
