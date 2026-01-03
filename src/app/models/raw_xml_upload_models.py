from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

class StateEnum(str, Enum):
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    DONE = "completed"

class RawXMLMergeRequestModel(BaseModel):
    """
    Request model for merging inference results into raw XML datasets.
    """

    raw_xml_repo: Annotated[str, Field(
        alias="raw_xml_repo",
        description="Hugging Face dataset repository containing the raw XML files.",
        title="Raw-XML-Repo",
        examples=["username/raw-xml-dataset"],
    )]

    inference_repo: Annotated[str, Field(
        alias="inference_repo",
        description="Hugging Face dataset repository that contains inference columns.",
        title="Inference-Repo",
        examples=["username/inference-dataset"],
    )]

    huggingface_token: Annotated[str, Field(
        alias="huggingface_token",
        description="Hugging Face token required for fetching and uploading datasets.",
        title="HuggingFace-Token",
        examples=["hf_1234567890"],
    )]

    output_repo_name: Annotated[str | None, Field(
        default=None,
        alias="output_repo_name",
        description="If provided, upload the updated XML dataset here. "
                    "If None, defaults to raw_xml_repo.",
        title="Output-Repo-Name",
        examples=["username/updated-xml-dataset"],
    )]

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

class RawXMLMergeResponseModel(BaseModel):
    """
    Response model describing the state and outcome
    of a raw XML + inference merge operation.
    """

    request_id: Annotated[str, Field(
        default_factory=lambda: str(uuid4()),
        alias="request_id",
        description="Unique identifier for this merge request.",
        title="Request-ID",
    )]

    source_raw_repo: Annotated[str, Field(
        alias="source_raw_repo",
        description="Raw XML dataset used as input.",
        title="Source-Raw-XML",
    )]

    source_inference_repo: Annotated[str, Field(
        alias="source_inference_repo",
        description="Inference dataset used to update XML.",
        title="Source-Inference",
    )]

    output_repo_name: Annotated[str, Field(
        alias="output_repo_name",
        description="Repository to which the updated dataset was uploaded.",
        title="Output-Repo",
    )]

    created_at: Annotated[datetime, Field(
        default_factory=datetime.now,
        alias="created_at",
        title="Created-At",
    )]

    runtime_seconds: Annotated[float, Field(
        default=0.0,
        alias="runtime",
        title="Runtime",
    )]

    state: Annotated[StateEnum, Field(
        default=StateEnum.IN_PROGRESS,
        alias="state",
        title="State",
        description="Current state of the XML merge process.",
    )]

    updated_records: Annotated[int, Field(
        default=0,
        alias="updated_records",
        title="Updated-Records",
        description="Number of XML rows updated with inferred text.",
    )]

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)
