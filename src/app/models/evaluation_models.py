from datetime import datetime
from enum import Enum
from typing import Annotated, List, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


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
        description="Primary dataset being processed"
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

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


# -----------------------------
# Evaluation request
# -----------------------------

class EvaluationRequestBase(BaseModel):
    huggingface_repo_name: Annotated[str, Field(
        alias="huggingface_repo_name",
        serialization_alias="source",
        description="Dataset to evaluate. Must contain inference_* column."
    )]

    huggingface_token: Annotated[str | None, Field(
        default=None,
        alias="huggingface_token",
    )]

    splits: Annotated[List[str] | None, Field(
        default=None,
        alias="splits",
    )]

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


class HFEvaluationRequestModel(EvaluationRequestBase):
    """Evaluate and upload groundtruth.txt, hypothesis.txt and report.json to HuggingFace."""
    pass


# -----------------------------
# Evaluation status
# -----------------------------

class EvaluationResponseModel(JobStatusBase):
    """
    Persistent status model for evaluation jobs.
    Stored in DB, exported to JSON, updated by workers.
    """

    job_type: Literal["evaluation"] = "evaluation"

    cer: Annotated[float | None, Field(
        default=None,
        alias="cer"
    )]

    gt_lines: Annotated[int | None, Field(
        default=None,
        alias="gt_lines"
    )]

    hypothesis_lines: Annotated[int | None, Field(
        default=None,
        alias="hypothesis_lines"
    )]

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)
