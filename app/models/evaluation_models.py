from datetime import datetime
from enum import Enum
from typing import Annotated, List
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    ConfigDict
)


class StateEnum(str, Enum):
    """
    Enum class for the state of the inference process.
    """
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    DONE = "completed"

class EvaluationRequestBase(BaseModel):
    """
    Base request model for evaluation jobs.
    Controls which dataset to evaluate and which splits to use.
    """

    huggingface_token: Annotated[str | None, Field(
        default=None,
        alias="huggingface_token",
        description="Required if dataset is private or HF upload is requested."
    )]

    huggingface_repo_name: Annotated[str, Field(
        alias="huggingface_repo_name",
        description="Dataset to evaluate. Must contain inference_* column."
    )]

    splits: Annotated[List[str] | None, Field(
        default=None,
        alias="splits",
        description="Splits to evaluate. If None, defaults to test > train."
    )]

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

class HFEvaluationRequestModel(EvaluationRequestBase):
    """Evaluate and upload groundtruth.txt, hypothesis.txt and report.json to HuggingFace."""
    pass

class EvaluationResponseModel(BaseModel):
    """
    Status model for evaluation jobs.
    Returned immediately, updated later.
    """

    request_id: Annotated[str, Field(
        default_factory=lambda: str(uuid4()),
        alias="request_id",
    )]

    source: Annotated[str, Field(
        alias="source",
        description="Dataset being evaluated."
    )]

    created_at: Annotated[datetime, Field(
        default_factory=datetime.now,
        alias="created_at"
    )]

    runtime_seconds: Annotated[float, Field(
        default=0.0,
        alias="runtime"
    )]

    state: Annotated[StateEnum, Field(
        default=StateEnum.IN_PROGRESS,
        alias="state"
    )]

    cer: Annotated[float | None, Field(
        default=None,
        alias="cer"
    )]

    gt_lines: Annotated[int | None, Field(
        default=None,
        alias="gt_lines"
    )]

    hypothesis_lines: Annotated[int | None, Field(
        default=None, alias="hypothesis_lines"
    )]

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)
