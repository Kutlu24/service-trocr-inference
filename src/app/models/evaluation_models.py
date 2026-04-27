from typing import Annotated, List, Literal
from app.models.shared_models import JobStatusBase
from pydantic import BaseModel, Field, ConfigDict

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
