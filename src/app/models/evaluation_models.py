from typing import Annotated, List, Literal
from app.models.shared_models import JobStatusBase
from pydantic import BaseModel, Field, ConfigDict, SecretStr


# -----------------------------
# Evaluation request
# -----------------------------

class EvaluationRequestBase(BaseModel):
    huggingface_repo_name: Annotated[str, Field(
        alias="huggingface_repo_name",
        serialization_alias="source",
        description=(
            "Hugging Face dataset repository to evaluate. "
            "The dataset must contain a ground-truth text column and at least one "
            "inference_* prediction column."
        ),
        json_schema_extra={"example": "my-org/my-dataset-with-inference"},
    )]

    huggingface_token: Annotated[SecretStr | None, Field(
        default=None,
        alias="huggingface_token",
        description=(
            "Optional Hugging Face token. Required for private repositories "
            "or when uploading evaluation results."
        ),
        json_schema_extra={"example": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
    )]

    splits: Annotated[List[str] | None, Field(
        default=None,
        alias="splits",
        description=(
            "Dataset splits to evaluate. If omitted, the evaluation package default is used. "
            "Recommended value: ['test']."
        ),
        json_schema_extra={"example": ["test"]},
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class HFEvaluationRequestModel(EvaluationRequestBase):
    """
    Evaluate inference results against ground truth and upload evaluation artifacts.

    Produces files such as groundtruth.txt, hypothesis.txt, and report.json.
    """
    pass


# -----------------------------
# Evaluation status
# -----------------------------

class EvaluationResponseModel(JobStatusBase):
    """
    Persistent status model for evaluation jobs.
    Stored in DB, exported to JSON, returned by the API, and updated by workers.
    """

    job_type: Literal["evaluation"] = "evaluation"

    cer: Annotated[float | None, Field(
        default=None,
        alias="cer",
        description="Character Error Rate for the evaluated prediction column.",
        json_schema_extra={"example": 0.0821},
    )]

    gt_lines: Annotated[int | None, Field(
        default=None,
        alias="gt_lines",
        description="Number of ground-truth lines used for evaluation.",
        json_schema_extra={"example": 1200},
    )]

    hypothesis_lines: Annotated[int | None, Field(
        default=None,
        alias="hypothesis_lines",
        description="Number of predicted lines used for evaluation.",
        json_schema_extra={"example": 1200},
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
    )