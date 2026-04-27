from datetime import datetime, UTC
from enum import Enum
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class StateEnum(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    COMPLETED = "completed"


class JobStatusBase(BaseModel):
    request_id: Annotated[str, Field(
        default_factory=lambda: str(uuid4()),
        alias="request_id",
    )]

    job_type: Annotated[str, Field(
        alias="job_type",
        description="Which pipeline this job belongs to.",
    )]

    source: Annotated[str, Field(
        alias="source",
        description="Primary dataset or repository this job runs on.",
    )]

    created_at: Annotated[datetime, Field(
        default_factory=lambda: datetime.now(UTC),
        alias="created_at",
    )]

    started_at: Annotated[datetime | None, Field(
        default=None,
        alias="started_at",
    )]

    ended_at: Annotated[datetime | None, Field(
        default=None,
        alias="ended_at",
    )]

    runtime_seconds: Annotated[float, Field(
        default=0.0,
        alias="runtime",
    )]

    state: Annotated[StateEnum, Field(
        default=StateEnum.PENDING,
        alias="state",
    )]

    retry_count: Annotated[int, Field(
        default=0,
        alias="retry_count",
    )]

    error_message: Annotated[str | None, Field(
        default=None,
        alias="error_message",
    )]

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
    )