from enum import Enum
from typing import Annotated
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentEnum(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class LogLevelEnum(str, Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    API_KEY: Annotated[str | None, Field(
        default=None,
        alias="api_key",
    )]

    ENVIRONMENT: Annotated[EnvironmentEnum, Field(
        default=EnvironmentEnum.DEVELOPMENT,
        alias="environment",
    )]

    REDIS_URL: Annotated[str, Field(
        default="redis://redis:6379/0",
        alias="redis_url",
    )]

    LOG_LEVEL: Annotated[LogLevelEnum, Field(
        default=LogLevelEnum.INFO,
        alias="log_level",
    )]

    LOG_TO_FILES: Annotated[bool, Field(
        default=False,
        alias="log_to_files",
    )]

    HTTPS_REDIRECT: Annotated[bool, Field(
        default=True,
        alias="https_redirect",
    )]

    CORS_ALLOWED_ORIGINS: Annotated[set[str], Field(
        default={"*"},
        alias="cors_allowed_origins",
    )]

    CORS_ALLOWED_HEADERS: Annotated[set[str], Field(
        default={"*"},
        alias="cors_allowed_headers",
    )]

    CORS_ALLOWED_METHODS: Annotated[set[str], Field(
        default={"GET", "POST", "OPTIONS"},
        alias="cors_allowed_methods",
    )]
    CELERY_TASK_TIME_LIMIT: Annotated[int, Field(
        default=3600,
        alias="celery_task_time_limit",
    )]
    CELERY_TASK_SOFT_TIME_LIMIT: Annotated[int, Field(
        default=3300,
        alias="celery_task_soft_time_limit",
    )]

    TORCH_VARIANT: Annotated[str, Field(
        default="cpu",
        alias="torch_variant",
    )]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == EnvironmentEnum.PRODUCTION

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )