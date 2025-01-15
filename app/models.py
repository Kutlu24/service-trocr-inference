from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel, Field, ConfigDict
from pydantic_core import core_schema
from typing import Optional, Any, Tuple
from enum import Enum


class StateEnum(str, Enum):
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    DONE = "done"


class PyObjectId(str):
    @classmethod
    def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ])
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def validate(cls, value) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise ValueError("Invalid ObjectId")

        return ObjectId(value)


class InferenceBaseModel(BaseModel):
    repo_name: str = Field(
        alias="repo_name",
        description="Name of the GitHub-repository.",
        title="Repository-Name",
        examples=["your_github_name/your_repo_name"],
        frozen=True,
    )
    repo_folder: str = Field(
        alias="repo_folder",
        description="Folder in the repository the files are fetched from.",
        title="Repository-Folder",
        examples=["xml", "page"],
    )
    directory: Optional[str] = Field(
        default="tmp",
        alias="directory",
        description="Directory to save the files temporarily to.",
        title="Directory",
        examples=["tmp"],
    )
    in_path: Optional[str] = Field(
        default="preprocessed",
        alias="in_path",
        description="Path to save the preprocessed files.",
        title="In-Path",
        examples=["fetched"],
    )
    out_path: Optional[str] = Field(
        default="inference_results",
        alias="out_path",
        description="Path to save the inference results.",
        title="Out-Path",
        examples=["inference_results"],
    )
    trocr_model: Optional[str] = Field(
        default="microsoft/trocr-large-handwritten",
        alias="trocr_model",
        description="TrOCR model used in inference process.",
        title="TrOCR_Model",
    )
    trocr_processor: Optional[str] = Field(
        default="microsoft/trocr-large-handwritten",
        alias="trocr_processor",
        description="TrOCR processor used in inference process.",
        title="TrOCR_Processor",
    )
    use_cuda: Optional[bool] = Field(
        default=True,
        alias="use_cuda",
        description="Whether to use cuda.",
        title="Use-Cuda",
    )
    do_resize: Optional[bool] = Field(
        default=True,
        alias="do_resize",
        description="Whether to resize image.",
        title="Do-Resize",
    )
    aspect_ratio_resize: Optional[bool] = Field(
        default=True,
        alias="aspect_ratio_resize",
        description="Whether to use aspect ratio to resize image.",
        title="Aspect-Ratio-Resize",
    )
    output_txt: Optional[bool] = Field(
        default=True,
        alias="output_txt",
        description="Whether to create an .txt file for inference results.",
        title="Output-txt",
    )
    output_xml: Optional[bool] = Field(
        default=True,
        alias="output_xml",
        description="Whether to save inference results in XML file.",
        title="Output-XML",
    )
    image_size: Optional[Tuple[int, int]] = Field(
        default=(384, 384),
        alias="image_size",
        description="size of image to be processed.",
        title="Image-Size",
    )
    preprocesing_uri: Optional[str] = Field(
        default="",
        alias="preprocesing_uri",
        description="URI to download preprocessing files from.",
        title="Preprocessing-URI",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class InferenceRequestModel(InferenceBaseModel):
    password: Optional[str] = Field(
        alias="password",
        description="Password to access the status.",
        title="Password (optional)",
        default=None,
    )
    github_access_token: str = Field(
        alias="github_access_token",
        description="Access token to authenticate with the GitHub API.",
        title="GitHub-Access-Token",
        examples=["ghp_1234567890"],
        default=None,
    )


class InferenceResponseModel(InferenceBaseModel):
    id: PyObjectId = Field(
        alias="_id",
        description="Unique identifier of the inference status.",
        title="Inference-Status-ID",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        alias="created_at",
        description="Timestamp of the inference status creation.",
        title="Created-At",
    )
    runtime_seconds: int = Field(
        default=0,
        alias="runtime",
        description="Runtime of the inference status in seconds.",
        title="Runtime",
    )
    progress: int = Field(
        default=0,
        alias="progress",
        description="Current progress of the inference status.",
        title="Progress",
    )
    state: StateEnum = Field(
        default=StateEnum.IN_PROGRESS,
        alias="state",
        description="Current state of the inference status.",
        title="State",
    )


class InferenceDBModel(InferenceResponseModel):
    password: str = Field(
        alias="password",
        description="Password to access the status.",
        title="Password",
        default=None,
    )
