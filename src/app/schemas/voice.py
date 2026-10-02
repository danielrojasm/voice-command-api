from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1)
    done: bool = False


class TaskReplace(BaseModel):
    title: str = Field(..., min_length=1)
    done: bool


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    done: bool | None = None


class Task(BaseModel):
    id: int
    title: str
    done: bool


class InstructionRequest(BaseModel):
    transcription: str = Field(..., min_length=1)


class InstructionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(..., min_length=1)
    method: str = Field(..., min_length=1)
    params: dict[str, Any]


class TranscribeFlowResponse(BaseModel):
    transcription: str = Field(..., min_length=1)
    instruction: InstructionPayload
    result: Any
