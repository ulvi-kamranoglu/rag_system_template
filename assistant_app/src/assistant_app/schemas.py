"""Public HTTP contracts; importing these models does not load GPU models."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    history: list[Message] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Source(BaseModel):
    title: str
    url: str
    date: str
    source: str = ""


class Timings(BaseModel):
    translate: float
    search: float
    llm: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    search_query: str
    timings: Timings
    intent: str
    language: str | None
    show_sources: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str | None
    retriever_ready: bool
    llm_ready: bool
    error: str | None


class ErrorResponse(BaseModel):
    error: str
