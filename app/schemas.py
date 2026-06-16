"""Pydantic schemas for the vLLM Security Gateway."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str = Field(min_length=1)
    name: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    n: int | None = None


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class RiskAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    matched_rules: list[str]
    blocked: bool


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
    error_type: str | None = None
