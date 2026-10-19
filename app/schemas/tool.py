from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas.chat import ChatRequest


class ToolDefinition(BaseModel):
    """Provider-neutral description of a backend tool."""

    name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=500)
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A validated tool request translated from a provider response."""

    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any]


class ToolModelResponse(BaseModel):
    """The first provider response: either text or a single tool request."""

    text: str | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def validate_response_shape(self) -> "ToolModelResponse":
        if (self.text is None) == (self.tool_call is None):
            raise ValueError("A tool response must contain exactly one of text or tool_call.")
        return self


class ToolChatRequest(ChatRequest):
    pass


class ToolChatResponse(BaseModel):
    message: str
    tool_name: str | None = None
