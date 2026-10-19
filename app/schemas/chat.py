from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class OutputFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    TABLE = "table"
    EMAIL = "email"
    HTML = "html"


class ConversationMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1, max_length=4_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[ConversationMessage] = Field(default_factory=list)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, allow_inf_nan=False)


class ChatResponse(BaseModel):
    message: str


class StructuredChatRequest(ChatRequest):
    pass


class StructuredChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=4_000)


class FormattedChatRequest(ChatRequest):
    output_format: OutputFormat


class FormattedChatResponse(BaseModel):
    output_format: OutputFormat
    content: StructuredChatResponse | str
