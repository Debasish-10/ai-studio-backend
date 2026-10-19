from itertools import chain

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.llm.provider import LLMProviderError
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    FormattedChatRequest,
    FormattedChatResponse,
    StructuredChatRequest,
    StructuredChatResponse,
)
from app.schemas.tool import ToolChatRequest, ToolChatResponse
from app.services.chat_service import ChatService, get_chat_service
from app.tools import ToolExecutionError

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        return chat_service.create_response(
            request.message,
            request.history,
            request.temperature,
        )
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None


@router.post("/chat/stream")
def stream_chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Stream plain-text chat response chunks."""

    chunks = iter(
        chat_service.stream_response(
            request.message,
            request.history,
            request.temperature,
        )
    )

    try:
        first_chunk = next(chunks)
    except StopIteration:
        return StreamingResponse(iter(()), media_type="text/plain")
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None

    return StreamingResponse(
        chain((first_chunk,), chunks),
        media_type="text/plain",
    )


@router.post("/chat/structured", response_model=StructuredChatResponse)
def structured_chat(
    request: StructuredChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StructuredChatResponse:
    try:
        return chat_service.create_structured_response(
            request.message,
            request.history,
            request.temperature,
        )
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None


@router.post("/chat/formatted", response_model=FormattedChatResponse)
def formatted_chat(
    request: FormattedChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> FormattedChatResponse:
    try:
        return chat_service.create_formatted_response(
            request.message,
            request.output_format,
            request.history,
            request.temperature,
        )
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None


@router.post("/chat/tools", response_model=ToolChatResponse)
def tool_chat(
    request: ToolChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ToolChatResponse:
    try:
        return chat_service.create_tool_response(
            request.message,
            request.history,
            request.temperature,
        )
    except (LLMProviderError, ToolExecutionError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None
