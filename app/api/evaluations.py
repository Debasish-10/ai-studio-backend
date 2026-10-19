from fastapi import APIRouter, Depends, HTTPException, status

from app.llm.provider import LLMProviderError
from app.schemas.evaluation import EvaluationRunRequest, EvaluationRunResponse
from app.services.chat_service import ChatService, get_chat_service
from app.services.evaluation_service import run_evaluation

router = APIRouter(tags=["evaluations"])


@router.post("/evaluations/run", response_model=EvaluationRunResponse)
def run_evaluation_endpoint(
    request: EvaluationRunRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> EvaluationRunResponse:
    try:
        return run_evaluation(request.dataset, chat_service, request.temperature)
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None
