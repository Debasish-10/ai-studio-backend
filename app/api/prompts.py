from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.core.database import get_db_session
from app.llm.provider import LLMProviderError
from app.repositories import PromptRepository, PromptVersionRepository
from app.schemas.chat import ChatResponse
from app.schemas.prompt import (
    PromptComparisonRequest,
    PromptComparisonResponse,
    PromptExperimentRequest,
    PromptCreate,
    PromptListQuery,
    PromptPage,
    PromptRead,
    PromptRequest,
    PromptResponse,
    PromptUpdate,
    PromptVersionRead,
)
from app.services.chat_service import ChatService, get_chat_service
from app.services.prompt_service import (
    PromptService,
    build_prompt,
    compare_prompt_outputs,
    run_prompt_experiment,
)

router = APIRouter(tags=["prompts"])


def get_prompt_service(session: Session = Depends(get_db_session)) -> PromptService:
    """Build a request-scoped prompt persistence service."""

    return PromptService(PromptRepository(session), PromptVersionRepository(session))


@router.post("/prompts/build", response_model=PromptResponse)
def build_prompt_endpoint(request: PromptRequest) -> PromptResponse:
    return build_prompt(request)


@router.post("/prompts/experiment", response_model=ChatResponse)
def run_prompt_experiment_endpoint(
    request: PromptExperimentRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        return run_prompt_experiment(request, chat_service)
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None


@router.post("/prompts/compare", response_model=PromptComparisonResponse)
def compare_prompt_outputs_endpoint(
    request: PromptComparisonRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> PromptComparisonResponse:
    try:
        return compare_prompt_outputs(request, chat_service)
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None


@router.post("/prompts", response_model=PromptRead, status_code=status.HTTP_201_CREATED)
def create_prompt_endpoint(
    request: PromptCreate,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptRead:
    return PromptRead.model_validate(prompt_service.create_prompt(request))


@router.get("/prompts", response_model=PromptPage)
def list_prompts_endpoint(
    query: PromptListQuery = Depends(),
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptPage:
    return prompt_service.list_prompts(query)


@router.get("/prompts/categories", response_model=list[str])
def list_prompt_categories_endpoint(
    prompt_service: PromptService = Depends(get_prompt_service),
) -> list[str]:
    return prompt_service.list_categories()


@router.get("/prompts/{prompt_id}", response_model=PromptRead)
def get_prompt_endpoint(
    prompt_id: UUID,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptRead:
    prompt = prompt_service.get_prompt(prompt_id)
    if prompt is None:
        raise _prompt_not_found()
    return PromptRead.model_validate(prompt)


@router.get("/prompts/{prompt_id}/versions", response_model=list[PromptVersionRead])
def list_prompt_versions_endpoint(
    prompt_id: UUID,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> list[PromptVersionRead]:
    versions = prompt_service.list_versions(prompt_id)
    if versions is None:
        raise _prompt_not_found()
    return versions


@router.get("/prompts/{prompt_id}/versions/{version}", response_model=PromptVersionRead)
def get_prompt_version_endpoint(
    prompt_id: UUID,
    version: int,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptVersionRead:
    prompt_version = prompt_service.get_version(prompt_id, version)
    if prompt_version is None:
        raise _prompt_not_found()
    return prompt_version


@router.patch("/prompts/{prompt_id}", response_model=PromptRead)
def update_prompt_endpoint(
    prompt_id: UUID,
    request: PromptUpdate,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptRead:
    prompt = prompt_service.update_prompt(prompt_id, request)
    if prompt is None:
        raise _prompt_not_found()
    return PromptRead.model_validate(prompt)


@router.put("/prompts/{prompt_id}/favorite", response_model=PromptRead)
def favorite_prompt_endpoint(
    prompt_id: UUID,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptRead:
    prompt = prompt_service.set_favorite(prompt_id, True)
    if prompt is None:
        raise _prompt_not_found()
    return PromptRead.model_validate(prompt)


@router.delete("/prompts/{prompt_id}/favorite", response_model=PromptRead)
def unfavorite_prompt_endpoint(
    prompt_id: UUID,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptRead:
    prompt = prompt_service.set_favorite(prompt_id, False)
    if prompt is None:
        raise _prompt_not_found()
    return PromptRead.model_validate(prompt)


@router.put("/prompts/{prompt_id}/template", response_model=PromptRead)
def mark_prompt_as_template_endpoint(
    prompt_id: UUID,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptRead:
    prompt = prompt_service.set_template(prompt_id, True)
    if prompt is None:
        raise _prompt_not_found()
    return PromptRead.model_validate(prompt)


@router.delete("/prompts/{prompt_id}/template", response_model=PromptRead)
def unmark_prompt_as_template_endpoint(
    prompt_id: UUID,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> PromptRead:
    prompt = prompt_service.set_template(prompt_id, False)
    if prompt is None:
        raise _prompt_not_found()
    return PromptRead.model_validate(prompt)


@router.delete("/prompts/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prompt_endpoint(
    prompt_id: UUID,
    prompt_service: PromptService = Depends(get_prompt_service),
) -> Response:
    if not prompt_service.delete_prompt(prompt_id):
        raise _prompt_not_found()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _prompt_not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt not found.")
