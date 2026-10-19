from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Prompt, PromptVersion
from app.repositories import PromptRepository, PromptVersionRepository
from app.schemas.chat import ChatResponse
from app.schemas.prompt import (
    Example,
    PromptCreate,
    PromptComparisonRequest,
    PromptComparisonResponse,
    PromptComparisonResult,
    PromptExperimentRequest,
    PromptListQuery,
    PromptPage,
    PromptRequest,
    PromptRead,
    PromptResponse,
    PromptUpdate,
    PromptVersionRead,
)
from app.services.chat_service import ChatService


class PromptService:
    """Coordinates prompt persistence and owns transaction boundaries."""

    def __init__(
        self,
        repository: PromptRepository,
        version_repository: PromptVersionRepository | None = None,
    ) -> None:
        self.repository = repository
        self.session: Session = repository.session
        self.version_repository = version_repository or PromptVersionRepository(self.session)

    def create_prompt(self, data: PromptCreate) -> Prompt:
        prompt = Prompt(**data.model_dump())

        def create() -> Prompt:
            self.repository.create(prompt)
            self.version_repository.create(self._snapshot(prompt, version=1))
            return prompt

        return self._persist(create)

    def get_prompt(self, prompt_id: UUID) -> Prompt | None:
        return self.repository.get_by_id(prompt_id)

    def list_prompts(self, query: PromptListQuery) -> PromptPage:
        prompts, total = self.repository.list_prompts(
            search=query.search,
            category=query.category,
            is_favorite=query.is_favorite,
            is_template=query.is_template,
            page=query.page,
            page_size=query.page_size,
        )
        return PromptPage(
            items=[PromptRead.model_validate(prompt) for prompt in prompts],
            page=query.page,
            page_size=query.page_size,
            total=total,
            total_pages=(total + query.page_size - 1) // query.page_size,
        )

    def list_categories(self) -> list[str]:
        return self.repository.list_categories()

    def update_prompt(self, prompt_id: UUID, data: PromptUpdate) -> Prompt | None:
        prompt = self.repository.get_by_id(prompt_id)
        if prompt is None:
            return None

        values = data.model_dump(exclude_unset=True)
        if all(getattr(prompt, field) == value for field, value in values.items()):
            return prompt

        # Prompt versions represent changes to the prompt instructions, not
        # presentation or library metadata.  Calculate this before mutating
        # the ORM instance so a mixed update can snapshot its final state.
        content_changed = (
            "content" in values and values["content"] != prompt.content
        )

        def update() -> Prompt:
            self.repository.update(prompt, values)
            if content_changed:
                version = self.version_repository.next_version_number(prompt.id)
                self.version_repository.create(
                    self._snapshot(prompt, version=version)
                )
            return prompt

        return self._persist(update)

    def delete_prompt(self, prompt_id: UUID) -> bool:
        prompt = self.repository.get_by_id(prompt_id)
        if prompt is None:
            return False

        self._persist(lambda: self.repository.delete(prompt))
        return True

    def set_favorite(self, prompt_id: UUID, is_favorite: bool) -> Prompt | None:
        return self.update_prompt(prompt_id, PromptUpdate(is_favorite=is_favorite))

    def set_template(self, prompt_id: UUID, is_template: bool) -> Prompt | None:
        return self.update_prompt(prompt_id, PromptUpdate(is_template=is_template))

    def list_versions(self, prompt_id: UUID) -> list[PromptVersionRead] | None:
        if self.repository.get_by_id(prompt_id) is None:
            return None
        return [
            PromptVersionRead.model_validate(version)
            for version in self.version_repository.list_for_prompt(prompt_id)
        ]

    def get_version(self, prompt_id: UUID, version: int) -> PromptVersionRead | None:
        if self.repository.get_by_id(prompt_id) is None:
            return None
        prompt_version = self.version_repository.get_by_prompt_and_version(prompt_id, version)
        if prompt_version is None:
            return None
        return PromptVersionRead.model_validate(prompt_version)

    @staticmethod
    def _snapshot(prompt: Prompt, version: int) -> PromptVersion:
        return PromptVersion(
            prompt_id=prompt.id,
            version=version,
            content=prompt.content,
        )

    def _persist(self, operation):
        try:
            result = operation()
            self.session.commit()
            return result
        except Exception:
            self.session.rollback()
            raise


def build_prompt(request: PromptRequest | PromptExperimentRequest) -> PromptResponse:
    """Build the Sprint 2 structured prompt from the supplied request."""

    examples = "\n\n".join(_format_example(example) for example in request.examples)
    prompt = "\n\n".join(
        [
            f"System Instructions:\n{request.system_instructions}",
            f"Context:\n{request.context}",
            f"Examples:\n{examples}",
            f"User Instructions:\n{request.user_instructions}",
        ]
    )

    return PromptResponse(prompt=prompt)


def run_prompt_experiment(
    request: PromptExperimentRequest,
    chat_service: ChatService,
) -> ChatResponse:
    """Build and run a zero-shot, one-shot, or few-shot prompt experiment."""

    _, response = _execute_prompt_experiment(request, chat_service)
    return response


def compare_prompt_outputs(
    request: PromptComparisonRequest,
    chat_service: ChatService,
) -> PromptComparisonResponse:
    """Run the same input against each supplied prompt configuration."""

    results = []
    for configuration in request.configurations:
        prompt, response = _execute_prompt_experiment(
            PromptExperimentRequest(
                shot_type=configuration.shot_type,
                system_instructions=configuration.system_instructions,
                context=configuration.context,
                examples=configuration.examples,
                temperature=configuration.temperature,
                user_instructions=request.user_instructions,
            ),
            chat_service,
        )
        results.append(
            PromptComparisonResult(
                label=configuration.label,
                prompt=prompt,
                output=response.message,
            )
        )

    return PromptComparisonResponse(results=results)


def _format_example(example: Example) -> str:
    return f"Input:\n{example.input}\nOutput:\n{example.output}"


def _execute_prompt_experiment(
    request: PromptExperimentRequest,
    chat_service: ChatService,
) -> tuple[str, ChatResponse]:
    prompt = build_prompt(request).prompt
    response = chat_service.create_response(prompt, temperature=request.temperature)
    return prompt, response
