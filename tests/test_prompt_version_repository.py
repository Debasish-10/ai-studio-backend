from unittest.mock import Mock
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models import PromptVersion
from app.repositories import PromptVersionRepository


def test_version_repository_creates_without_committing() -> None:
    session = Mock(spec=Session)
    repository = PromptVersionRepository(session)
    version = PromptVersion(
        prompt_id=uuid4(),
        version=1,
        content="Say hello",
    )

    assert repository.create(version) is version

    session.add.assert_called_once_with(version)
    session.flush.assert_called_once_with()
    session.commit.assert_not_called()


def test_version_repository_reads_next_number_and_history_in_descending_order() -> None:
    session = Mock(spec=Session)
    repository = PromptVersionRepository(session)
    prompt_id = uuid4()
    version = Mock(spec=PromptVersion)
    scalar_result = Mock()
    scalar_result.all.return_value = [version]
    session.scalar.side_effect = [2, version]
    session.scalars.return_value = scalar_result

    assert repository.next_version_number(prompt_id) == 3
    assert repository.list_for_prompt(prompt_id) == [version]
    assert repository.get_by_prompt_and_version(prompt_id, 2) is version

    history_statement = session.scalars.call_args.args[0]
    assert history_statement._order_by_clauses
    assert session.commit.call_count == 0
