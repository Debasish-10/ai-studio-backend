from unittest.mock import Mock
from uuid import uuid4

from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.models import Prompt
from app.repositories import PromptRepository


def test_repository_creates_without_committing() -> None:
    session = Mock(spec=Session)
    repository = PromptRepository(session)
    prompt = Prompt(title="Greeting", content="Say hello")

    result = repository.create(prompt)

    assert result is prompt
    session.add.assert_called_once_with(prompt)
    session.flush.assert_called_once_with()
    session.commit.assert_not_called()


def test_repository_gets_and_lists_filtered_prompts_with_database_pagination() -> None:
    session = Mock(spec=Session)
    repository = PromptRepository(session)
    prompt_id = uuid4()
    prompt = Prompt(title="Greeting", content="Say hello")
    scalar_result = Mock()
    scalar_result.all.return_value = [prompt]
    session.scalars.return_value = scalar_result

    assert repository.get_by_id(prompt_id) is session.get.return_value
    session.scalar.return_value = 23

    assert repository.list_prompts(
        search="React",
        category="coding",
        is_favorite=True,
        is_template=False,
        page=2,
        page_size=10,
    ) == ([prompt], 23)

    session.get.assert_called_once_with(Prompt, prompt_id)
    statement = session.scalars.call_args.args[0]
    total_statement = session.scalar.call_args.args[0]
    assert statement.column_descriptions[0]["entity"] is Prompt
    assert statement._order_by_clauses
    assert statement._offset_clause.value == 10
    assert statement._limit_clause.value == 10
    statement_sql = str(statement.compile(dialect=postgresql.dialect()))
    total_sql = str(total_statement.compile(dialect=postgresql.dialect()))
    assert "ILIKE" in statement_sql
    assert "ILIKE" in total_sql
    assert "prompts.category" in statement_sql
    assert "prompts.is_favorite" in statement_sql
    assert "prompts.is_template" in statement_sql


def test_repository_lists_distinct_categories_with_a_database_query() -> None:
    session = Mock(spec=Session)
    repository = PromptRepository(session)
    scalar_result = Mock()
    scalar_result.all.return_value = ["general", "marketing"]
    session.scalars.return_value = scalar_result

    assert repository.list_categories() == ["general", "marketing"]

    statement = session.scalars.call_args.args[0]
    assert statement._distinct is True
    assert statement._order_by_clauses


def test_repository_updates_and_deletes_without_committing() -> None:
    session = Mock(spec=Session)
    repository = PromptRepository(session)
    prompt = Prompt(title="Greeting", content="Say hello")

    assert repository.update(prompt, {"title": "Welcome", "is_favorite": True}) is prompt
    repository.delete(prompt)

    assert prompt.title == "Welcome"
    assert prompt.is_favorite is True
    assert session.flush.call_count == 2
    session.delete.assert_called_once_with(prompt)
    session.commit.assert_not_called()
