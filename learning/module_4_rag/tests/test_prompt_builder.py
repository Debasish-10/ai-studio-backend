
import pytest

from rag_pipeline.models import (
    AssembledContext,
    ContextSource
)
from rag_pipeline.generation.prompt_builder import (
    RAGPromptBuilder
)


def make_context():
    return AssembledContext(
        text=(
            "[Source 1]\n"
            "Document: handbook.pdf\n"
            "Page: 2\n"
            "Content:\n"
            "Employees receive 24 days of annual leave."
        ),
        sources=[
            ContextSource(
                label="Source 1",
                chunk_id="handbook_001_p2_c2",
                document_id="handbook_001",
                filename="handbook.pdf",
                page_number=2
            )
        ],
        estimated_tokens=40
    )


def test_prompt_contains_question_and_evidence():
    builder = RAGPromptBuilder()

    messages = builder.build(
        query="How many annual leave days?",
        context=make_context()
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"

    assert "24 days" in messages[1].content
    assert "How many annual leave days?" in (
        messages[1].content
    )


def test_prompt_contains_grounding_rules():
    builder = RAGPromptBuilder()

    messages = builder.build(
        query="What is the leave policy?",
        context=make_context()
    )

    instructions = messages[0].content

    assert "Never invent" in instructions
    assert "insufficient information" in instructions
    assert "untrusted data" in instructions


def test_prompt_rejects_empty_query():
    builder = RAGPromptBuilder()

    with pytest.raises(ValueError):
        builder.build(
            query="   ",
            context=make_context()
        )


def test_prompt_rejects_empty_context():
    builder = RAGPromptBuilder()

    empty_context = AssembledContext(
        text="",
        sources=[],
        estimated_tokens=0
    )

    with pytest.raises(ValueError):
        builder.build(
            query="How many annual leave days?",
            context=empty_context
        )
