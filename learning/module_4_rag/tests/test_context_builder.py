
from rag_pipeline.context_builder import ContextBuilder
from rag_pipeline.models import RetrievalResult


def make_result(chunk_id, text, score=0.85):
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="handbook_001",
        text=text,
        source="employee_handbook.pdf",
        page_number=2,
        similarity=score
    )


def test_context_builder_removes_duplicates():
    results = [
        make_result("chunk_1", "Employees receive 24 days."),
        make_result("chunk_2", "Employees receive 24 days."),
        make_result("chunk_3", "Remote work is allowed.")
    ]

    builder = ContextBuilder(max_context_tokens=1000)

    context = builder.build(results)

    assert len(context.sources) == 2
    assert context.text.count(
        "Employees receive 24 days."
    ) == 1


def test_context_builder_handles_empty_results():
    builder = ContextBuilder(max_context_tokens=1000)

    context = builder.build([])

    assert context.text == ""
    assert context.sources == []
    assert context.estimated_tokens == 0


def test_context_builder_respects_small_budget():
    results = [
        make_result(
            "chunk_1",
            "Employees receive 24 days of annual leave."
        )
    ]

    builder = ContextBuilder(max_context_tokens=2)

    context = builder.build(results)

    assert context.text == ""
    assert len(context.sources) == 0
