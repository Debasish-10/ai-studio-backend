
from rag_pipeline.models import RetrievalResult
from rag_pipeline.context_builder import ContextBuilder
from rag_pipeline.generation.prompt_builder import (
    RAGPromptBuilder
)


def main():
    query = (
        "How many annual leave days do employees receive, "
        "and how many can be carried forward?"
    )

    # Sample evidence from our handbook.
    results = [
        RetrievalResult(
            chunk_id="handbook_001_p2_c2",
            document_id="handbook_001",
            text=(
                "Full-time permanent employees in India "
                "receive 24 working days of paid annual "
                "leave per calendar year."
            ),
            source="employee_handbook.pdf",
            page_number=2,
            similarity=0.92
        ),
        RetrievalResult(
            chunk_id="handbook_001_p2_c4",
            document_id="handbook_001",
            text=(
                "Employees may carry forward a maximum "
                "of 10 unused annual leave days into "
                "the next calendar year."
            ),
            source="employee_handbook.pdf",
            page_number=2,
            similarity=0.89
        )
    ]

    context_builder = ContextBuilder(
        max_context_tokens=1000
    )

    context = context_builder.build(results)

    prompt_builder = RAGPromptBuilder()

    messages = prompt_builder.build(
        query=query,
        context=context
    )

    for message in messages:
        print(f"\n===== {message.role.upper()} =====")
        print(message.content)


if __name__ == "__main__":
    main()
