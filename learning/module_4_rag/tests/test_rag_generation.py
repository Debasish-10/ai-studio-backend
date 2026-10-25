
from unittest.mock import Mock

from rag_pipeline.models import (
    RankedCandidate,
    LLMGenerationResult
)
from rag_pipeline.context_builder import ContextBuilder
from rag_pipeline.generation.prompt_builder import (
    RAGPromptBuilder
)
from rag_pipeline.generation.rag_generation import (
    RAGGenerationService
)


def test_generates_answer_from_retrieved_evidence():
    retrieval = Mock()
    llm = Mock()

    retrieval.retrieve.return_value = (
        Mock(strategy="semantic"),
        [
            RankedCandidate(
                chunk_id="handbook_001_p2_c2",
                document_id="handbook_001",
                text="Employees receive 24 days of annual leave.",
                source="handbook.pdf",
                page_number=2,
                score=0.95
            )
        ]
    )

    llm.generate.return_value = LLMGenerationResult(
        text="Employees receive 24 days [Source 1].",
        model="fake-model",
        input_tokens=100,
        output_tokens=15
    )

    service = RAGGenerationService(
        retrieval_service=retrieval,
        context_builder=ContextBuilder(1000),
        prompt_builder=RAGPromptBuilder(),
        llm_service=llm
    )

    result = service.generate(
        query="How many annual leave days?",
        filters={"document_id": "handbook_001"}
    )

    assert result.status == "answered"
    assert "24 days" in result.answer
    assert len(result.sources) == 1
    assert result.input_tokens == 100

    llm.generate.assert_called_once()


def test_no_evidence_skips_llm():
    retrieval = Mock()
    llm = Mock()

    retrieval.retrieve.return_value = (
        Mock(strategy="semantic"),
        []
    )

    service = RAGGenerationService(
        retrieval_service=retrieval,
        context_builder=ContextBuilder(1000),
        prompt_builder=RAGPromptBuilder(),
        llm_service=llm
    )

    result = service.generate(
        query="Does NovaTech provide pet insurance?",
        filters={"document_id": "handbook_001"}
    )

    assert result.status == "no_context"
    assert result.sources == []

    llm.generate.assert_not_called()
