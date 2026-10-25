
from rag_pipeline.models import ContextSource
from rag_pipeline.generation.citation_validator import CitationValidator

def make_sources():
    return [
        ContextSource(
            label="Source 1",
            chunk_id="handbook_001_p2_c2",
            document_id="handbook_001",
            filename="employee_handbook.pdf",
            page_number=2
        ),
        ContextSource(
            label="Source 2",
            chunk_id="handbook_001_p2_c4",
            document_id="handbook_001",
            filename="employee_handbook.pdf",
            page_number=2
        )
    ]


def test_valid_citations():
    result = CitationValidator().validate(
        answer=(
            "Employees receive 24 days [Source 1]. "
            "They may carry forward 10 days [Source 2]."
        ),
        available_sources=make_sources()
    )

    assert result.valid is True
    assert result.has_citations is True
    assert len(result.citations) == 2
    assert result.invalid_labels == []


def test_unknown_citation():
    result = CitationValidator().validate(
        answer="Employees receive 24 days [Source 99].",
        available_sources=make_sources()
    )

    assert result.valid is False
    assert result.invalid_labels == ["Source 99"]
    assert result.citations == []


def test_duplicate_citation():
    result = CitationValidator().validate(
        answer=(
            "Annual leave is 24 days [Source 1]. "
            "That is the annual entitlement [Source 1]."
        ),
        available_sources=make_sources()
    )

    assert result.valid is True
    assert len(result.citations) == 1


def test_answer_without_citation():
    result = CitationValidator().validate(
        answer="The documents do not specify pet insurance.",
        available_sources=make_sources()
    )

    assert result.valid is True
    assert result.has_citations is False
    assert result.citations == []
