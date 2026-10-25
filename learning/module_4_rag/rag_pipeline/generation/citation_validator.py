
import re

from rag_pipeline.models import (
    ContextSource,
    ResolvedCitation,
    CitationValidationResult
)


class CitationValidator:

    # Matches [Source 1], [Source 2], etc.
    CITATION_PATTERN = re.compile(
        r"(?:\[Source\s+(\d+)\]|【Source\s+(\d+)】)"
    )

    def validate(
        self,
        answer: str,
        available_sources: list[ContextSource]
    ) -> CitationValidationResult:

        source_map = {
            source.label: source
            for source in available_sources
        }

        matches = self.CITATION_PATTERN.findall(answer)

        # Preserve first-occurrence order.
        referenced_labels = list(
        dict.fromkeys(
            f"Source {standard or unicode}"
            for standard, unicode in matches
        )
    )

        citations = []
        invalid_labels = []

        for label in referenced_labels:
            source = source_map.get(label)

            if source is None:
                invalid_labels.append(label)
                continue

            citations.append(
                ResolvedCitation(
                    label=label,
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    filename=source.filename,
                    page_number=source.page_number
                )
            )

        return CitationValidationResult(
            valid=not invalid_labels,
            citations=citations,
            invalid_labels=invalid_labels,
            has_citations=bool(referenced_labels)
        )
