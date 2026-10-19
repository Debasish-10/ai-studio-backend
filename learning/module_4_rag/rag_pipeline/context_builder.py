
from rag_pipeline.models import (
    RetrievalResult,
    ContextSource,
    AssembledContext
)


def estimate_tokens(text: str) -> int:
    # Approximation only, not model-exact tokenization.
    return max(1, (len(text) + 3) // 4)


class ContextBuilder:

    def __init__(self, max_context_tokens: int = 2000):
        if max_context_tokens <= 0:
            raise ValueError(
                "max_context_tokens must be positive."
            )

        self.max_context_tokens = max_context_tokens

    def build(
        self,
        results: list[RetrievalResult]
    ) -> AssembledContext:

        context_parts = []
        sources = []
        seen_texts = set()

        used_tokens = 0

        for result in results:
            normalized_text = " ".join(
                result.text.split()
            )

            if not normalized_text:
                continue

            # Skip identical passages
            if normalized_text in seen_texts:
                continue

            label = f"Source {len(sources) + 1}"

            context_block = (
                f"[{label}]\n"
                f"Document: {result.source}\n"
                f"Page: {result.page_number}\n"
                f"Content:\n{result.text.strip()}"
            )

            required_tokens = estimate_tokens(
                context_block
            )

            if used_tokens + required_tokens > self.max_context_tokens:
                continue

            context_parts.append(context_block)

            sources.append(
                ContextSource(
                    label=label,
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    filename=result.source,
                    page_number=result.page_number
                )
            )

            seen_texts.add(normalized_text)
            used_tokens += required_tokens

        return AssembledContext(
            text="\n\n---\n\n".join(context_parts),
            sources=sources,
            estimated_tokens=estimate_tokens(
                "\n\n---\n\n".join(context_parts)
            ) if context_parts else 0
        )
