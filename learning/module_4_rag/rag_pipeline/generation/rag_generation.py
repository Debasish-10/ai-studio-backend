
from rag_pipeline.models import RetrievalResult,RAGAnswer


class RAGGenerationService:

    def __init__(
        self,
        retrieval_service,
        context_builder,
        prompt_builder,
        llm_service,
        citation_validator
    ):
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        self.citation_validator = citation_validator

    def prepare(
        self,
        query: str,
        filters: dict,
        top_k: int = 5
    ) -> RAGAnswer:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Step 1: Adaptive retrieval
        plan, candidates = (
            self.retrieval_service.retrieve(
                query=query,
                filters=filters,
                top_k=top_k
            )
        )

        # Step 2: Convert existing ranked candidates
        # into the evidence format ContextBuilder uses.
        from rag_pipeline.models import RetrievalResult

        evidence = [
            RetrievalResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                source=c.source,
                page_number=c.page_number,
                similarity=0.0
            )
            for c in candidates
        ]

        # Step 3: Assemble context
        context = self.context_builder.build(evidence)

        # Step 4: Handle missing context
        if not context.text:
            return RAGAnswer(
                query=query,
                answer=(
                    "I couldn't find sufficient information "
                    "in the retrieved documents to answer."
                ),
                status="no_context"
            )

        # We are not calling the LLM yet.
        # The context is now ready for Part 2.
        return RAGAnswer(
            query=query,
            answer="",
            status="context_ready",
            sources=context.sources
        )

    def generate(
            self,
            query: str,
            filters: dict,
            top_k: int = 5
        ) -> RAGAnswer:

            if not query.strip():
                raise ValueError("Query cannot be empty.")

            # 1. Retrieve candidate passages.
            plan, candidates = self.retrieval_service.retrieve(
                query=query,
                filters=filters,
                top_k=top_k
            )

            # 2. Convert to our current ContextBuilder model.
            # The placeholder similarity is not a real score.
            evidence = [
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    text=c.text,
                    source=c.source,
                    page_number=c.page_number,
                    similarity=0.0
                )
                for c in candidates
            ]

            # 3. Assemble evidence and source mapping.
            context = self.context_builder.build(evidence)

            # 4. No evidence: skip the LLM call.
            if not context.text.strip():
                return RAGAnswer(
                    query=query,
                    answer=(
                        "I couldn't find sufficient information "
                        "in the available documents to answer."
                    ),
                    status="no_context"
                )

            # 5. Build the grounded messages.
            messages = self.prompt_builder.build(
                query=query,
                context=context
            )

            # 6. Generate answer through provider interface.
            generation = self.llm_service.generate(messages)
            
            # NEW: Validate citation labels.
            validation = self.citation_validator.validate(
                answer=generation.text,
                available_sources=context.sources
            )

            # Reject references to unknown source labels.
            if not validation.valid:
                return RAGAnswer(
                    query=query,
                    answer=(
                        "I couldn't produce an answer with "
                        "verifiable source references."
                    ),
                    status="invalid_citations",
                    sources=context.sources,
                    model=generation.model,
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens
                )

            # This state means generation succeeded and all
            # recognized citation labels were resolvable.
            # It does NOT guarantee factual support.
            return RAGAnswer(
                query=query,
                answer=generation.text,
                status=(
                    "generated_with_citations"
                    if validation.has_citations
                    else "generated_without_citations"
                ),
                sources=context.sources,
                citations=validation.citations,
                model=generation.model,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens
            )
