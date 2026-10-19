
from rag_pipeline.retrieval import RetrievalService
from rag_pipeline.context_builder import ContextBuilder
from rag_pipeline.models import RAGPipelineResult


class RAGPipeline:

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder
    ):
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder

    def prepare_context(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.50
    ) -> RAGPipelineResult:

        # Step 1: Retrieve candidate chunks
        results = self.retrieval_service.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters={"document_id": document_id}
        )

        # Step 2: Assemble context
        context = self.context_builder.build(results)

        # Step 3: Determine result status
        status = (
            "ready"
            if context.text
            else "no_context"
        )

        return RAGPipelineResult(
            query=query,
            retrieved_count=len(results),
            context=context,
            status=status
        )
