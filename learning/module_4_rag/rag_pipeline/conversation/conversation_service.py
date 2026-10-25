
class ConversationService:

    def __init__(
        self,
        memory,
        query_rewriter,
        rag_service
    ):
        self.memory = memory
        self.query_rewriter = query_rewriter
        self.rag_service = rag_service

    def ask(
        self,
        conversation_id: str,
        query: str,
        filters: dict
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # 1. Load history BEFORE adding current question.
        history = self.memory.get_recent(
            conversation_id
        )

        # 2. Rewrite for retrieval.
        retrieval_query = self.query_rewriter.rewrite(
            query=query,
            history=history
        )

        # 3. Call existing grounded RAG service.
        result = self.rag_service.generate(
            query=retrieval_query,
            filters=filters
        )

        # 4. Store original user message, not rewrite.
        self.memory.add(
            conversation_id,
            "user",
            query
        )

        # 5. Store completed assistant response.
        self.memory.add(
            conversation_id,
            "assistant",
            result.answer
        )

        return result
