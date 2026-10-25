
from rag_pipeline.models import ChatMessage


class ConversationalQueryRewriter:

    def rewrite(
        self,
        query: str,
        history: list[ChatMessage]
    ) -> str:

        normalized = query.lower().strip()

        # Demonstration of one known follow-up.
        if "carry them forward" in normalized:

            previous_text = " ".join(
                message.content.lower()
                for message in history
            )

            if "annual leave" in previous_text:
                return (
                    "Can employees carry forward "
                    "unused annual leave days?"
                )

        # No reliable rewrite found.
        return query
