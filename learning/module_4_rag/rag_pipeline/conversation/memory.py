
from collections import defaultdict

from rag_pipeline.models import ChatMessage


class ConversationMemory:

    def __init__(self, max_messages: int = 6):
        self.max_messages = max_messages
        self._sessions = defaultdict(list)

    def add(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> None:

        self._sessions[conversation_id].append(
            ChatMessage(
                role=role,
                content=content
            )
        )

        # Keep only the latest N messages.
        self._sessions[conversation_id] = (
            self._sessions[conversation_id][
                -self.max_messages:
            ]
        )

    def get_recent(
        self,
        conversation_id: str
    ) -> list[ChatMessage]:

        return list(
            self._sessions.get(conversation_id, [])
        )

    def clear(self, conversation_id: str) -> None:
        self._sessions.pop(conversation_id, None)
