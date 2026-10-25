
from unittest.mock import Mock

from rag_pipeline.conversation.memory import (
    ConversationMemory
)
from rag_pipeline.conversation.query_rewriter import (
    ConversationalQueryRewriter
)
from rag_pipeline.conversation.conversation_service import (
    ConversationService
)


def test_memory_is_isolated_by_conversation():
    memory = ConversationMemory(max_messages=4)

    memory.add("chat_1", "user", "Annual leave")
    memory.add("chat_2", "user", "Security policy")

    assert len(memory.get_recent("chat_1")) == 1
    assert len(memory.get_recent("chat_2")) == 1

    assert (
        memory.get_recent("chat_1")[0].content
        == "Annual leave"
    )


def test_memory_keeps_recent_messages():
    memory = ConversationMemory(max_messages=2)

    memory.add("chat_1", "user", "First")
    memory.add("chat_1", "assistant", "Second")
    memory.add("chat_1", "user", "Third")

    history = memory.get_recent("chat_1")

    assert [m.content for m in history] == [
        "Second",
        "Third"
    ]


def test_follow_up_query_rewriting():
    memory = ConversationMemory()

    memory.add(
        "chat_1",
        "user",
        "What is the annual leave policy?"
    )

    rewriter = ConversationalQueryRewriter()

    rewritten = rewriter.rewrite(
        query="Can they carry them forward?",
        history=memory.get_recent("chat_1")
    )

    assert "annual leave" in rewritten.lower()


def test_conversation_uses_rewritten_query():
    memory = ConversationMemory()
    rewriter = ConversationalQueryRewriter()
    rag_service = Mock()

    memory.add(
        "chat_1",
        "user",
        "What is the annual leave policy?"
    )

    rag_service.generate.return_value = Mock(
        answer="Up to 10 days [Source 1].",
        status="generated_with_citations"
    )

    service = ConversationService(
        memory=memory,
        query_rewriter=rewriter,
        rag_service=rag_service
    )

    service.ask(
        conversation_id="chat_1",
        query="Can they carry them forward?",
        filters={"document_id": "handbook_001"}
    )

    called_query = (
        rag_service.generate.call_args.kwargs["query"]
    )

    assert "annual leave" in called_query.lower()

    history = memory.get_recent("chat_1")

    # The original user message is stored.
    assert history[-2].content == (
        "Can they carry them forward?"
    )
