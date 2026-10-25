
from dotenv import load_dotenv

from rag_pipeline.models import ChatMessage
from rag_pipeline.generation.groq_llm import (
    GroqLLMService
)


def main():
    load_dotenv()

    llm = GroqLLMService()

    messages = [
        ChatMessage(
            role="system",
            content="Explain clearly and briefly."
        ),
        ChatMessage(
            role="user",
            content="What is conversational RAG?"
        )
    ]

    print("\nStreaming response:\n")

    collected = []

    for text in llm.stream(messages):
        print(text, end="", flush=True)
        collected.append(text)

    full_answer = "".join(collected)

    print("\n\nCompleted.")
    print("Total characters:", len(full_answer))


if __name__ == "__main__":
    main()
