
from dotenv import load_dotenv

from rag_pipeline.models import ChatMessage
from rag_pipeline.generation.groq_llm import GroqLLMService


def main():
    load_dotenv()

    llm = GroqLLMService()

    result = llm.generate([
        ChatMessage(
            role="system",
            content="Answer in one short sentence."
        ),
        ChatMessage(
            role="user",
            content="What is retrieval-augmented generation?"
        )
    ])

    print("Answer:", result.text)
    print("Model:", result.model)
    print("Input tokens:", result.input_tokens)
    print("Output tokens:", result.output_tokens)


if __name__ == "__main__":
    main()
