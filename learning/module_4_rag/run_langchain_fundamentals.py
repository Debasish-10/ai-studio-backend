
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# 1. Our learning corpus
documents = [
    Document(
        page_content=(
            "Employees receive 24 working days "
            "of annual leave each year."
        ),
        metadata={"source": "handbook", "page": 2},
    ),
    Document(
        page_content=(
            "Employees can carry forward up to "
            "10 unused annual leave days."
        ),
        metadata={"source": "handbook", "page": 3},
    ),
    Document(
        page_content=(
            "Employees may work remotely "
            "two days per week."
        ),
        metadata={"source": "handbook", "page": 4},
    ),
]


# 2. Query preprocessing
def normalize_query(query: str) -> str:
    return query.strip().lower()


# 3. Simplified deterministic retrieval
def retrieve_document(query: str) -> dict:
    if "carry forward" in query:
        selected = documents[1]
    elif "remote" in query:
        selected = documents[2]
    else:
        selected = documents[0]

    return {
        "question": query,
        "context": selected.page_content,
        "source": selected.metadata,
    }


# 4. Prepare input for prompt
def prepare_prompt_input(data: dict) -> dict:
    return {
        "question": data["question"],
        "context": data["context"],
    }


# 5. Reusable prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer only using the supplied context. "
        "If the answer is missing, say you don't know."
    ),
    (
        "human",
        "Context:\n{context}\n\n"
        "Question: {question}"
    ),
])


# 6. Compose a pipeline
chain = (
    RunnableLambda(normalize_query)
    | RunnableLambda(retrieve_document)
    | RunnableLambda(prepare_prompt_input)
    | prompt
)


def main():
    question = (
        "  How many days can employees carry forward?  "
    )

    result = chain.invoke(question)

    print("\nOUTPUT TYPE:")
    print(type(result).__name__)

    print("\nGENERATED PROMPT:")
    for message in result.messages:
        print(f"\n{message.type.upper()}:")
        print(message.content)


if __name__ == "__main__":
    main()
