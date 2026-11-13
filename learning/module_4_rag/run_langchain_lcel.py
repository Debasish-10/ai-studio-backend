
import asyncio

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)


# -----------------------------------
# 1. Example documents
# -----------------------------------

documents = [
    Document(
        page_content=(
            "Employees receive 24 working days "
            "of annual leave each year."
        ),
        metadata={"page": 2},
    ),
    Document(
        page_content=(
            "Employees can carry forward up to "
            "10 unused annual leave days."
        ),
        metadata={"page": 3},
    ),
    Document(
        page_content=(
            "Employees may work remotely "
            "two days per week."
        ),
        metadata={"page": 4},
    ),
]


# -----------------------------------
# 2. Our normal Python functions
# -----------------------------------

def normalize_query(query: str) -> str:
    return query.strip().lower()


def retrieve_documents(query: str) -> list[Document]:
    # Demo only. Replace with Chroma later.
    if "carry forward" in query:
        return [documents[1]]

    if "remote" in query:
        return [documents[2]]

    return [documents[0]]


def format_documents(docs: list[Document]) -> str:
    return "\n\n".join(
        (
            f"[Page {doc.metadata['page']}]\n"
            f"{doc.page_content}"
        )
        for doc in docs
    )


# -----------------------------------
# 3. Convert functions to Runnables
# -----------------------------------

normalizer = RunnableLambda(normalize_query)

retriever = RunnableLambda(retrieve_documents)

formatter = RunnableLambda(format_documents)


# -----------------------------------
# 4. Sequential context pipeline
# -----------------------------------

context_chain = (
    normalizer
    | retriever
    | formatter
)


# -----------------------------------
# 5. Parallel input construction
# -----------------------------------

input_chain = RunnableParallel(
    context=context_chain,
    question=RunnablePassthrough(),
)


# -----------------------------------
# 6. Prompt template
# -----------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful HR assistant. "
        "Answer using only the provided context. "
        "If information is missing, say you don't know.",
    ),
    (
        "human",
        "Context:\n{context}\n\n"
        "Question:\n{question}",
    ),
])


# -----------------------------------
# 7. Complete LCEL pipeline
# -----------------------------------

rag_prompt_chain = input_chain | prompt


async def main():
    question = (
        "  How many days can employees carry forward?  "
    )

    # A. RunnableSequence
    print("\n=== A. CONTEXT CHAIN ===")
    context = context_chain.invoke(question)
    print(context)

    # B. RunnableParallel
    print("\n=== B. PARALLEL INPUTS ===")
    inputs = input_chain.invoke(question)
    print(inputs)

    # C. Complete prompt chain
    print("\n=== C. PROMPT ===")
    result = rag_prompt_chain.invoke(question)

    for message in result.messages:
        print(f"\n{message.type.upper()}:")
        print(message.content)

    # D. Batch execution
    print("\n=== D. BATCH ===")

    questions = [
        "How many annual leave days?",
        "How many days can employees carry forward?",
        "How many remote work days?",
    ]

    results = context_chain.batch(questions)

    for question, context in zip(questions, results):
        print("\nQuestion:", question)
        print("Context:", context)

    # E. Async invocation
    print("\n=== E. ASYNC ===")

    async_result = await rag_prompt_chain.ainvoke(
        "What is the remote work policy?"
    )

    print(
        async_result.messages[-1].content
    )


if __name__ == "__main__":
    asyncio.run(main())
