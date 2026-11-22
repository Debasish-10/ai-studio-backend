
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.retrievers.self_query.base import (
    SelfQueryRetriever,
)
from langchain_classic.chains.query_constructor.schema import (
    AttributeInfo,
)

from langchain_community.query_constructors.chroma import (
    ChromaTranslator,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------------
# 1. Sample documents
# ----------------------------------

documents = [
    Document(
        page_content=(
            "Employees can carry forward up to "
            "10 unused annual leave days."
        ),
        metadata={
            "category": "HR",
            "year": 2025,
            "source_type": "PDF",
        },
    ),
    Document(
        page_content=(
            "Annual leave carry-forward requires "
            "approval from the reporting manager."
        ),
        metadata={
            "category": "HR",
            "year": 2023,
            "source_type": "PDF",
        },
    ),
    Document(
        page_content=(
            "Employees may work remotely "
            "two days per week."
        ),
        metadata={
            "category": "HR",
            "year": 2025,
            "source_type": "Website",
        },
    ),
    Document(
        page_content=(
            "Production deployments must pass "
            "automated security checks."
        ),
        metadata={
            "category": "Engineering",
            "year": 2025,
            "source_type": "PDF",
        },
    ),
    Document(
        page_content=(
            "Database migrations require "
            "backward-compatible schema changes."
        ),
        metadata={
            "category": "Engineering",
            "year": 2024,
            "source_type": "Website",
        },
    ),
    Document(
        page_content=(
            "Employees must submit reimbursement "
            "claims within 30 days."
        ),
        metadata={
            "category": "Finance",
            "year": 2025,
            "source_type": "PDF",
        },
    ),
]


# ----------------------------------
# 2. Embedding model
# ----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_ID,
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },
)


# ----------------------------------
# 3. Temporary Chroma collection
# ----------------------------------

vector_store = Chroma(
    collection_name=f"self_query_{uuid4().hex}",
    embedding_function=embeddings,
)

vector_store.add_documents(documents)


# ----------------------------------
# 4. Configure LLM
# ----------------------------------

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing")

llm = ChatGroq(
    model=os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    ),
    temperature=0,
)


# ----------------------------------
# 5. Describe metadata schema
# ----------------------------------

metadata_field_info = [
    AttributeInfo(
        name="category",
        description=(
            "Company document category. "
            "Allowed values: HR, Engineering, Finance."
        ),
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="Publication year of the document.",
        type="integer",
    ),
    AttributeInfo(
        name="source_type",
        description=(
            "Document source. "
            "Allowed values: PDF, Website."
        ),
        type="string",
    ),
]


# ----------------------------------
# 6. Create self-query retriever
# ----------------------------------

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    document_contents=(
        "Company documents covering HR policies, "
        "engineering practices and finance rules."
    ),
    metadata_field_info=metadata_field_info,
    structured_query_translator=ChromaTranslator(),
    search_kwargs={"k": 5},
    verbose=True,
)


# ----------------------------------
# 7. Execute and inspect
# ----------------------------------

def run_query(question: str):
    print("\n" + "=" * 60)
    print("QUESTION:", question)

    # Inspect LLM-generated structured query.
    structured_query = retriever.query_constructor.invoke(
        {"query": question}
    )

    print("\nGENERATED SEMANTIC QUERY:")
    print(structured_query.query)

    print("\nGENERATED FILTER:")
    print(structured_query.filter)

    # Normal retriever invocation.
    # It constructs its own query again.
    results = retriever.invoke(question)

    print("\nRETRIEVED DOCUMENTS:")

    for index, doc in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Text:", doc.page_content)
        print("Metadata:", doc.metadata)


def main():
    questions = [
        (
            "Find annual leave carry-forward "
            "rules in HR documents from 2025."
        ),
        (
            "Show engineering practices "
            "published after 2024."
        ),
        (
            "Find remote work policies "
            "from websites."
        ),
    ]

    for question in questions:
        run_query(question)


if __name__ == "__main__":
    main()
