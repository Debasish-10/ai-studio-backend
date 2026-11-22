
import re
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------------
# 1. Sample company documents
# -----------------------------------

documents = [
    Document(
        id="D1",
        page_content=(
            "Project Atlas is owned by "
            "the Platform Engineering team."
        ),
        metadata={
            "type": "ownership",
            "project": "Atlas",
            "source_id": "D1",
        },
    ),
    Document(
        id="D2",
        page_content=(
            "Platform Engineering requires "
            "two approvals before a production "
            "deployment can proceed."
        ),
        metadata={
            "type": "deployment_policy",
            "team": "Platform Engineering",
            "source_id": "D2",
        },
    ),
    Document(
        id="D3",
        page_content=(
            "Frontend Engineering requires "
            "one approval before production deployment."
        ),
        metadata={
            "type": "deployment_policy",
            "team": "Frontend Engineering",
            "source_id": "D3",
        },
    ),
    Document(
        id="D4",
        page_content=(
            "Project Nova is owned by "
            "the Frontend Engineering team."
        ),
        metadata={
            "type": "ownership",
            "project": "Nova",
            "source_id": "D4",
        },
    ),
]


# -----------------------------------
# 2. Initialize embeddings + ChromaDB
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_ID,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name=f"multi_hop_{uuid4().hex}",
    embedding_function=embeddings,
)

vector_store.add_documents(
    documents,
    ids=[doc.id for doc in documents],
)


# -----------------------------------
# 3. Hop 1: Find Atlas ownership
# -----------------------------------

def hop_1(state: dict) -> dict:
    results = vector_store.similarity_search(
        "Who owns Project Atlas?",
        k=1,
        filter={
            "$and": [
                {"type": {"$eq": "ownership"}},
                {"project": {"$eq": "Atlas"}},
            ]
        },
    )

    if not results:
        raise ValueError("Project ownership not found")

    ownership_doc = results[0]

    # Deterministic parsing for this sample format.
    match = re.search(
        r"owned by the (.+?) team\.",
        ownership_doc.page_content,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            "Could not extract the project owner"
        )

    owner = match.group(1).strip()

    return {
        **state,
        "owner": owner,
        "ownership_doc": ownership_doc,
    }


# -----------------------------------
# 4. Hop 2: Retrieve owner's policy
# -----------------------------------

def hop_2(state: dict) -> dict:
    owner = state["owner"]

    results = vector_store.similarity_search(
        f"{owner} production deployment approvals",
        k=1,
        filter={
            "$and": [
                {
                    "type": {
                        "$eq": "deployment_policy"
                    }
                },
                {
                    "team": {
                        "$eq": owner
                    }
                },
            ]
        },
    )

    if not results:
        raise ValueError(
            f"No deployment policy found for {owner}"
        )

    return {
        **state,
        "policy_doc": results[0],
    }


# -----------------------------------
# 5. Build evidence-backed result
# -----------------------------------

def build_result(state: dict) -> dict:
    ownership = state["ownership_doc"]
    policy = state["policy_doc"]

    return {
        "question": state["question"],
        "owner": state["owner"],
        "evidence": [
            {
                "source": ownership.metadata["source_id"],
                "text": ownership.page_content,
            },
            {
                "source": policy.metadata["source_id"],
                "text": policy.page_content,
            },
        ],
        "answer": (
            f"{ownership.page_content} "
            f"{policy.page_content} "
            "[D1] [D2]"
        ),
    }


# -----------------------------------
# 6. Compose sequential Runnables
# -----------------------------------

multi_hop_chain = (
    RunnableLambda(hop_1)
    | RunnableLambda(hop_2)
    | RunnableLambda(build_result)
)


# -----------------------------------
# 7. Execute
# -----------------------------------

def main():
    question = (
        "What is the production deployment approval "
        "policy for the team that owns Project Atlas?"
    )

    result = multi_hop_chain.invoke({
        "question": question,
    })

    print("\nQUESTION:")
    print(result["question"])

    print("\nHOP 1 — PROJECT OWNER:")
    print(result["owner"])

    print("\nRETRIEVED EVIDENCE:")
    for evidence in result["evidence"]:
        print(
            f"[{evidence['source']}] "
            f"{evidence['text']}"
        )

    print("\nFINAL ANSWER:")
    print(result["answer"])


if __name__ == "__main__":
    main()
