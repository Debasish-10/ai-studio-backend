
from rag_pipeline.vectorstores.chroma_store import ChromaStore


def main():
    store = ChromaStore()

    chunks = store.get_chunks(
        filters={"document_id": "handbook_001"}
    )

    for chunk in chunks:
        print("\n" + "=" * 60)
        print("Chunk ID:", chunk["id"])
        print("Page:", chunk["page"])
        print("Text:")
        print(chunk["text"])


if __name__ == "__main__":
    main()
