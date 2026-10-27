
import fitz

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from rag_pipeline.chunking.token_aware_chunker import (
    TokenAwareChunker
)


PDF_PATH = "sample_docs/employee_handbook.pdf"


def load_pages(pdf_path: str) -> list[str]:
    with fitz.open(pdf_path) as pdf:
        return [page.get_text() for page in pdf]


def summarize(name, chunks, token_chunker):
    token_lengths = [
        token_chunker.count_tokens(chunk)
        for chunk in chunks
    ]

    print(f"\n{name}")
    print("-" * 45)
    print("Total chunks:", len(chunks))
    print("Average tokens:",
          round(sum(token_lengths) / len(token_lengths), 2))
    print("Max tokens:", max(token_lengths))
    print(
        "Above model limit:",
        sum(
            count > token_chunker.model_max_tokens
            for count in token_lengths
        )
    )

    print("\nSample chunk:\n")
    print(chunks[0][:600])


def main():
    pages = load_pages(PDF_PATH)

    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    token_chunker = TokenAwareChunker(
        chunk_size=200,
        chunk_overlap=40
    )

    character_chunks = []
    token_chunks = []

    for page_text in pages:
        if not page_text.strip():
            continue

        character_chunks.extend(
            character_splitter.split_text(page_text)
        )

        token_chunks.extend(
            token_chunker.split_text(page_text)
        )

    summarize(
        "Character-based baseline",
        character_chunks,
        token_chunker
    )

    summarize(
        "Token-aware strategy",
        token_chunks,
        token_chunker
    )


if __name__ == "__main__":
    main()
