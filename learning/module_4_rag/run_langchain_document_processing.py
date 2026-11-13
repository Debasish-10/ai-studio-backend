
from pathlib import Path
from statistics import mean

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer


PDF_PATH = (
    Path(__file__).resolve().parent
    / "sample_docs"
    / "employee_handbook.pdf"
)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf():
    loader = PyMuPDFLoader(
        str(PDF_PATH),
        mode="page",
    )

    pages = loader.load()

    # Add application-specific document identity.
    for page in pages:
        page.metadata["document_id"] = "handbook_001"
        page.metadata["document_version"] = 1

    return pages


def get_splitters(tokenizer):
    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )

    token_splitter = (
        RecursiveCharacterTextSplitter
        .from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=120,
            chunk_overlap=24,
            add_start_index=True,
        )
    )

    return character_splitter, token_splitter


def inspect_chunks(label, chunks, tokenizer):
    lengths = [
        len(
            tokenizer.encode(
                chunk.page_content,
                add_special_tokens=True,
                truncation=False,
            )
        )
        for chunk in chunks
    ]

    print(f"\n--- {label} ---")
    print("Chunk count:", len(chunks))
    print("Average tokens:", round(mean(lengths), 2))
    print("Maximum tokens:", max(lengths))

    print("\nFIRST CHUNK")
    print(chunks[0].page_content[:400])
    print("\nMETADATA")
    print(chunks[0].metadata)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF missing: {PDF_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    # 1. Load PDF pages.
    pages = load_pdf()

    print("TOTAL PAGES:", len(pages))

    print("\nFIRST PAGE METADATA")
    print(pages[0].metadata)

    print("\nFIRST PAGE CONTENT")
    print(pages[0].page_content[:300])

    # 2. Configure splitters.
    char_splitter, token_splitter = get_splitters(
        tokenizer
    )

    # 3. Split into LangChain Documents.
    char_chunks = char_splitter.split_documents(
        pages
    )

    token_chunks = token_splitter.split_documents(
        pages
    )

    # 4. Inspect output.
    inspect_chunks(
        "CHARACTER-BASED",
        char_chunks,
        tokenizer,
    )

    inspect_chunks(
        "TOKEN-AWARE",
        token_chunks,
        tokenizer,
    )

    # 5. Check the model's input constraints.
    # MiniLM has a configured limit of 256 tokens.
    too_long = [
        chunk
        for chunk in token_chunks
        if len(
            tokenizer.encode(
                chunk.page_content,
                add_special_tokens=True,
                truncation=False,
            )
        ) > 256
    ]

    print("\nCHUNKS EXCEEDING 256 TOKENS:", len(too_long))


if __name__ == "__main__":
    main()
