
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# 1. PDF path
PDF_PATH = (
    Path(__file__).parent
    / "sample_docs"
    / "employee_handbook.pdf"
)


# 2. Extract PDF pages with metadata
reader = PdfReader(PDF_PATH)

documents = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    if not text.strip():
        continue

    documents.append({
        "text": text,
        "metadata": {
            "source": PDF_PATH.name,
            "page": page_number
        }
    })

print("Extracted pages:", len(documents))


# 3. Configure recursive splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)


# 4. Split each page into chunks
chunks = []

for document in documents:
    page_chunks = splitter.split_text(document["text"])

    for chunk_number, chunk_text in enumerate(
        page_chunks,
        start=1
    ):
        chunks.append({
            "text": chunk_text,
            "metadata": {
                **document["metadata"],
                "chunk_number": chunk_number
            }
        })

print("Total chunks:", len(chunks))


# 5. Inspect chunks
for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print("Metadata:", chunk["metadata"])
    print("Characters:", len(chunk["text"]))
    print("Text:", chunk["text"])


# 6. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 7. Generate embeddings
chunk_texts = [chunk["text"] for chunk in chunks]

if not chunk_texts:
    raise ValueError(
        "No extractable text found in the PDF."
    )

embeddings = model.encode(chunk_texts)

print("\nEmbedding matrix shape:", embeddings.shape)


# 8. Inspect first embedding
print("First chunk metadata:", chunks[0]["metadata"])
print("First embedding shape:", embeddings[0].shape)
