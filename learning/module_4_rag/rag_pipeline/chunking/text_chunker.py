from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.models import (
    DocumentPage,
    DocumentChunk
)
from rag_pipeline.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


class TextChunker:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def split(
        self,
        pages: list[DocumentPage],
        document_id: str
    ) -> list[DocumentChunk]:

        chunks = []

        for page in pages:
            page_chunks = self.splitter.split_text(
                page.text
            )

            for chunk_number, text in enumerate(
                page_chunks,
                start=1
            ):
                chunk_id = (
                    f"{document_id}"
                    f"_p{page.page_number}"
                    f"_c{chunk_number}"
                )

                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=text,
                        document_id=document_id,
                        page_number=page.page_number,
                        source=page.source,
                        chunk_number=chunk_number
                    )
                )

        return chunks