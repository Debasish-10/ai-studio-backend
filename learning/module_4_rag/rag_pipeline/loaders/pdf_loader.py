from pathlib import Path
from pypdf import PdfReader

from rag_pipeline.models import DocumentPage


class PDFLoader:

    def load(self, file_path: Path) -> list[DocumentPage]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        reader = PdfReader(file_path)
        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            pages.append(
                DocumentPage(
                    text=text,
                    page_number=page_number,
                    source=file_path.name
                )
            )

        return pages