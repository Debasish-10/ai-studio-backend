
from transformers import AutoTokenizer

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


class TokenAwareChunker:

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        model_max_tokens: int = 256
    ):
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "Overlap must be smaller than chunk size."
            )

        if chunk_size + 2 > model_max_tokens:
            raise ValueError(
                "Chunk size leaves insufficient room "
                "for special tokens."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_max_tokens = model_max_tokens

        self.splitter = (
            RecursiveCharacterTextSplitter
            .from_huggingface_tokenizer(
                tokenizer=self.tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ]
            )
        )

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False
            )
        )

    def split_text(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)

        # Verify against the actual encoded input length.
        for chunk in chunks:
            actual_tokens = self.count_tokens(chunk)

            if actual_tokens > self.model_max_tokens:
                raise ValueError(
                    "Generated chunk exceeds model limit: "
                    f"{actual_tokens} tokens."
                )

        return chunks
