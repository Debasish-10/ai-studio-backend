
import pytest

from rag_pipeline.chunking.token_aware_chunker import (
    TokenAwareChunker
)


@pytest.fixture(scope="module")
def chunker():
    return TokenAwareChunker(
        chunk_size=200,
        chunk_overlap=40
    )


def test_short_text_stays_intact(chunker):
    text = "Employees receive 24 annual leave days."

    chunks = chunker.split_text(text)

    assert chunks == [text]


def test_long_text_is_split(chunker):
    text = (
        "Employees may carry forward unused leave. "
        * 100
    )

    chunks = chunker.split_text(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunker.count_tokens(chunk) <= 256


def test_empty_text(chunker):
    assert chunker.split_text("") == []


def test_overlap_validation():
    with pytest.raises(ValueError):
        TokenAwareChunker(
            chunk_size=100,
            chunk_overlap=100
        )
