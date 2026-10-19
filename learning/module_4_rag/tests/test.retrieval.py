
from unittest.mock import Mock

from rag_pipeline.retrieval import RetrievalService
from rag_pipeline.models import RetrievalResult


def test_retrieval_applies_similarity_threshold():

    fake_embeddings = Mock()
    fake_embeddings.embed_query.return_value = [[0.1, 0.2]]

    fake_store = Mock()

    fake_store.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_1",
            document_id="handbook_001",
            text="Annual leave is 24 days.",
            source="handbook.pdf",
            page_number=1,
            similarity=0.85
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            document_id="handbook_001",
            text="Office cafeteria menu.",
            source="handbook.pdf",
            page_number=3,
            similarity=0.30
        )
    ]

    service = RetrievalService(
        embedding_service=fake_embeddings,
        vector_store=fake_store
    )

    results = service.retrieve(
        query="How many annual leave days?",
        top_k=5,
        similarity_threshold=0.50,
        document_id="handbook_001"
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_1"

    fake_store.search.assert_called_once_with(
        query_embedding=[[0.1, 0.2]],
        top_k=5,
        document_id="handbook_001"
    )
