
import numpy as np
from unittest.mock import Mock

from rag_pipeline.models import RankedCandidate
from rag_pipeline.retrieval_strategies.mmr_selector import (
    MMRSelector
)


def make_candidate(chunk_id, score):
    return RankedCandidate(
        chunk_id=chunk_id,
        document_id="handbook_001",
        text=f"Text {chunk_id}",
        source="handbook.pdf",
        page_number=1,
        score=score
    )


def test_mmr_prefers_diverse_chunks():
    candidates = [
        make_candidate("A", 0.95),
        make_candidate("B", 0.93),
        make_candidate("C", 0.85)
    ]

    # A and B are almost identical.
    # C is semantically different.
    mock_embedding_service = Mock()

    mock_embedding_service.embed_documents.return_value = (
        np.array([
            [1.0, 0.0],
            [0.99, 0.141],
            [0.0, 1.0]
        ], dtype=np.float32)
    )

    selector = MMRSelector(
        embedding_service=mock_embedding_service,
        lambda_param=0.5
    )

    results = selector.select(
        candidates=candidates,
        top_k=2
    )

    assert [r.chunk_id for r in results] == ["A", "C"]


def test_mmr_empty_candidates():
    selector = MMRSelector(
        embedding_service=Mock()
    )

    assert selector.select([]) == []
