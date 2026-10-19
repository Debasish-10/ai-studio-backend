
from rag_pipeline.models import RankedCandidate
from rag_pipeline.retrieval_strategies.rank_fusion import (
    ReciprocalRankFusion
)


def make_candidate(chunk_id):
    return RankedCandidate(
        chunk_id=chunk_id,
        document_id="handbook_001",
        text=f"Text for {chunk_id}",
        source="handbook.pdf",
        page_number=1,
        score=0.0
    )


def test_rrf_combines_rankings():
    semantic = [
        make_candidate("A"),
        make_candidate("B"),
        make_candidate("C")
    ]

    bm25 = [
        make_candidate("B"),
        make_candidate("D"),
        make_candidate("A")
    ]

    fusion = ReciprocalRankFusion(rank_constant=60)

    results = fusion.fuse(
        [semantic, bm25],
        top_k=4
    )

    assert len(results) == 4
    assert results[0].chunk_id == "B"
    assert results[1].chunk_id == "A"
    assert len({item.chunk_id for item in results}) == 4
