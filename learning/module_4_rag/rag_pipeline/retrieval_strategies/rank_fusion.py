
from rag_pipeline.models import RankedCandidate


class ReciprocalRankFusion:

    def __init__(self, rank_constant: int = 60):
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive")

        self.rank_constant = rank_constant

    def fuse(
        self,
        result_lists: list[list],
        top_k: int = 5
    ) -> list[RankedCandidate]:

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        fused_scores = {}
        candidate_map = {}

        for results in result_lists:
            seen_in_list = set()

            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id

                # Prevent duplicate contributions from
                # the same retriever.
                if chunk_id in seen_in_list:
                    continue

                seen_in_list.add(chunk_id)

                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0.0)
                    + 1.0 / (self.rank_constant + rank)
                )

                if chunk_id not in candidate_map:
                    candidate_map[chunk_id] = result

        ranked_ids = sorted(
            fused_scores,
            key=lambda cid: fused_scores[cid],
            reverse=True
        )[:top_k]

        return [
            RankedCandidate(
                chunk_id=cid,
                document_id=candidate_map[cid].document_id,
                text=candidate_map[cid].text,
                source=candidate_map[cid].source,
                page_number=candidate_map[cid].page_number,
                score=fused_scores[cid]
            )
            for cid in ranked_ids
        ]
