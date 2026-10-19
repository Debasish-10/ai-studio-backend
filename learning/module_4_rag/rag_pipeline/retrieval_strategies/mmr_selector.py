
import numpy as np

from rag_pipeline.models import RankedCandidate


class MMRSelector:

    def __init__(
        self,
        embedding_service,
        lambda_param: float = 0.5
    ):
        if not 0 <= lambda_param <= 1:
            raise ValueError(
                "lambda_param must be between 0 and 1"
            )

        self.embedding_service = embedding_service
        self.lambda_param = lambda_param

    def select(
        self,
        candidates: list[RankedCandidate],
        top_k: int = 5
    ) -> list[RankedCandidate]:

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not candidates:
            return []

        # Expect candidates sorted by reranker score.
        candidates = sorted(
            candidates,
            key=lambda c: c.score,
            reverse=True
        )

        # Step 1: Generate normalized chunk embeddings.
        embeddings = np.asarray(
            self.embedding_service.embed_documents(
                [c.text for c in candidates]
            ),
            dtype=np.float32
        )

        # Our embedding service normalizes vectors.
        # Therefore, dot product = cosine similarity.
        similarity_matrix = embeddings @ embeddings.T

        n = len(candidates)

        # Step 2: Convert reranker positions to [0, 1].
        if n == 1:
            relevance = np.array([1.0])
        else:
            relevance = np.array([
                1.0 - i / (n - 1)
                for i in range(n)
            ])

        # Step 3: Select highest-relevance chunk first.
        selected = [0]
        remaining = set(range(1, n))

        # Step 4: Iteratively select diverse chunks.
        while remaining and len(selected) < top_k:

            best_index = None
            best_score = float("-inf")

            for index in sorted(remaining):
                max_similarity = max(
                    similarity_matrix[index][chosen]
                    for chosen in selected
                )

                mmr_score = (
                    self.lambda_param * relevance[index]
                    - (1 - self.lambda_param)
                    * max_similarity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index

            selected.append(best_index)
            remaining.remove(best_index)

        return [candidates[i] for i in selected]
