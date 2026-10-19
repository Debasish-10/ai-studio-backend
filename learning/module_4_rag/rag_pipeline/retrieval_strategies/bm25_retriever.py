
import re

from rank_bm25 import BM25Okapi

from rag_pipeline.models import BM25Result


def tokenize(text: str) -> list[str]:
    # Keep identifiers such as HR-2026-LV-042 together.
    return re.findall(
        r"[a-z0-9]+(?:[-_][a-z0-9]+)*",
        text.lower()
    )


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = [
            chunk
            for chunk in chunks
            if tokenize(chunk["text"])
        ]

        if not self.chunks:
            raise ValueError(
                "Cannot build BM25 index with empty chunks."
            )

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.index = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[BM25Result]:

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        print("Query tokens:", query_tokens)
        print("BM25 scores:", scores)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            # Ignore zero/negative lexical scores.
            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append(
                BM25Result(
                    chunk_id=chunk["id"],
                    document_id=chunk["document_id"],
                    text=chunk["text"],
                    source=chunk["source"],
                    page_number=chunk["page"],
                    score=score
                )
            )

            if len(results) >= top_k:
                break

        return results
