
from time import perf_counter
from statistics import mean

from evaluation_dataset import EVALUATION_DATASET

from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.vectorstores.chroma_store import ChromaStore
from rag_pipeline.retrieval import RetrievalService

from rag_pipeline.retrieval_strategies.bm25_retriever import BM25Retriever
from rag_pipeline.retrieval_strategies.rank_fusion import ReciprocalRankFusion
from rag_pipeline.retrieval_strategies.reranker import CrossEncoderReranker
from rag_pipeline.retrieval_strategies.mmr_selector import MMRSelector
from rag_pipeline.retrieval_strategies.query_processor import QueryProcessor
from rag_pipeline.retrieval_strategies.retrieval_router import RetrievalRouter
from rag_pipeline.retrieval_strategies.adaptive_retrieval import (
    AdaptiveRetrievalService
)

from rag_pipeline.evaluation.retrieval_metrics import (
    evaluate_retrieval
)


def evaluate_pipeline():
    filters = {"document_id": "handbook_001"}
    k = 5

    store = ChromaStore()
    chunks = store.get_chunks(filters=filters)

    if not chunks:
        raise RuntimeError("No indexed chunks available")

    valid_ids = {chunk["id"] for chunk in chunks}

    for case in EVALUATION_DATASET:
        relevant = set(case["relevant_chunk_ids"])

        if not relevant or not relevant.issubset(valid_ids):
            raise ValueError(
                f"Invalid gold labels for {case['id']}. "
                "Inspect chunks and replace placeholder IDs."
            )

    # Initialize models before timing requests.
    embeddings = EmbeddingService()

    dense = RetrievalService(
        embedding_service=embeddings,
        vector_store=store
    )

    adaptive = AdaptiveRetrievalService(
        dense=dense,
        bm25=BM25Retriever(chunks),
        fusion=ReciprocalRankFusion(),
        reranker=CrossEncoderReranker(),
        mmr=MMRSelector(
            embedding_service=embeddings,
            lambda_param=0.5
        ),
        router=RetrievalRouter()
    )

    processor = QueryProcessor()

    scores = {
        "semantic": [],
        "adaptive": []
    }

    latencies = {
        "semantic": [],
        "adaptive": []
    }

    # Warm up inference outside the benchmark.
    embeddings.embed_query("warmup")
    # Cross-encoder cold start should also be warmed up
    # separately for rigorous latency benchmarking.

    for case in EVALUATION_DATASET:
        query = case["query"]
        relevant_ids = set(case["relevant_chunk_ids"])

        print(f"\nQUESTION: {query}")

        # A. Semantic baseline
        start = perf_counter()

        semantic_results = dense.retrieve(
            query=query,
            top_k=k,
            similarity_threshold=0.0,
            filters=filters
        )

        semantic_ms = (perf_counter() - start) * 1000

        # B. Adaptive pipeline
        start = perf_counter()

        rewritten = processor.rewrite(query)

        plan, adaptive_results = adaptive.retrieve(
            query=rewritten,
            filters=filters,
            top_k=k
        )

        adaptive_ms = (perf_counter() - start) * 1000

        semantic_ids = [
            item.chunk_id for item in semantic_results
        ]

        adaptive_ids = [
            item.chunk_id for item in adaptive_results
        ]

        semantic_metrics = evaluate_retrieval(
            semantic_ids, relevant_ids, k
        )

        adaptive_metrics = evaluate_retrieval(
            adaptive_ids, relevant_ids, k
        )

        scores["semantic"].append(semantic_metrics)
        scores["adaptive"].append(adaptive_metrics)

        latencies["semantic"].append(semantic_ms)
        latencies["adaptive"].append(adaptive_ms)

        print("Selected strategy:", plan.strategy)
        print("Gold IDs:", sorted(relevant_ids))
        print("Semantic IDs:", semantic_ids)
        print("Adaptive IDs:", adaptive_ids)

        print("Semantic metrics:", semantic_metrics)
        print("Adaptive metrics:", adaptive_metrics)

    print("\n===== FINAL EVALUATION =====")

    for name in ["semantic", "adaptive"]:
        print(f"\n{name.upper()}")

        for metric in ["hit", "recall", "mrr"]:
            avg = mean(
                case[metric] for case in scores[name]
            )

            print(f"{metric}@{k}: {avg:.3f}")

        print(
            "Average latency (ms):",
            round(mean(latencies[name]), 2)
        )


if __name__ == "__main__":
    evaluate_pipeline()
