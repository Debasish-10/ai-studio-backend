
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderRerankingEvaluator,
)

from model_eval_dataset import (
    QUERIES,
    CORPUS,
    RELEVANT_DOCS,
)


EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


def build_retrieval_samples(top_k=5):
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    corpus_ids = list(CORPUS.keys())
    corpus_texts = [
        CORPUS[doc_id] for doc_id in corpus_ids
    ]

    document_embeddings = embedder.encode_document(
        corpus_texts,
        normalize_embeddings=True
    )

    samples = []

    for query_id, query in QUERIES.items():
        query_embedding = embedder.encode_query(
            query,
            normalize_embeddings=True
        )

        scores = (
            document_embeddings @ query_embedding
        )

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        retrieved_docs = [
            corpus_texts[i] for i in ranked_indices
        ]

        positive_docs = [
            CORPUS[doc_id]
            for doc_id in RELEVANT_DOCS[query_id]
        ]

        samples.append({
            "query": query,
            "positive": positive_docs,
            "documents": retrieved_docs,
        })

    return samples


def main():
    # Actual first-stage retrieval results.
    samples = build_retrieval_samples(top_k=5)

    reranker = CrossEncoder(RERANKER_MODEL)

    evaluator = CrossEncoderRerankingEvaluator(
        samples=samples,
        name="novatech-reranking-demo",
        at_k=5,
        mrr_at_k=5,
        batch_size=8,
        always_rerank_positives=False,
        write_csv=False,
    )

    results = evaluator(reranker)

    print("\nCROSS-ENCODER EVALUATION")

    for metric_name, value in sorted(results.items()):
        print(f"{metric_name}: {value:.4f}")

    print("\nPRIMARY METRIC")
    print(evaluator.primary_metric)
    print(results[evaluator.primary_metric])


if __name__ == "__main__":
    main()
