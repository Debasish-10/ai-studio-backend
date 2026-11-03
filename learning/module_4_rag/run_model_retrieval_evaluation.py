
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
)

from model_eval_dataset import (
    QUERIES,
    CORPUS,
    RELEVANT_DOCS,
)


MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]


def create_evaluator():
    return InformationRetrievalEvaluator(
        queries=QUERIES,
        corpus=CORPUS,
        relevant_docs=RELEVANT_DOCS,
        name="novatech-policy-demo",
        mrr_at_k=[3, 5],
        ndcg_at_k=[3, 5],
        accuracy_at_k=[1, 3, 5],
        precision_recall_at_k=[1, 3, 5],
        map_at_k=[5],
        batch_size=16,
        show_progress_bar=False,
        write_csv=False,
        main_score_function="cosine",
    )


def main():
    evaluator = create_evaluator()

    print("Queries:", len(QUERIES))
    print("Corpus passages:", len(CORPUS))

    for model_id in MODELS:
        print("\n" + "=" * 65)
        print("MODEL:", model_id)
        print("=" * 65)

        model = SentenceTransformer(model_id)

        # Built-in evaluation: no manual retrieval loop needed.
        results = evaluator(model)

        for metric_name, score in sorted(results.items()):
            print(f"{metric_name}: {score:.4f}")

        print("\nPRIMARY METRIC")
        print(evaluator.primary_metric)
        print(results[evaluator.primary_metric])


if __name__ == "__main__":
    main()
