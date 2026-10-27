
import re
import time
from dataclasses import dataclass

import fitz
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from chunking_eval_dataset import EVAL_CASES


PDF_PATH = "sample_docs/employee_handbook.pdf"

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-small-en-v1.5"


@dataclass
class ExperimentConfig:
    name: str
    model_id: str
    strategy: str
    chunk_size: int
    chunk_overlap: int


CONFIGS = [
    ExperimentConfig(
        "A_char_500",
        MINILM,
        "character",
        500,
        100
    ),
    ExperimentConfig(
        "B_token_120",
        MINILM,
        "token",
        120,
        24
    ),
    ExperimentConfig(
        "C_token_200",
        MINILM,
        "token",
        200,
        40
    ),
    ExperimentConfig(
        "D_bge_token_200",
        BGE,
        "token",
        200,
        40
    ),
    ExperimentConfig(
        "E_bge_token_200",
        BGE,
        "token",
        120,
        24
    )
]


def load_pages(path: str) -> list[str]:
    with fitz.open(path) as pdf:
        return [
            page.get_text()
            for page in pdf
        ]


def create_chunks(pages, config, tokenizer):

    if config.strategy == "character":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    else:
        splitter = (
            RecursiveCharacterTextSplitter
            .from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        )

    chunks = []

    # Keep identical page extraction for all runs.
    for page_text in pages:
        if page_text.strip():
            chunks.extend(
                splitter.split_text(page_text)
            )

    return chunks


def token_count(text, tokenizer):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )


def contains_evidence(text, patterns):
    normalized = " ".join(
        text.lower().split()
    )

    return all(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def evaluate_retrieval(
    model,
    chunks,
    document_embeddings
):

    hit_at_3 = []
    reciprocal_ranks = []
    query_times = []
    uncovered = []

    for case in EVAL_CASES:

        # Label relevant evidence for this chunking.
        relevant_indices = {
            i
            for i, chunk in enumerate(chunks)
            if contains_evidence(
                chunk,
                case["evidence_patterns"]
            )
        }

        if not relevant_indices:
            uncovered.append(case["id"])

        start = time.perf_counter()

        query_embedding = model.encode_query(
            case["query"],
            normalize_embeddings=True
        )

        # Exact cosine ranking for normalized vectors.
        similarities = (
            document_embeddings @ query_embedding
        )

        top_indices = np.argsort(
            similarities
        )[::-1][:5]

        elapsed = time.perf_counter() - start
        query_times.append(elapsed)

        first_relevant_rank = None

        for rank, index in enumerate(top_indices, 1):
            if int(index) in relevant_indices:
                first_relevant_rank = rank
                break

        hit_at_3.append(
            first_relevant_rank is not None
            and first_relevant_rank <= 3
        )

        reciprocal_ranks.append(
            1 / first_relevant_rank
            if first_relevant_rank
            else 0.0
        )

        print(
            f"  {case['id']:18} "
            f"first relevant rank: "
            f"{first_relevant_rank or 'not found'}"
        )

    return {
        "hit_at_3": np.mean(hit_at_3),
        "mrr_at_5": np.mean(reciprocal_ranks),
        "mean_query_ms": (
            np.mean(query_times) * 1000
        ),
        "uncovered": uncovered
    }


def main():
    pages = load_pages(PDF_PATH)

    # Reuse each loaded model across its experiments.
    model_cache = {}

    for config in CONFIGS:
        print("\n" + "=" * 65)
        print("EXPERIMENT:", config.name)
        print("=" * 65)

        if config.model_id not in model_cache:
            model_cache[config.model_id] = (
                SentenceTransformer(config.model_id)
            )

        model = model_cache[config.model_id]
        tokenizer = model.tokenizer

        chunks = create_chunks(
            pages,
            config,
            tokenizer
        )

        lengths = [
            token_count(c, tokenizer)
            for c in chunks
        ]

        oversized = sum(
            n > model.max_seq_length
            for n in lengths
        )

        print("Model:", config.model_id)
        print("Chunks:", len(chunks))
        print("Average tokens:", round(np.mean(lengths), 2))
        print("Max tokens:", max(lengths))
        print("Above model limit:", oversized)

        start = time.perf_counter()

        document_embeddings = model.encode_document(
            chunks,
            normalize_embeddings=True,
            batch_size=32
        )

        embedding_seconds = (
            time.perf_counter() - start
        )

        metrics = evaluate_retrieval(
            model,
            chunks,
            document_embeddings
        )

        print("\nRESULTS")
        print("Hit@3:", round(metrics["hit_at_3"], 3))
        print("MRR@5:", round(metrics["mrr_at_5"], 3))
        print(
            "Mean query time (ms):",
            round(metrics["mean_query_ms"], 2)
        )
        print(
            "Embedding time (sec):",
            round(embedding_seconds, 2)
        )
        print(
            "Questions without labeled evidence:",
            metrics["uncovered"]
        )


if __name__ == "__main__":
    main()
