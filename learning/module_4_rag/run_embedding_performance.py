
import time
import statistics

import torch
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZES = [1, 8, 16, 32, 64]

# Synthetic document-like passages for benchmarking.
# Avoid changing input text between experiments.
BASE_TEXTS = [
    (
        "Employees receive 24 working days of annual "
        "leave and may carry forward unused leave "
        "according to the company policy."
    ),
    (
        "Employees may work remotely two days per "
        "week with manager approval and must comply "
        "with the company's security requirements."
    ),
    (
        "Medical benefits include coverage for "
        "eligible employees according to the "
        "insurance plan and applicable policies."
    ),
    (
        "All employees must complete mandatory "
        "security training and follow the company "
        "information security guidelines."
    ),
]

DOCUMENTS = [
    BASE_TEXTS[i % len(BASE_TEXTS)]
    for i in range(256)
]


def synchronize_device(device: str):
    """Ensure queued GPU work finishes before timing."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_batch(model, batch_size, device):
    # Warm-up call
    model.encode_document(
        DOCUMENTS[:32],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    synchronize_device(device)

    durations = []

    for _ in range(3):
        start = time.perf_counter()

        embeddings = model.encode_document(
            DOCUMENTS,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        synchronize_device(device)

        elapsed = time.perf_counter() - start
        durations.append(elapsed)

    median_duration = statistics.median(durations)

    throughput = len(DOCUMENTS) / median_duration

    return {
        "batch_size": batch_size,
        "seconds": median_duration,
        "throughput": throughput,
        "shape": embeddings.shape,
    }


def benchmark_device(device):
    print(f"\n{'=' * 55}")
    print("DEVICE:", device)
    print(f"{'=' * 55}")

    model = SentenceTransformer(
        MODEL_ID,
        device=device
    )

    model.eval()

    for batch_size in BATCH_SIZES:
        result = benchmark_batch(
            model,
            batch_size,
            device
        )

        print(
            f"Batch {batch_size:<3} | "
            f"Time: {result['seconds']:.3f}s | "
            f"Throughput: {result['throughput']:.1f} "
            f"chunks/sec | "
            f"Shape: {result['shape']}"
        )

    # Interactive query latency: batch size 1
    query = "How many annual leave days are available?"

    model.encode_query(query)
    synchronize_device(device)

    query_times = []

    for _ in range(20):
        start = time.perf_counter()

        model.encode_query(
            query,
            normalize_embeddings=True
        )

        synchronize_device(device)

        query_times.append(
            (time.perf_counter() - start) * 1000
        )

    print(
        "Single query median latency:",
        round(statistics.median(query_times), 2),
        "ms"
    )

    del model


def main():
    print("CUDA:", torch.cuda.is_available())
    print("MPS:", torch.backends.mps.is_available())

    devices = ["cpu"]

    if torch.backends.mps.is_available():
        devices.append("mps")
    elif torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        benchmark_device(device)


if __name__ == "__main__":
    main()

