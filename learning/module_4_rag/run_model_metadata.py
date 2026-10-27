
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    api = HfApi()

    # Hub metadata; no need to load the weights.
    info = api.model_info(MODEL_ID)

    print("Model:", info.id)
    print("Revision:", info.sha)
    print("Downloads:", info.downloads)
    print("License:", (info.cardData or {}).get("license"))

    print("\nLoading model...")
    model = SentenceTransformer(MODEL_ID)

    print("Embedding dimension:",
          model.get_embedding_dimension())

    print("Maximum sequence length:",
          model.max_seq_length)

    print("Similarity function:",
          model.similarity_fn_name)

    print("Model modules:")
    print(model)

    print("Tokenizer vocabulary size:",
          len(model.tokenizer))


if __name__ == "__main__":
    main()
