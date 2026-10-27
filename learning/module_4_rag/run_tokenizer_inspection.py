
from sentence_transformers import SentenceTransformer


def main():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    tokenizer = model.tokenizer

    text = (
        "Employees can carry forward up to 10 "
        "unused annual leave days under "
        "policy HR-2026-LV-043."
    )

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )

    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    print("Original text:")
    print(text)

    print("\nTokens:")
    print(tokens)

    print("\nToken IDs:")
    print(token_ids)

    print("\nNumber of tokens:")
    print(len(token_ids))

    print("\nModel max sequence length:")
    print(model.max_seq_length)

    print("\nEmbedding dimension:")
    print(
        model.get_embedding_dimension()
    )

    # samples = [
    #     "Employees receive 24 annual leave days.",
    #     "HR-2026-LV-043",
    #     "API_KEY=prod-service-2026-v2",
    #     "Employees can work remotely twice a week."
    # ]

    # for sample in samples:
    #     encoded = tokenizer(
    #         sample,
    #         add_special_tokens=True,
    #         truncation=False
    #     )

    #     print(
    #         f"Characters: {len(sample):3} | "
    #         f"Tokens: {len(encoded['input_ids']):3} | "
    #         f"{sample}"
    #     )

if __name__ == "__main__":
    main()
