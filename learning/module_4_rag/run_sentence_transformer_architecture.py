
from pyexpat import model

import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # 1. Load model
    model = SentenceTransformer(MODEL_ID)
    model.eval()

    print("\nMODEL ARCHITECTURE")
    print(model)

    print("\nEMBEDDING DIMENSION")
    print(model.get_sentence_embedding_dimension())

    print("\nMAX SEQUENCE LENGTH")
    print(model.max_seq_length)

    print("\nMODEL DEVICE")
    print(model.device)

    text = "Employees receive 24 annual leave days."

    # 2. Tokenize
    features = model.tokenize([text])

    print("\nTOKENIZED INPUT")
    print("Input IDs shape:", features["input_ids"].shape)
    print("Attention mask:", features["attention_mask"])

    # 3. Move inputs to the model's device
    # Important for Apple MPS / CUDA
    features = batch_to_device(features, model.device)

    print("\nDEVICE VERIFICATION")
    print("Model device:", model.device)
    print("Input device:", features["input_ids"].device)

    # 4. Run Transformer
    transformer = model[0]

    with torch.no_grad():
        output = transformer(features)

    token_embeddings = output["token_embeddings"]

    print("\nTRANSFORMER OUTPUT")
    print("Token embeddings:", token_embeddings.shape)

    # 5. Apply pooling
    pooling = model[1]

    with torch.no_grad():
        pooled_output = pooling(output)

    sentence_embedding = pooled_output["sentence_embedding"]

    print("\nAFTER POOLING")
    print("Sentence embedding:", sentence_embedding.shape)

    # 6. L2 normalization
    normalized_embedding = torch.nn.functional.normalize(
        sentence_embedding,
        p=2,
        dim=1
    )

    print("\nAFTER NORMALIZATION")
    print("Shape:", normalized_embedding.shape)
    print(
        "Vector magnitude:",
        torch.linalg.vector_norm(
            normalized_embedding,
            dim=1
        ).item()
    )

    # 7. Compare with SentenceTransformer API
    final_embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    print("\nFINAL EMBEDDING USING model.encode()")
    print("Shape:", final_embedding.shape)
    print("First five values:", final_embedding[:5])

    # 8. Verify manual pipeline matches encode()
    is_same = torch.allclose(
        normalized_embedding.squeeze(0).cpu(),
        final_embedding.cpu(),
        atol=1e-5
    )

    print("\nMANUAL PIPELINE MATCHES encode():", is_same)


if __name__ == "__main__":
    main()
