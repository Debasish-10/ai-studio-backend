
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Our small document collection
documents = [
    "Employees receive 24 days of annual leave.",
    "Employees can work remotely twice per week.",
    "The company provides health insurance.",
    "Employees can carry forward unused leave.",
    "The office cafeteria serves lunch."
]


# 3. Convert documents into embeddings
document_embeddings = model.encode(documents)

print("Document embeddings:", document_embeddings.shape)


# 4. User query
query = "What medical benefits does the company provide?"

# Generate query embedding using the same model
query_embedding = model.encode(query)


# 5. Cosine similarity function
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    return float(dot_product / (magnitude_a * magnitude_b))


# 6. Calculate similarity for every document
results = []

for index, document_embedding in enumerate(document_embeddings):
    score = cosine_similarity(
        query_embedding,
        document_embedding
    )

    results.append({
        "document": documents[index],
        "score": score
    })


# 7. Sort documents by similarity
results.sort(
    key=lambda result: result["score"],
    reverse=True
)


# 8. Retrieve top 3 documents
top_k = 3

print("\nTop matching documents:\n")

for result in results[:top_k]:
    print(
        f"Score: {result['score']:.4f} | "
        f"Document: {result['document']}"
    )
