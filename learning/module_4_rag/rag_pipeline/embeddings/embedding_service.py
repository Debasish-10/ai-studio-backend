import numpy as np
from sentence_transformers import SentenceTransformer

from rag_pipeline.config import EMBEDDING_MODEL


class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def embed_documents(
        self,
        texts: list[str]
    ) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    def embed_query(self, query: str) -> np.ndarray:

        embeddings = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )