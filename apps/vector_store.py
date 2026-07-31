from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DOCUMENTS_FILE = Path(__file__).parent / "documents.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBEDDING_MODEL)


def load_documents(path: Path = DOCUMENTS_FILE) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


documents = load_documents()
document_embeddings = model.encode(documents, normalize_embeddings=True)


def search(query: str, top_k: int = 3) -> list[str]:
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = document_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]
