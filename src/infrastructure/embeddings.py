"""Single source of truth for playbook embeddings.

Seeding, search and memory upserts must all use the same model and vector size,
otherwise Qdrant rejects the vectors and retrieval silently returns nothing.
"""
import os

from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config.llm import get_genai_client

# text-embedding-004 is retired; gemini-embedding-001 supports reduced output sizes.
EMBEDDING_MODEL = os.getenv("PHOENIX_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("PHOENIX_EMBEDDING_DIM", "768"))

COLLECTION_NAME = "remediation_playbooks"


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised when an existing Qdrant collection was built with a different vector size."""


def embed_text(text: str, task_type: str) -> list[float]:
    """Embeds text with the shared model. task_type is RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT."""
    response = get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM
        )
    )
    vector = response.embeddings[0].values
    if len(vector) != EMBEDDING_DIM:
        raise EmbeddingDimensionMismatch(
            f"{EMBEDDING_MODEL} returned {len(vector)} dims, expected {EMBEDDING_DIM}.")
    return vector


def ensure_collection(qdrant: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Creates the collection if missing, and fails loudly if its vector size has drifted."""
    if not qdrant.collection_exists(collection_name=collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        return

    existing_size = qdrant.get_collection(collection_name).config.params.vectors.size
    if existing_size != EMBEDDING_DIM:
        raise EmbeddingDimensionMismatch(
            f"Collection '{collection_name}' stores {existing_size}-dim vectors but "
            f"{EMBEDDING_MODEL} is configured for {EMBEDDING_DIM}. "
            f"Drop and re-seed the collection, or set PHOENIX_EMBEDDING_DIM={existing_size}.")
