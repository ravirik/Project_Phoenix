"""The playbook collection must match the configured embedding size, or retrieval breaks."""
from types import SimpleNamespace

import pytest

from src.infrastructure import embeddings
from src.infrastructure.embeddings import (
    EMBEDDING_DIM,
    EmbeddingDimensionMismatch,
    embed_text,
    ensure_collection,
)


class FakeQdrant:
    def __init__(self, existing_size=None):
        self.existing_size = existing_size
        self.created_with = None

    def collection_exists(self, collection_name):
        return self.existing_size is not None

    def create_collection(self, collection_name, vectors_config):
        self.created_with = vectors_config.size

    def get_collection(self, collection_name):
        vectors = SimpleNamespace(size=self.existing_size)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))


def test_creates_missing_collection_with_configured_size():
    qdrant = FakeQdrant()
    ensure_collection(qdrant)
    assert qdrant.created_with == EMBEDDING_DIM


def test_accepts_matching_collection():
    qdrant = FakeQdrant(existing_size=EMBEDDING_DIM)
    ensure_collection(qdrant)
    assert qdrant.created_with is None


def test_rejects_collection_with_other_size():
    with pytest.raises(EmbeddingDimensionMismatch, match="3072"):
        ensure_collection(FakeQdrant(existing_size=3072))


def _fake_client(vector, calls):
    def embed_content(model, contents, config):
        calls.append((model, config.task_type, config.output_dimensionality))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vector)])
    return SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))


def test_embed_text_uses_shared_model_and_size(monkeypatch):
    calls = []
    client = _fake_client([0.0] * EMBEDDING_DIM, calls)
    monkeypatch.setattr(embeddings, "get_genai_client", lambda: client)

    assert len(embed_text("OOM crash", task_type="RETRIEVAL_QUERY")) == EMBEDDING_DIM
    assert calls == [(embeddings.EMBEDDING_MODEL, "RETRIEVAL_QUERY", EMBEDDING_DIM)]


def test_embed_text_rejects_wrong_size(monkeypatch):
    client = _fake_client([0.0] * 3072, [])
    monkeypatch.setattr(embeddings, "get_genai_client", lambda: client)

    with pytest.raises(EmbeddingDimensionMismatch):
        embed_text("OOM crash", task_type="RETRIEVAL_QUERY")
