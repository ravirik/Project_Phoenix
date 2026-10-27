from qdrant_client import QdrantClient

from src.infrastructure.embeddings import COLLECTION_NAME, EMBEDDING_DIM, ensure_collection


class PlaybookVectorStore:
    def __init__(self, host: str = "127.0.0.1", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = COLLECTION_NAME
        self.vector_size = EMBEDDING_DIM

    def initialize_schema(self):
        """Idempotent setup: ensures the collection exists without overwriting."""
        exists = self.client.collection_exists(collection_name=self.collection_name)
        ensure_collection(self.client, self.collection_name)
        if exists:
            print(
                f"[Vector Store] Collection '{self.collection_name}' is ready.")
        else:
            print(
                f"[Vector Store] Created '{self.collection_name}' ({self.vector_size} dims).")


if __name__ == "__main__":
    # Standalone test execution
    store = PlaybookVectorStore()
    store.initialize_schema()
