from qdrant_client import QdrantClient
from qdrant_client.http import models


class PlaybookVectorStore:
    def __init__(self, host: str = "127.0.0.1", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "remediation_playbooks"

        # Updated this from 768 to 3072 to match the new model
        self.vector_size = 3072

    def initialize_schema(self):
        """Idempotent setup: ensures the collection exists without overwriting."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            print(
                f"[Vector Store] Initializing new collection: '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print("[Vector Store] Schema created successfully.")
        else:
            print(
                f"[Vector Store] Collection '{self.collection_name}' is ready.")


if __name__ == "__main__":
    # Standalone test execution
    store = PlaybookVectorStore()
    store.initialize_schema()
