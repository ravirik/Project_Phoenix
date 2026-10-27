import os
import uuid
from dotenv import load_dotenv
from crewai.tools import tool
from pydantic import BaseModel, Field
import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.infrastructure.embeddings import (
    COLLECTION_NAME,
    EmbeddingDimensionMismatch,
    embed_text,
    ensure_collection,
)

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL", "http://phoenix-qdrant-vector:6333")
qdrant = QdrantClient(url=qdrant_url)


@tool("search_historical_playbooks")
def search_historical_playbooks(incident_signature: str) -> str:
    """
    Searches historical incident database for verified remediation code.
    Pass failure logs or incident signature as incident_signature string.
    """
    print(
        f"\n[Tool Execution] Embedding query via GenAI: '{incident_signature[:40]}...'")

    try:
        if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
            return "No historical playbooks found (Vector collection empty or not initialized yet)."

        ensure_collection(qdrant)
        query_vector = embed_text(incident_signature, task_type="RETRIEVAL_QUERY")

        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=1
        )

        search_results = search_response.points
        if not search_results:
            return "No historical playbooks found for this specific incident signature."

        best_match = search_results[0].payload
        return f"""
        VERIFIED HISTORICAL FIX FOUND:
        - Incident Match: {best_match.get('incident', 'Unknown')}
        - Remediation Code: {best_match.get('code', '')}
        - Explanation: {best_match.get('explanation', '')}
        """

    except EmbeddingDimensionMismatch as e:
        # A config error, not an empty memory: surface it instead of letting the agent guess.
        logfire.error("Playbook retrieval misconfigured: {error}", error=str(e))
        return f"ERROR: playbook memory is misconfigured and was not searched. {e}"
    except Exception as e:
        logfire.error("Playbook retrieval failed: {error}", error=str(e))
        return f"No historical playbooks retrieved. Exception: {str(e)}"


def upsert_playbook(incident_signature: str, code_patch: str) -> None:
    """Commits a verified remediation patch to long-term vector memory."""
    ensure_collection(qdrant)
    vector = embed_text(incident_signature, task_type="RETRIEVAL_DOCUMENT")

    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "incident": incident_signature,
                    "code": code_patch,
                    "explanation": "Self-healed by Phoenix Orchestrator"
                }
            )
        ]
    )
    print(
        f"[Memory Pipeline] ✅ Successfully committed new playbook {point_id[:8]} to Qdrant.")
