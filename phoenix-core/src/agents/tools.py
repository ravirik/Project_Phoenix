import os
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Initialize production clients
client = genai.Client(
    vertexai=True,
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_LOCATION", "us-central1")
)
qdrant = QdrantClient(host="127.0.0.1", port=6333)

# Define the strict contract schema for the LLM


class PlaybookInput(BaseModel):
    incident_signature: str = Field(
        description="The exact text signature of the error or drift metric encountered (e.g., 'Memory spiked during Parquet file read')."
    )


@tool("search_historical_playbooks", args_schema=PlaybookInput)
def search_historical_playbooks(incident_signature: str) -> str:
    """
    Searches the historical incident database for verified remediation code.
    Use this tool EVERY TIME an anomaly is detected to find past solutions.
    """
    print(
        f"\n[Tool Execution] Embedding search query: '{incident_signature[:40]}...'")

    # 1. Convert the incoming alert into a 3072-dimension vector
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=incident_signature,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )
    query_vector = response.embeddings[0].values

    # 2. Perform Cosine Similarity Search in Qdrant (v1.9.0 syntax)
    search_results = qdrant.search(
        collection_name="remediation_playbooks",
        query_vector=query_vector,
        limit=1  # Pull the highest confidence playbook match
    )

    # 3. Format the response for the LLM
    if not search_results:
        return "No historical playbooks found for this specific incident signature."

    best_match = search_results[0].payload

    formatted_context = f"""
    VERIFIED HISTORICAL FIX FOUND:
    - Incident Match: {best_match.get('incident', 'Unknown')}
    - Remediation Code: {best_match.get('code', '')}
    - Explanation: {best_match.get('explanation', '')}
    """

    return formatted_context


if __name__ == "__main__":
    # Standalone verification test using schema structure
    result = search_historical_playbooks.invoke(
        {"incident_signature": "Memory spiked during Parquet file read."}
    )
    print(result)
