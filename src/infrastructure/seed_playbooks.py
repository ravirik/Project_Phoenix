import os
import uuid
from dotenv import load_dotenv  # <-- 1. Add this import

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Tell Python to read existing .env file
load_dotenv()

# 1. Initialize Clients
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
qdrant = QdrantClient(host="127.0.0.1", port=6333)
COLLECTION_NAME = "remediation_playbooks"

# 2. Define Historical Playbooks (Our verified "Memory")
PLAYBOOKS = [
    {
        "incident_signature": "CPU utilization spiked to 99.9%. Schema drift and data outliers detected in streaming pipeline.",
        "remediation_code": "df = df.clip(upper=df.quantile(0.99), axis=1)",
        "explanation": "Caps numerical outliers at the 99th percentile to stabilize downstream ML models."
    },
    {
        "incident_signature": "OOM (Out of Memory) crash during bulk Parquet ingestion.",
        "remediation_code": "df = pd.read_parquet(uri, engine='pyarrow').dropna(how='all')",
        "explanation": "Drops fully null rows at the PyArrow engine level to reduce memory footprint before Pandas allocation."
    },
    {
        "incident_signature": "Type mismatch error: integer column receiving string values.",
        "remediation_code": "df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').fillna(-1).astype(int)",
        "explanation": "Forces numeric conversion, nullifying invalid strings, and safely casting back to integer."
    }
]


def seed_database():
    print("[Ingestion] Starting playbook vectorization...")
    points = []

    for playbook in PLAYBOOKS:
        # 3. Generate 768-dimensional embeddings using the modern syntax
        response = client.models.embed_content(
            # <-- Updated as text-embedding-004 has been retird or shut down
            model="gemini-embedding-001",
            contents=playbook["incident_signature"],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        # 4. Extract the raw vector array from the new response object
        vector = response.embeddings[0].values

        # 5. Construct the Qdrant Point (Vector + Metadata)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "incident": playbook["incident_signature"],
                "code": playbook["remediation_code"],
                "explanation": playbook["explanation"]
            }
        )
        points.append(point)
        print(
            f"[Ingestion] Embedded playbook: {playbook['incident_signature'][:40]}...")

    # 6. Load into Vector Database
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(
        f"\n[Ingestion] Successfully loaded {len(points)} playbooks into Qdrant.")


if __name__ == "__main__":
    seed_database()
