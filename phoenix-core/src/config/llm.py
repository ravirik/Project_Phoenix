import os
from dotenv import load_dotenv
from crewai import LLM
from google import genai

load_dotenv()

PROJECT_ID = (os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip(' "\'')
LOCATION = (os.getenv("GCP_LOCATION") or "us-central1").strip(' "\'')

RAW_MODEL_NAME = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip(' "\'')
CLEAN_MODEL_NAME = RAW_MODEL_NAME.replace("vertex_ai/", "").replace("gemini/", "")


def get_crewai_llm() -> LLM:
    """
    Returns a unified CrewAI LLM instance.
    Increased max_tokens to 4096 to prevent script truncation in Node 4.
    """
    if PROJECT_ID:
        return LLM(
            model=f"vertex_ai/{CLEAN_MODEL_NAME}",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.0,
            max_tokens=4096
        )
    else:
        return LLM(
            model=f"gemini/{CLEAN_MODEL_NAME}",
            temperature=0.0,
            max_tokens=4096
        )


_genai_client = None


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        if PROJECT_ID:
            _genai_client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location=LOCATION
            )
        else:
            api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip(' "\'')
            if not api_key:
                raise ValueError("CRITICAL AUTH ERROR: Neither GCP_PROJECT_ID nor GEMINI_API_KEY is configured.")
            _genai_client = genai.Client(api_key=api_key)

    return _genai_client