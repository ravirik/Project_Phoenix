import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# FORCE dotenv to overwrite any stale cached variables in the terminal session
load_dotenv(override=True)

def get_llm() -> BaseChatModel:
    """Factory method dynamically injecting the chosen LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        safe_key = api_key[:7] + "..." if api_key else "NOT_FOUND"
        print(f"\n[Infrastructure] Booting Gemini API. Key signature: {safe_key}")
        print("[Infrastructure] Utilizing model pointer: gemini-flash-latest")
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0.0,
            api_key=api_key
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="deepseek-chat", 
            temperature=0.0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama3", temperature=0.0)
        
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama3-8b-8192", temperature=0.0)
        
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Check your .env file.")
