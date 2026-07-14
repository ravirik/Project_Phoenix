import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

def get_llm() -> BaseChatModel:
    """
    Factory method to dynamically inject the chosen LLM provider.
    Ensures zero vendor lock-in for the Agentic RAG pipeline.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Using gemini-1.5-flash for high-speed, cost-effective reasoning
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.0,
            api_key=os.getenv("GEMINI_API_KEY")
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        # DeepSeek uses the OpenAI SDK format with a custom base URL
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
