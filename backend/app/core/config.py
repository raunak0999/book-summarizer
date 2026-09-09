"""
Central configuration. All secrets/env-driven values live here so the
rest of the codebase never reads os.environ directly (easier to test,
easier to swap providers later).
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- Database ---
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/bookagent"

    # --- Auth ---
    jwt_secret: str = "change-me-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 1 day

    # --- LLM / Embedding providers: "groq", "gemini", "azure_openai", "github_models", "aws_bedrock", "local", "openai" ---
    llm_provider: str = "gemini"
    chat_provider: str = "gemini"
    embedding_provider: str = "gemini"


    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment_chat: str = "gpt-4o-mini"
    azure_openai_deployment_embedding: str = "text-embedding-3-small"
    azure_openai_api_version: str = "2024-08-01-preview"

    # GitHub Models
    github_models_token: str = ""
    github_models_endpoint: str = "https://models.inference.ai.azure.com"
    github_models_chat_model: str = "gpt-4o-mini"
    github_models_embedding_model: str = "text-embedding-3-small"

    # Groq
    groq_api_key: str = ""
    groq_chat_model: str = "openai/gpt-oss-20b"

    groq_endpoint: str = "https://api.groq.com/openai/v1"

    # Gemini
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-001"


    # AWS Bedrock (alternative)
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # --- Chunking / RAG / Embeddings ---
    local_embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    top_k_retrieval: int = 5

    embedding_dim: int = 768


    # --- Observability ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "book-agent"

    # --- Uploads ---
    max_upload_mb: int = 60
    upload_dir: str = "/tmp/book-agent-uploads"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
