"""
RAG Pipeline Configuration

Centralized configuration for the enhanced RAG pipeline.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenSearchConfig(BaseModel):
    """OpenSearch vector store configuration."""
    endpoint: str = Field(default_factory=lambda: os.getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200"))
    user: str = Field(default_factory=lambda: os.getenv("OPENSEARCH_USER", "admin"))
    password: str = Field(default_factory=lambda: os.getenv("OPENSEARCH_PASSWORD", "admin"))
    general_index: str = Field(default_factory=lambda: os.getenv("OPENSEARCH_GENERAL_INDEX", "isaac_documents"))
    user_docs_index: str = Field(default_factory=lambda: os.getenv("OPENSEARCH_USER_DOCS_INDEX", "isaac_user_documents"))
    citations_index: str = Field(default_factory=lambda: os.getenv("OPENSEARCH_CITATIONS_INDEX", "isaac_citations"))
    
    # Vector search settings
    embedding_dimension: int = 1536  # OpenAI text-embedding-3-small
    default_top_k: int = 5


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.1


class SemanticScholarConfig(BaseModel):
    """Semantic Scholar API configuration."""
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    rate_limit_requests: int = 100  # requests per 5 minutes
    timeout: int = 30


class RedisConfig(BaseModel):
    """Redis cache configuration."""
    enabled: bool = Field(default_factory=lambda: os.getenv("REDIS_URL") is not None)
    url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    ttl: int = 3600  # 1 hour default TTL


class RAGConfig(BaseModel):
    """Main RAG pipeline configuration."""
    opensearch: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    semantic_scholar: SemanticScholarConfig = Field(default_factory=SemanticScholarConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    
    # Document processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_context_length: int = 50000
    
    # Search settings
    hybrid_alpha: float = 0.5  # Weight between vector (alpha) and BM25 (1-alpha)
    rerank_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/rag_pipeline.log"
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        return cls(
            opensearch=OpenSearchConfig(),
            openai=OpenAIConfig(),
            semantic_scholar=SemanticScholarConfig(),
            redis=RedisConfig(),
        )


# Global config instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config


def set_config(config: RAGConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
