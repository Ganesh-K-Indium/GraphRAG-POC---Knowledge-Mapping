"""
Centralized configuration for the GraphRAG POC application.
All settings are read from environment variables with sensible defaults.
Uses python-dotenv to load from a .env file if present.
"""

# SSL bypass MUST run before any HTTPS-capable library initialises.
# The patch is defined once in core/ssl_patch.py and imported here.
import core.ssl_patch  # noqa: F401

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    """Immutable settings object. Instantiate once and reuse."""

    # --- OpenAI / LLM ---
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o")
    )
    max_tokens_extraction: int = field(
        default_factory=lambda: int(os.environ.get("MAX_TOKENS_EXTRACTION", "4000"))
    )

    # --- Embeddings ---
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_DIMENSION", "1024"))
    )

    # --- Qdrant Vector Store ---
    qdrant_host: str = field(
        default_factory=lambda: os.environ.get("QDRANT_HOST", "localhost")
    )
    qdrant_port: int = field(
        default_factory=lambda: int(os.environ.get("QDRANT_PORT", "6333"))
    )
    qdrant_collection: str = field(
        default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "graphrag_elements")
    )
    qdrant_api_key: str = field(
        default_factory=lambda: os.environ.get("QDRANT_API_KEY", "")
    )

    # --- Neo4j Graph Store ---
    neo4j_uri: str = field(
        default_factory=lambda: os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.environ.get("NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.environ.get("NEO4J_PASSWORD", "password")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.environ.get("NEO4J_DATABASE", "neo4j")
    )

    # --- PostgreSQL (workspace metadata) ---
    postgres_url: str = field(
        default_factory=lambda: os.environ.get(
            "POSTGRES_URL",
            "postgresql://graphrag:graphrag@localhost:5432/graphrag",
        )
    )

    # --- Quality thresholds ---
    confidence_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
    )

    # --- Logging ---
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )

    # --- Document processing ---
    max_chunk_chars: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CHUNK_CHARS", "3000"))
    )
    chunk_overlap_chars: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP_CHARS", "200"))
    )

    def validate(self) -> None:
        """Raise ValueError for any critical mis-configuration."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"CONFIDENCE_THRESHOLD must be between 0 and 1, got {self.confidence_threshold}"
            )
        if self.max_tokens_extraction <= 0:
            raise ValueError(
                f"MAX_TOKENS_EXTRACTION must be positive, got {self.max_tokens_extraction}"
            )


# Module-level singleton — import with: from config.settings import settings
settings = Settings()
