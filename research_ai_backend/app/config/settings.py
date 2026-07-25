"""
Centralized environment / settings configuration.

Every agent, tool and service pulls its credentials and tunables from a
single `Settings` instance (see `get_settings()` below) instead of reading
`os.environ` directly, so the whole backend stays configurable from one
`.env` file.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ----- Core LLM -----
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # ----- Web / search tools -----
    tavily_api_key: str = ""
    serper_api_key: str = ""

    # ----- Vector store -----
    vector_store_provider: str = "chroma"  # chroma | qdrant
    chroma_persist_dir: str = "./data/chroma"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # ----- Knowledge graph store -----
    graph_store_provider: str = "networkx"  # networkx | neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ----- Speech -----
    whisper_model_size: str = "base"
    deepgram_api_key: str = ""

    # ----- App -----
    app_env: str = "development"
    cors_allow_origins: str = "http://localhost:3000,http://localhost:5173"
    log_level: str = "INFO"
    max_upload_mb: int = 50

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> "Settings":
    """Cached settings accessor — import and call this everywhere."""
    return Settings()
