"""
Central application settings.

Reads from the environment / ``.env`` file via ``pydantic-settings``.
All other modules import the singleton ``settings`` from here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
     """Typed application settings – auto-loaded from ``.env`` / env vars."""

     model_config = SettingsConfigDict(
          env_file=".env",
          env_file_encoding="utf-8",
          case_sensitive=False,
          extra="ignore",
      )

      # ── LLM provider selection ───────────────────────────────────────────
     llm_provider: Literal[
          "openai",
          "openai-chat",
          "ollama",
          "groq",
          "gemini",
          "anthropic",
      ] = Field(default="ollama", alias="LLM_PROVIDER")

     llm_model: str = Field(default="llama3.2", alias="LLM_MODEL")

     llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

     reasoning_effort: str | None = Field(
          default=None, alias="REASONING_EFFORT"
      )

      # ── Per-provider base URLs (auto-applied; override in .env if needed) ─
     ollama_base_url: str = Field(
          default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL"
      )
     openai_base_url: str = Field(
          default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
      )
     groq_base_url: str = Field(
          default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL"
      )
     gemini_base_url: str = Field(
          default="https://generativelanguage.googleapis.com/v1beta/openai/",
          alias="GEMINI_BASE_URL",
      )

      # ── API keys ─────────────────────────────────────────────────────────
     openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
     groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
     gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
     anthropic_api_key: str | None = Field(
          default=None, alias="ANTHROPIC_API_KEY"
      )

     # ── Vision model (optional) ────────────────────────────────────────────
     vision_model: str = Field(default="llava", alias="VISION_MODEL")

      # ── MCP server ───────────────────────────────────────────────────────
     mcp_transport: Literal["stdio", "sse", "http"] = Field(
          default="stdio", alias="MCP_TRANSPORT"
      )
     mcp_host: str = Field(default="127.0.0.1", alias="MCP_HOST")
     mcp_port: int = Field(default=8000, alias="MCP_PORT")

      # ── Workflow defaults ─────────────────────────────────────────────────
     topic: str = Field(default="list comprehension", alias="TOPIC")
     num_questions: int = Field(default=2, alias="NUM_QUESTIONS")
     num_test_cases: int = Field(default=2, alias="NUM_TEST_CASES")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
     """Return the cached ``Settings`` singleton."""
     return Settings()


# Module-level singleton – import this everywhere:
#     from mls_agents.config import settings
settings: Settings = get_settings()
