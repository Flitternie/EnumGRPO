"""Minimal config for the MCP-backed DuckDB agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Root of the `agent/` package (used for run artifacts).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Maximum iterations for the main agent's conversation.
# Can be overridden via the AGENT_MAX_ITERATIONS environment variable.
MAX_ITERATION_PER_RUN = int(os.getenv("AGENT_MAX_ITERATIONS") or 30)


@dataclass(slots=True)
class RuntimeConfig:
    model_name: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    max_iteration_per_run: int | None = None   # None = use MAX_ITERATION_PER_RUN default
    mcp_result_max_chars: int | None = None    # None = use MCP_RESULT_MAX_CHARS env / default (4000)


def get_model_name(override: str | None = None, default: str | None = None) -> str:
    """
    Agent LLM (OpenAI-compatible via LiteLLM in OpenHands):
    CLI override > env AGENT_MODEL > default (if provided).
    """
    if override and str(override).strip():
        return str(override).strip()
    env_model = (os.getenv("AGENT_MODEL") or "").strip()
    if env_model:
        return env_model
    if default and str(default).strip():
        return str(default).strip()
    raise ValueError("Agent model is not set. Provide --model or set AGENT_MODEL.")


def get_api_key() -> str:
    """
    Agent LLM API key.

    Note: For some LiteLLM providers (e.g. Bedrock), credentials are sourced
    from AWS environment variables and no API key should be passed.
    """
    model = (os.getenv("AGENT_MODEL") or "").strip()
    if model.lower().startswith("bedrock/"):
        # Bedrock auth uses AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN.
        return ""

    api_key = (os.getenv("AGENT_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Environment variable AGENT_API_KEY is not set.")
    return api_key


def get_base_url() -> str | None:
    """
    Optional base URL for the agent LLM (OpenAI-compatible). If unset, OpenHands defaults apply.
    """
    url = (os.getenv("AGENT_BASE_URL") or "").strip()
    return url or None
