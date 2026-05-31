"""Shared LLM utilities for the db_revise project.

This module is the single place for LLM configuration, credential resolution,
and call helpers.  Three consumers use it with different access patterns:

  - LLM operators (tools/llm_map.py, tools/llm_reduce.py)
      Sync, high-throughput, short single-turn prompts.
      Entry point: ``llmop_call``

  - Experience distillation (learning/experience_updater.py)
      Async, multi-message chat completions.
      Entry point: ``chat_complete_async``

  - Both share: model/provider config helpers and text post-processing.

All LLM calls go through ``litellm``, which handles Bedrock and OpenAI-compatible
endpoints transparently using the same Chat Completions API surface.

Environment variables
---------------------
Operator tools (LLMOP_* namespace):
    LLMOP_MODEL        e.g. "moonshotai.kimi-k2.5" or "gpt-4o"
    LLMOP_API_KEY      required unless model is Bedrock
    LLMOP_BASE_URL     optional custom OpenAI-compatible base URL
    LLMOP_CONCURRENCY  max parallel calls (default 8, clamped to [1, 32])
    LLMOP_TIMEOUT_S    per-request litellm timeout in seconds (default 3600)

Learning LLM (LEARNING_LLM_* namespace, falls back to AGENT_* equivalents):
    LEARNING_LLM_MODEL    defaults to AGENT_MODEL
    LEARNING_LLM_API_KEY  required when not Bedrock; falls back to AGENT_API_KEY
    LEARNING_LLM_BASE_URL optional; falls back to AGENT_BASE_URL

AWS Bedrock (shared by both namespaces when provider=bedrock):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional)
    AWS_REGION | AWS_DEFAULT_REGION  (default: us-east-1)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Per-request timeout forwarded to litellm for every LLM-operator call.
# Set LLMOP_TIMEOUT_S in the environment to override. 0 = no timeout (not recommended).
LLMOP_TIMEOUT_S: int = int((os.getenv("LLMOP_TIMEOUT_S") or "3600").strip() or "3600")


# ---------------------------------------------------------------------------
# General text / value helpers
# ---------------------------------------------------------------------------

def strip_wrapping_quotes(s: str) -> str:
    t = (s or "").strip()
    if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
        return t[1:-1].strip()
    return t


def truncate_cell(v: Any, *, max_chars: int) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v)
    if max_chars > 0 and len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


def payload_for_error(payload: Dict[str, Any], *, max_chars: int = 1200) -> str:
    try:
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        s = repr(payload)
    if max_chars > 0 and len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


# ---------------------------------------------------------------------------
# LLM output post-processing
# ---------------------------------------------------------------------------

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning trace blocks from model output."""
    s = "" if text is None else str(text)
    if not s:
        return ""
    s2 = re.sub(r"(?is)<think>.*?</think>", "", s)
    if "</think>" in s2.lower():
        lower = s2.lower()
        idx = lower.rfind("</think>")
        s2 = s2[idx + len("</think>"):]
    lower2 = s2.lower()
    open_idx = lower2.find("<think>")
    if open_idx != -1:
        s2 = s2[:open_idx]
    return s2.strip()


def llmop_postprocess_output_text(text: str) -> str:
    """Standard LLM-operator output cleanup: strip quotes, think tags, and whitespace."""
    s = strip_wrapping_quotes((text or "").strip())
    s = strip_think_tags(s)
    return (s or "").strip()


# ---------------------------------------------------------------------------
# LLM-operator planning helpers
# ---------------------------------------------------------------------------

def get_llmop_concurrency(*, default: int = 8, env_var: str = "LLMOP_CONCURRENCY") -> int:
    """Max parallel LLM calls for LLM-operator tools; clamped to [1, 32]."""
    raw = (os.getenv(env_var) or "").strip()
    if not raw:
        n = int(default)
    else:
        try:
            n = int(raw)
        except Exception as e:
            raise RuntimeError(f"Invalid {env_var}={raw!r} (expected int 1-32)") from e
    return max(1, min(int(n), 32))


def heuristic_tokens_from_chars(chars: int) -> int:
    """Rough heuristic: ~4 chars/token."""
    try:
        c = max(0, int(chars))
    except Exception:
        c = 0
    return int((c + 3) // 4)


def plan_id(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def plan_fingerprint_payload(payload: Dict[str, Any], *, non_semantic_keys: Iterable[str]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("plan payload must be a dict")
    drop = set(str(k) for k in (non_semantic_keys or []))
    return {k: v for (k, v) in payload.items() if k not in drop}


# ---------------------------------------------------------------------------
# Provider / model configuration
# ---------------------------------------------------------------------------

def is_bedrock_model(model: str) -> bool:
    """Return True if the model string targets AWS Bedrock (litellm 'bedrock/' prefix)."""
    return str(model or "").lower().startswith("bedrock/")


def _bedrock_supports_thinking(model: str) -> bool:
    """Return True only for Bedrock-routed Anthropic Claude models that accept the
    ``thinking`` parameter.  Other Bedrock models (e.g. Kimi, Llama) do not support
    it and raise ``UnsupportedParamsError`` when it is present.
    """
    m = str(model or "").lower()
    return m.startswith("bedrock/") and "anthropic" in m


def get_llmop_model() -> str:
    model = (os.getenv("LLMOP_MODEL") or "").strip()
    if not model:
        raise RuntimeError("LLMOP model not configured (set LLMOP_MODEL)")
    return model


def get_learning_model() -> str:
    model = (os.getenv("LEARNING_LLM_MODEL") or os.getenv("AGENT_MODEL") or "").strip()
    if not model:
        raise RuntimeError("Learning LLM model not configured (set LEARNING_LLM_MODEL or AGENT_MODEL)")
    return model


def apply_litellm_provider_kwargs(
    kwargs: Dict[str, Any],
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_portkey_openai_cache: bool = False,
) -> Dict[str, Any]:
    """Apply provider-specific LiteLLM auth/routing kwargs in one place.

    Supports:
      - Bedrock: credentials come from AWS env vars; optionally disable Claude thinking.
      - OpenAI-compatible routes: pass api_key/base_url as usual.
      - Portkey Anthropic-native routes: pass Portkey key via x-portkey-api-key
        while giving LiteLLM's Anthropic adapter a harmless placeholder api_key.

    The function mutates and returns ``kwargs`` for ergonomic use by call builders.
    """
    m = str(model or "").strip()
    model_lower = m.lower()
    base = (base_url or "").strip()
    base_lower = base.lower()
    is_portkey = "portkey.ai" in base_lower
    is_portkey_anthropic_route = is_portkey and model_lower.startswith("anthropic/")
    is_portkey_openai_route = is_portkey and model_lower.startswith("openai/")
    is_anthropic_model = ("claude" in model_lower) or ("anthropic" in model_lower)

    if is_bedrock_model(m):
        # litellm reads AWS creds from the environment automatically.
        if _bedrock_supports_thinking(m):
            kwargs["thinking"] = {"type": "disabled"}
        return kwargs

    key = (api_key or "").strip()
    if key:
        if is_portkey_anthropic_route:
            headers = dict(kwargs.get("extra_headers") or {})
            headers["x-portkey-api-key"] = key
            kwargs["extra_headers"] = headers
            # LiteLLM's Anthropic adapter validates/passes api_key separately;
            # Portkey authenticates from x-portkey-api-key.
            kwargs["api_key"] = "dummy"
        else:
            kwargs["api_key"] = key

    if base:
        normalized_base = base.rstrip("/")
        # LiteLLM appends /v1/messages for anthropic/*; Portkey's Anthropic
        # route expects the gateway root, not the OpenAI-style /v1 base.
        if is_portkey_anthropic_route and normalized_base.endswith("/v1"):
            normalized_base = normalized_base[:-3]
        kwargs["base_url"] = normalized_base

    if enable_portkey_openai_cache and is_portkey_openai_route and is_anthropic_model:
        body_key = "litellm_extra_body" if "litellm_extra_body" in kwargs else "extra_body"
        extra_body = dict(kwargs.get(body_key) or {})
        extra_body["cache_control"] = {"type": "ephemeral"}
        kwargs[body_key] = extra_body
        headers = dict(kwargs.get("extra_headers") or {})
        headers["x-portkey-strict-open-ai-compliance"] = "false"
        kwargs["extra_headers"] = headers

    return kwargs


# ---------------------------------------------------------------------------
# litellm call kwargs builders
# ---------------------------------------------------------------------------

def _llmop_litellm_kwargs(*, model: str, prompt: str) -> Dict[str, Any]:
    """Build litellm.completion kwargs for an LLM-operator single-turn call."""
    m = str(model or "").strip()
    messages = [{"role": "user", "content": str(prompt or "")}]
    kwargs: Dict[str, Any] = {"model": m, "messages": messages}

    if LLMOP_TIMEOUT_S > 0:
        kwargs["timeout"] = LLMOP_TIMEOUT_S

    if is_bedrock_model(m):
        apply_litellm_provider_kwargs(kwargs, model=m)
    else:
        api_key = (os.getenv("LLMOP_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("LLMOP API key not configured (set LLMOP_API_KEY)")
        base_url = (os.getenv("LLMOP_BASE_URL") or "").strip() or None
        apply_litellm_provider_kwargs(kwargs, model=m, api_key=api_key, base_url=base_url)

    return kwargs


def _learning_litellm_kwargs(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Build litellm.completion / acompletion kwargs for a learning/distillation call."""
    m = str(model or "").strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs: Dict[str, Any] = {"model": m, "messages": messages, "temperature": temperature}

    if is_bedrock_model(m):
        apply_litellm_provider_kwargs(kwargs, model=m)
    else:
        api_key = (os.getenv("LEARNING_LLM_API_KEY") or os.getenv("AGENT_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "No API key for learning LLM. "
                "Set LEARNING_LLM_API_KEY or reuse AGENT_API_KEY in the environment."
            )
        base_url = (os.getenv("LEARNING_LLM_BASE_URL") or os.getenv("AGENT_BASE_URL") or "").strip() or None
        apply_litellm_provider_kwargs(kwargs, model=m, api_key=api_key, base_url=base_url)

    return kwargs


# ---------------------------------------------------------------------------
# Token usage extraction (works on litellm ModelResponse for all providers)
# ---------------------------------------------------------------------------

def extract_token_usage(resp: Any) -> Optional[Dict[str, int]]:
    """Extract token usage from a litellm ``ModelResponse``.

    LiteLLM follows the OpenAI-shaped ``usage`` object (see LiteLLM docs), including:

    - ``prompt_tokens``: all prompt tokens (cache-miss and cache-hit input combined).
    - ``completion_tokens``: output tokens.
    - ``total_tokens``: typically ``prompt_tokens + completion_tokens``.
    - ``prompt_tokens_details.cached_tokens``: tokens that were a cache hit on this call.
    - ``completion_tokens_details.reasoning_tokens``: reasoning / thinking tokens when present.
    - ``cache_creation_input_tokens`` (Anthropic): tokens written to the prompt cache; billed separately.

    Returned dict uses operator-friendly keys (aligned with MCP / OpenHands):

    - ``input_tokens`` / ``output_tokens`` / ``total_tokens`` — from the fields above.
    - ``cache_read_tokens`` — from ``prompt_tokens_details.cached_tokens``, or provider aliases.
    - ``cache_write_tokens`` — from ``cache_creation_input_tokens``, or nested details when present.
    - ``reasoning_tokens`` — from ``completion_tokens_details.reasoning_tokens``, or top-level aliases.
    """
    try:
        usage = getattr(resp, "usage", None)
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")
        if usage is None:
            return None

        def _get(obj: Any, key: str) -> Optional[int]:
            v = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
            try:
                return int(v) if v is not None else None
            except Exception:
                return None

        def _get_first(obj: Any, keys: Tuple[str, ...]) -> Optional[int]:
            for key in keys:
                v = _get(obj, key)
                if v is not None:
                    return v
            return None

        def _nested_dict(obj: Any, key: str) -> Optional[Dict[str, Any]]:
            raw = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
            if raw is None:
                return None
            if isinstance(raw, dict):
                return raw
            dump = getattr(raw, "model_dump", None)
            if callable(dump):
                try:
                    d = dump()
                    return d if isinstance(d, dict) else None
                except Exception:
                    return None
            return None

        def _int_from_dict(d: Dict[str, Any], key: str) -> Optional[int]:
            if key not in d or d[key] is None:
                return None
            try:
                return int(d[key])
            except Exception:
                return None

        # Base counts (LiteLLM normalises to prompt_tokens / completion_tokens / total_tokens).
        # Use `is not None` instead of `or` to avoid treating 0 as falsy.
        _pt = _get(usage, "prompt_tokens")
        input_tokens = _pt if _pt is not None else _get(usage, "input_tokens")
        _ct = _get(usage, "completion_tokens")
        output_tokens = _ct if _ct is not None else _get(usage, "output_tokens")
        total_tokens = _get(usage, "total_tokens")

        if total_tokens is None:
            if input_tokens is not None and output_tokens is not None:
                total_tokens = int(input_tokens + output_tokens)
            elif input_tokens is not None:
                total_tokens = int(input_tokens)
            elif output_tokens is not None:
                total_tokens = int(output_tokens)

        out: Dict[str, Optional[int]] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        ptd = _nested_dict(usage, "prompt_tokens_details")
        ctd = _nested_dict(usage, "completion_tokens_details")

        # Cache hits: LiteLLM puts these in prompt_tokens_details.cached_tokens first.
        cache_read: Optional[int] = None
        if isinstance(ptd, dict):
            cache_read = _int_from_dict(ptd, "cached_tokens")
            if cache_read is None:
                cache_read = _int_from_dict(ptd, "cache_read_tokens")
        if cache_read is None:
            cache_read = _get_first(
                usage,
                (
                    "cache_read_input_tokens",
                    "cache_read_tokens",
                    "prompt_cache_read_tokens",
                ),
            )

        # Cache writes: Anthropic exposes cache_creation_input_tokens on usage; OpenAI may nest.
        cache_write = _get(usage, "cache_creation_input_tokens")
        if cache_write is None and isinstance(ptd, dict):
            cache_write = _int_from_dict(ptd, "cache_creation_tokens")
        if cache_write is None:
            cache_write = _get_first(
                usage,
                (
                    "cache_write_tokens",
                    "prompt_cache_creation_tokens",
                ),
            )

        if cache_read is not None:
            out["cache_read_tokens"] = cache_read
        if cache_write is not None:
            out["cache_write_tokens"] = cache_write

        # Reasoning: LiteLLM puts these in completion_tokens_details.reasoning_tokens first.
        reasoning: Optional[int] = None
        if isinstance(ctd, dict):
            reasoning = _int_from_dict(ctd, "reasoning_tokens")
        if reasoning is None:
            reasoning = _get_first(usage, ("reasoning_tokens", "thinking_tokens"))
        if reasoning is not None:
            out["reasoning_tokens"] = reasoning

        if all(v is None for v in out.values()):
            return None
        return {k: int(v) for (k, v) in out.items() if v is not None}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# High-level call helpers
# ---------------------------------------------------------------------------

def llmop_call(*, model: str, prompt: str) -> Tuple[str, Optional[Dict[str, int]]]:
    """
    Synchronous single-turn LLM call for operator tools (llm_map, llm_reduce).

    Returns (output_text, token_usage).  Works for both Bedrock and
    OpenAI-compatible endpoints via litellm.
    """
    import litellm  # deferred so the module is cheap to import

    kwargs = _llmop_litellm_kwargs(model=model, prompt=prompt)
    resp = litellm.completion(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    text = strip_think_tags(text)
    return text, extract_token_usage(resp)


async def llmop_call_async(*, model: str, prompt: str) -> Tuple[str, Optional[Dict[str, int]]]:
    """Async variant of ``llmop_call`` for contexts that already have an event loop."""
    import litellm

    kwargs = _llmop_litellm_kwargs(model=model, prompt=prompt)
    resp = await litellm.acompletion(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    text = strip_think_tags(text)
    return text, extract_token_usage(resp)


async def chat_complete_async(
    model: str,
    *,
    system: str,
    user: str,
    temperature: float = 0.2,
) -> str:
    """
    Async chat completion for training / experience distillation.

    Uses the LEARNING_LLM_* credential namespace (falls back to AGENT_*).
    Returns the assistant message content as a plain string.
    """
    import litellm

    kwargs = _learning_litellm_kwargs(model=model, system=system, user=user, temperature=temperature)
    resp = await litellm.acompletion(**kwargs)
    return resp.choices[0].message.content or ""
