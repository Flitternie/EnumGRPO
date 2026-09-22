"""EnumGRPO integration layer for an unmodified Palimpzest checkout.

The upstream repository is resolved through ``PALIMPZEST_SYSTEM_DIR`` (or the
local ``system/`` fallback). This module applies benchmark-specific behavior at
runtime without modifying that checkout:

* route hosted-vLLM usage into Palimpzest's stats;
* omit operator temperature to match the main system;
* retry and exclude model/API infrastructure failures;
* journal every completed operator call so timeout/OOM costs survive;
* fall back to Palimpzest's bundled model metadata when offline.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_USAGE_FIELDS = (
    "input_text_tokens",
    "output_text_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "embedding_input_tokens",
)
_usage_lock = threading.Lock()
_usage_totals: dict[str, float] = {}
_installed = False


def _import_palimpzest_with_local_model_metadata():
    """Import Palimpzest without making model metadata a network dependency.

    Upstream constructs its built-in ``Model`` objects while importing the
    package, before our normal runtime patches can be installed.  Serve the
    bundled metadata file for that one URL during the initial import so a
    blocked S3 request cannot leave the package half-imported.
    """
    import requests

    data_url = (
        "https://palimpzest-research.s3.us-east-1.amazonaws.com/"
        "pz_models_information.json"
    )
    system_root = Path(
        os.getenv("PALIMPZEST_SYSTEM_DIR")
        or (Path(__file__).resolve().parent / "system")
    ).expanduser().resolve()
    local_path = (
        system_root
        / "src"
        / "palimpzest"
        / "utils"
        / "pz_models_information.json"
    )
    local_data = json.loads(local_path.read_text(encoding="utf-8"))
    original_get = requests.get

    class LocalResponse:
        def json(self):
            return local_data

    def local_metadata_get(url, *args, **kwargs):
        if str(url) == data_url:
            return LocalResponse()
        return original_get(url, *args, **kwargs)

    requests.get = local_metadata_get
    try:
        import palimpzest.query.generators.generators as generators
    finally:
        requests.get = original_get
    return generators


class PZInfrastructureError(BaseException):
    """Model/API failure after infrastructure retries are exhausted.

    This intentionally derives from BaseException so Palimpzest's broad
    ``except Exception`` does not silently turn infrastructure failures into
    False/NULL semantic outputs. The benchmark runner catches it explicitly.
    """


def reset_live_usage() -> None:
    global _usage_totals
    with _usage_lock:
        _usage_totals = {
            **{field: 0.0 for field in _USAGE_FIELDS},
            "total_llm_calls": 0.0,
            "calls_with_usage": 0.0,
            "unmetered_failed_calls": 0.0,
            "excluded_infra_calls": 0.0,
            "in_flight_calls": 0.0,
        }


def snapshot_live_usage() -> dict[str, int]:
    with _usage_lock:
        return {key: int(value) for key, value in _usage_totals.items()}


def _append_usage_journal(event: dict[str, Any]) -> None:
    path = os.environ.get("PZ_USAGE_JOURNAL_PATH", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")
            fh.flush()
    except Exception as exc:
        logger.warning("Could not write PZ usage journal %s: %s", path, exc)


def _record_call_started() -> None:
    with _usage_lock:
        _usage_totals["in_flight_calls"] += 1
        _append_usage_journal({"event": "start"})


def _record_call_finished(
    usage: dict[str, int] | None, *, excluded_infra: bool = False
) -> None:
    with _usage_lock:
        _usage_totals["in_flight_calls"] = max(
            0.0, _usage_totals["in_flight_calls"] - 1
        )
        event: dict[str, Any] = {"event": "finish"}
        if excluded_infra:
            _usage_totals["excluded_infra_calls"] += 1
            event["excluded_infra_call"] = 1
        elif usage is None:
            _usage_totals["total_llm_calls"] += 1
            _usage_totals["unmetered_failed_calls"] += 1
            event["unmetered_failed_call"] = 1
        else:
            _usage_totals["total_llm_calls"] += 1
            _usage_totals["calls_with_usage"] += 1
            for field in _USAGE_FIELDS:
                value = int(usage.get(field, 0) or 0)
                _usage_totals[field] += value
                event[field] = value
        _append_usage_journal(event)


def _response_usage(response: Any) -> dict[str, int]:
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return {field: 0 for field in _USAGE_FIELDS}
    if hasattr(usage_obj, "model_dump"):
        usage = usage_obj.model_dump()
    elif isinstance(usage_obj, dict):
        usage = usage_obj
    else:
        usage = {}
    details = usage.get("prompt_tokens_details") or {}
    cache_read = int(
        usage.get("cache_read_input_tokens")
        or details.get("cached_tokens")
        or 0
    )
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    return {
        "input_text_tokens": max(0, prompt - cache_read - cache_creation),
        "output_text_tokens": int(usage.get("completion_tokens") or 0),
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "embedding_input_tokens": 0,
    }


class _LiteLLMProxy:
    """Proxy only Palimpzest generators, leaving the external planner untouched."""

    def __init__(self, module: Any):
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def completion(self, *args: Any, **kwargs: Any) -> Any:
        # Operators in the main system leave temperature unset.
        kwargs.pop("temperature", None)
        max_attempts = max(
            1, int(os.environ.get("E1_MAX_INFRA_ATTEMPTS", "3") or "3")
        )
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            _record_call_started()
            try:
                response = self._module.completion(*args, **kwargs)
                _record_call_finished(_response_usage(response))
                return response
            except Exception as exc:
                last_error = exc
                _record_call_finished(None, excluded_infra=True)
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
            except BaseException:
                # Query timeout/OOM is a benchmarked system failure, not infra.
                _record_call_finished(None)
                raise
        raise PZInfrastructureError(
            f"model/API call failed after {max_attempts} attempts: {last_error}"
        )


def _patch_prompt_usage() -> None:
    from palimpzest.prompts.prompt_manager import PromptManager

    if getattr(PromptManager.extract_usage_stats, "_enumgrpo_patched", False):
        return
    original = PromptManager.extract_usage_stats

    def extract_usage_stats(self, usage: dict, is_audio_op: bool) -> dict[str, int]:
        stats = original(self, usage, is_audio_op)
        if self.model.is_vllm_model():
            details = usage.get("prompt_tokens_details") or {}
            cache_read = int(details.get("cached_tokens") or 0)
            stats["cache_read_tokens"] = cache_read
            stats["input_text_tokens"] = max(
                0, int(usage.get("prompt_tokens") or 0) - cache_read
            )
        return stats

    extract_usage_stats._enumgrpo_patched = True  # type: ignore[attr-defined]
    PromptManager.extract_usage_stats = extract_usage_stats


def _patch_model_metrics_fallback() -> None:
    import json as json_module

    from palimpzest.utils import model_info_helpers

    cls = model_info_helpers.ModelMetricsManager
    if getattr(cls._load_data, "_enumgrpo_patched", False):
        return

    def _load_data(self) -> None:
        if self._metrics_cache is not None:
            return
        try:
            response = model_info_helpers.requests.get(self.data_url, timeout=15)
            self._metrics_cache = response.json()
            if not self._metrics_cache:
                raise ValueError("remote model metrics empty")
        except Exception as exc:
            logger.warning(
                "Could not fetch Palimpzest model metrics (%s); using local copy",
                exc,
            )
            local = Path(model_info_helpers.__file__).with_name(
                "pz_models_information.json"
            )
            try:
                self._metrics_cache = json_module.loads(
                    local.read_text(encoding="utf-8")
                )
            except Exception:
                self._metrics_cache = {}

    _load_data._enumgrpo_patched = True  # type: ignore[attr-defined]
    cls._load_data = _load_data


def install_pz_integration() -> None:
    """Install benchmark adapters once after importing the upstream package."""
    global _installed
    if _installed:
        return
    generators = _import_palimpzest_with_local_model_metadata()

    if not isinstance(generators.litellm, _LiteLLMProxy):
        generators.litellm = _LiteLLMProxy(generators.litellm)
    _patch_prompt_usage()
    _patch_model_metrics_fallback()
    _installed = True


reset_live_usage()
