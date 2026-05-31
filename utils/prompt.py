"""Utilities for building agent system prompts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLAIN_PROMPT = str(_REPO_ROOT / "agent" / "codebase" / "prompts" / "agentic_db_plain.md")


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _append_experiences_section(base_text: str, experiences: Dict[str, str], db_label: str = "") -> str:
    """Return base_text with a '## Learned Experiences' section appended."""
    if not experiences:
        raise ValueError("experience pool is empty -- nothing to append")
    intro = (
        f"The following guidelines were learned from past queries on this database. "
        "Apply any that are relevant to the current task."
        if db_label else
        "The following guidelines were learned from past queries. "
        "Apply any that are relevant to the current task."
    )
    lines = [
        base_text.rstrip(),
        "",
        "## Learned Experiences",
        "",
        intro,
        "",
    ]
    for key, desc in experiences.items():
        lines.append(f"- **{key}**: {desc}")
    return "\n".join(lines)


def _base_text(base_prompt_path: str = _PLAIN_PROMPT) -> str:
    p = Path(base_prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"Base prompt not found: {p}")
    return p.read_text(encoding="utf-8").rstrip()


# ---------------------------------------------------------------------------
# Public helpers (kept for external callers)
# ---------------------------------------------------------------------------

def build_experience_prompt(experience_json_path: Path, base_prompt_path: str = _PLAIN_PROMPT) -> str:
    """Build a single composite prompt from a global experience JSON.

    The JSON must have an ``experiences`` dict keyed by experience ID.
    Raises ValueError if the experiences pool is empty or malformed.
    """
    data = json.loads(experience_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in '{experience_json_path}', got {type(data).__name__}")
    raw = data.get("experiences", {})
    if not isinstance(raw, dict):
        raise ValueError(
            f"'experiences' key in '{experience_json_path}' must be a dict, got {type(raw).__name__}"
        )
    experiences = {str(k): str(v) for k, v in raw.items()}
    if not experiences:
        raise ValueError(f"'experiences' dict in '{experience_json_path}' is empty")
    return _append_experiences_section(_base_text(base_prompt_path), experiences)


def build_db_experience_prompts(
    db_experiences_json: Path,
    out_dir: Path,
    base_prompt_path: str = _PLAIN_PROMPT,
) -> Dict[str, str]:
    """Load a DB-specific experience JSON and write one composite prompt per database.

    The JSON must contain a ``db_experiences`` key mapping
    ``db_key -> {exp_id -> text}``.

    Returns ``{db_key: absolute_prompt_path}`` for every non-empty pool.
    Raises ValueError if the file is malformed or every pool is empty.
    """
    data = json.loads(db_experiences_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in '{db_experiences_json}', got {type(data).__name__}"
        )
    raw = data.get("db_experiences", {})
    if not isinstance(raw, dict):
        raise ValueError(
            f"'db_experiences' key in '{db_experiences_json}' must be a dict, "
            f"got {type(raw).__name__}"
        )
    db_experiences: Dict[str, Dict[str, str]] = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            raise ValueError(
                f"db_experiences['{k}'] must be a dict, got {type(v).__name__}"
            )
        pool = {str(eid): str(etxt) for eid, etxt in v.items()}
        if pool:
            db_experiences[str(k)] = pool

    if not db_experiences:
        raise ValueError(
            f"All db experience pools in '{db_experiences_json}' are empty"
        )

    base = _base_text(base_prompt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, str] = {}
    for db_key, pool in db_experiences.items():
        prompt_text = _append_experiences_section(base, pool, db_label=db_key)
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in db_key)
        out_path = out_dir / f"_db_exp_{safe_key}.md"
        out_path.write_text(prompt_text, encoding="utf-8")
        result[db_key] = str(out_path)
    return result


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def resolve_prompt(
    prompt_file: Optional[str],
    out_dir: Path,
) -> Tuple[str, Dict[str, str]]:
    """Resolve ``--prompt_file`` to either a single prompt path or a per-db map.

    Auto-detects the input type and returns ``(single_path, db_map)`` where
    exactly one of the two is populated:

    - **None / empty** -- ``("", {})``; caller uses the agent default.
    - **\\*.md file** -- ``(abs_path, {})``; used as-is for every query.
    - **\\*.json with ``experiences`` key** -- global experience JSON; a
      composite prompt is written to ``out_dir/_experience_prompt.md`` and
      ``(path, {})`` is returned.
    - **\\*.json with ``db_experiences`` key** -- per-db experience JSON; one
      prompt per database is written and ``("", {db_key: path})`` is returned.

    Raises ``ValueError`` or ``FileNotFoundError`` on any problem (missing
    file, wrong format, empty experiences, unrecognised extension, etc.).
    """
    if not prompt_file:
        return ("", {})

    p = Path(prompt_file)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"--prompt_file path does not exist: {p}")

    ext = p.suffix.lower()

    # --- .md file: use directly ---
    if ext == ".md":
        return (str(p), {})

    # --- .json file: auto-detect global vs DB-specific by key ---
    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object in '{p}', got {type(data).__name__}"
            )
        if "db_experiences" in data:
            db_map = build_db_experience_prompts(p, out_dir)
            return ("", db_map)
        if "experiences" in data:
            composite = build_experience_prompt(p)
            tmp_path = out_dir / "_experience_prompt.md"
            tmp_path.write_text(composite, encoding="utf-8")
            return (str(tmp_path), {})
        raise ValueError(
            f"JSON file '{p}' has neither an 'experiences' nor a 'db_experiences' key"
        )

    raise ValueError(
        f"Unrecognised --prompt_file extension '{ext}' for path '{p}'. "
        "Expected a .md file or a .json experience file."
    )


def resolve_prompt_file(prompt_file: Optional[str], out_dir: Path) -> str:
    """Legacy single-path wrapper around resolve_prompt.

    Returns the single prompt path, or empty string.  Raises if a directory
    or DB-specific JSON is passed (those require the full resolve_prompt API).
    """
    single, db_map = resolve_prompt(prompt_file, out_dir)
    if db_map:
        raise ValueError(
            "resolve_prompt_file cannot handle DB-specific experience inputs. "
            "Use resolve_prompt() instead."
        )
    return single
