"""Shared SWAN/PZ runtime support.

This module deliberately contains no planner or prompt-loading code so the execution
stage can import it without making query generation reachable.
"""
from __future__ import annotations

import datetime as _dt
import decimal as _dec
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SYSTEM_ROOT = Path(
    os.getenv("PALIMPZEST_SYSTEM_DIR") or (THIS_DIR / "system")
).expanduser().resolve()
SYSTEM_SRC = SYSTEM_ROOT / "src"

if str(SYSTEM_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(THIS_DIR / ".env")
except Exception:
    pass

from pz_integration import (  # noqa: E402
    PZInfrastructureError,
    install_pz_integration,
    reset_live_usage,
    snapshot_live_usage,
)

PRICE = {
    "input": 1.00 / 1_000_000,
    "cache_read": 0.10 / 1_000_000,
    "cache_write": 1.25 / 1_000_000,
    "output": 5.00 / 1_000_000,
}
LIVE_USAGE_KEYS = (
    "input_text_tokens",
    "output_text_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "embedding_input_tokens",
)


def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or "").strip() or default


def env_int(name: str, default: int) -> int:
    try:
        raw = env(name)
        return int(raw) if raw else int(default)
    except Exception:
        return int(default)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_db_path(db: str) -> Path:
    directories: list[Path] = []
    configured = env("DB_FILES_DIR")
    if configured:
        directories.append(Path(configured))
    directories.append(REPO_ROOT / "swan" / "database")
    wanted = db.lower()
    if wanted.endswith(".duckdb"):
        wanted = wanted[:-7]
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.suffix.lower() == ".duckdb" and path.stem.lower() == wanted:
                return path.resolve()
    raise FileNotFoundError(
        f"duckdb not found for db={db!r} (looked in {[str(path) for path in directories]})"
    )


def _json_safe_df(df):
    import pandas as pd

    def normalize(value):
        try:
            if value is None or (
                not isinstance(value, (list, tuple, dict)) and pd.isna(value)
            ):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date, _dt.time)):
            return value.isoformat()
        if isinstance(value, _dec.Decimal):
            return float(value)
        return value

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]) or df[column].dtype == object:
            df[column] = df[column].map(normalize)
    return df


class PZTableLoader:
    """Lazily expose live DuckDB tables as Palimpzest MemoryDatasets."""

    def __init__(self, db_path: Path, *, row_cap: int = 0):
        import duckdb

        self._con = duckdb.connect(str(db_path), read_only=True)
        self._row_cap = int(row_cap or 0)
        self._prefix = db_path.stem
        self._names = {
            str(name).lower(): str(name)
            for (name,) in self._con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
            if not str(name).startswith("sqlite_")
        }
        self._cache: dict[str, Any] = {}

    def table_names(self) -> list[str]:
        return sorted(self._names.values())

    def schema_text(self, *, sample_rows: int) -> str:
        lines: list[str] = []
        sample_max_chars = env_int("E1_ROW_SAMPLE_MAX_CHARS", 6000)
        for table_name in self.table_names():
            info = self._con.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            columns = ", ".join(f"{row[1]} ({row[2]})" for row in info)
            lines.extend((f"TABLE {table_name}:", f"  columns: {columns}"))
            records = self._con.execute(
                f'SELECT * FROM "{table_name}" LIMIT {max(0, int(sample_rows))}'
            ).df().to_dict("records")
            serialized = json.dumps(records, ensure_ascii=False, default=str)
            lines.append(f"  sample_rows: {serialized[:sample_max_chars]}")
            lines.append("")
        return "\n".join(lines).strip()

    def table(self, name: str):
        import palimpzest as pz

        key = str(name).strip().strip('"').lower().split(".")[-1]
        real_name = self._names.get(key)
        if real_name is None:
            raise KeyError(f"table {name!r} not found; available: {self.table_names()}")
        if real_name not in self._cache:
            sql = f'SELECT * FROM "{real_name}"'
            if self._row_cap > 0:
                sql += f" LIMIT {self._row_cap}"
            frame = _json_safe_df(self._con.execute(sql).df())
            self._cache[real_name] = pz.MemoryDataset(
                id=f"{self._prefix}__{real_name}", vals=frame
            )
        return self._cache[real_name]

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass


def build_pipeline_from_code(code: str, db: PZTableLoader):
    import palimpzest as pz

    namespace: dict[str, Any] = {"pz": pz, "__builtins__": __builtins__}
    exec(compile(code, "<generated_program>", "exec"), namespace)  # noqa: S102
    function = namespace.get("build_pipeline")
    if not callable(function):
        raise ValueError("program does not define build_pipeline(db)")
    return function(db)


def _strip_provider(model_id: str) -> str:
    return model_id.strip().split("/", 1)[-1]


def build_operator_models():
    api_base = env("PZ_OP_API_BASE") or env("LLMOP_BASE_URL")
    model_id = env("PZ_OP_MODEL") or env("LLMOP_MODEL")
    if not model_id:
        raise RuntimeError(
            "Operator model is not configured. Set LLMOP_MODEL in the repository "
            ".env or PZ_OP_MODEL in baseline/palimpzest/.env."
        )
    import palimpzest as pz

    if not api_base:
        try:
            return [pz.Model._registry.get(model_id) or pz.Model(model_id)]
        except Exception as exc:
            raise RuntimeError(
                f"Could not construct PZ operator Model {model_id!r}: {exc}. "
                "Set PZ_OP_API_BASE (or LLMOP_BASE_URL) for hosted_vllm."
            ) from exc
    if not model_id.startswith("hosted_vllm/"):
        model_id = f"hosted_vllm/{_strip_provider(model_id)}"
    if not env("VLLM_API_KEY"):
        key = env("PZ_OP_API_KEY") or env("LLMOP_API_KEY")
        if key:
            os.environ["VLLM_API_KEY"] = key
    return [pz.Model(model_id, api_base=api_base.rstrip("/"))]


def build_config():
    import palimpzest as pz

    policy = {
        "maxquality": pz.MaxQuality,
        "mincost": pz.MinCost,
        "mintime": pz.MinTime,
    }.get(env("PZ_POLICY", "maxquality").lower(), pz.MaxQuality)()
    return pz.QueryProcessorConfig(
        policy=policy,
        optimizer_strategy="pareto",
        execution_strategy="parallel",
        available_models=build_operator_models(),
        max_workers=env_int("LLMOP_CONCURRENCY", 8),
        progress=False,
        allow_model_selection=True,
    )


def reset_usage() -> None:
    install_pz_integration()
    reset_live_usage()


def snapshot_usage() -> dict[str, int]:
    return snapshot_live_usage()


def empty_usage() -> dict[str, Any]:
    return {
        **{key: 0 for key in LIVE_USAGE_KEYS},
        "total_llm_calls": 0,
        "optimization_cost_usd": 0.0,
        "per_operator": [],
    }


def extract_usage(collection) -> dict[str, Any]:
    stats = getattr(collection, "execution_stats", None)
    usage = empty_usage()
    if stats is not None:
        for key in LIVE_USAGE_KEYS:
            usage[key] = int(getattr(stats, key, 0) or 0)
        usage["optimization_cost_usd"] = float(
            getattr(stats, "optimization_cost", 0.0) or 0.0
        )
    plan_stats = getattr(stats, "plan_stats", {}) if stats is not None else {}
    if not plan_stats and getattr(collection, "plan_stats", None) is not None:
        plan_stats = {"plan": collection.plan_stats}
    for plan_id, plan in (plan_stats or {}).items():
        for operator_id, operator in (getattr(plan, "operator_stats", {}) or {}).items():
            calls = sum(
                int(getattr(record, "total_llm_calls", 0) or 0)
                for record in getattr(operator, "record_op_stats_lst", []) or []
            )
            usage["total_llm_calls"] += calls
            usage["per_operator"].append(
                {
                    "plan_id": str(plan_id),
                    "op_id": str(operator_id),
                    "op_name": getattr(operator, "op_name", ""),
                    "model_name": getattr(operator, "model_name", None),
                    "input_text_tokens": int(
                        getattr(operator, "input_text_tokens", 0) or 0
                    ),
                    "output_text_tokens": int(
                        getattr(operator, "output_text_tokens", 0) or 0
                    ),
                    "cache_read_tokens": int(
                        getattr(operator, "cache_read_tokens", 0) or 0
                    ),
                    "cache_creation_tokens": int(
                        getattr(operator, "cache_creation_tokens", 0) or 0
                    ),
                    "total_op_cost": float(
                        getattr(operator, "total_op_cost", 0.0) or 0.0
                    ),
                    "llm_calls": calls,
                }
            )
    return usage


def recompute_cost(usage: dict[str, Any]) -> float:
    return (
        int(usage.get("input_text_tokens", 0)) * PRICE["input"]
        + int(usage.get("cache_read_tokens", 0)) * PRICE["cache_read"]
        + int(usage.get("cache_creation_tokens", 0)) * PRICE["cache_write"]
        + int(usage.get("output_text_tokens", 0)) * PRICE["output"]
    )


def read_usage_journal(path: Path) -> dict[str, int]:
    totals = {key: 0 for key in LIVE_USAGE_KEYS}
    starts = finishes = excluded = unmetered = metered = 0
    if path.is_file():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("event") == "start":
                    starts += 1
                elif event.get("event") == "finish":
                    finishes += 1
                    if event.get("excluded_infra_call"):
                        excluded += 1
                    elif event.get("unmetered_failed_call"):
                        unmetered += 1
                    else:
                        metered += 1
                        for key in LIVE_USAGE_KEYS:
                            totals[key] += int(float(event.get(key, 0) or 0))
        except Exception:
            pass
    in_flight = max(0, starts - finishes)
    return {
        **totals,
        "total_llm_calls": metered + unmetered + in_flight,
        "calls_with_usage": metered,
        "unmetered_failed_calls": unmetered + in_flight,
        "excluded_infra_calls": excluded,
        "in_flight_calls": in_flight,
    }


def sanitize_qid(qid: str) -> str:
    safe = "".join(
        "_" if character in {"/", "\\", "\x00"} or ord(character) < 32 else character
        for character in qid
    )
    return safe.strip().strip(".")
