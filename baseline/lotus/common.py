"""Shared data, schema, and serialization helpers for the LOTUS SWAN baseline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SYSTEM_DIR = Path(
    os.getenv("LOTUS_SYSTEM_DIR") or (THIS_DIR / "system")
).expanduser().resolve()
PROMPTS_DIR = THIS_DIR / "prompts"


def resolve_under_repo(path_value: str | Path) -> Path:
    """Resolve relative paths from the repository root, independent of the CWD."""
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_records(
    rows: list[dict[str, Any]],
    *,
    run_ids: list[str],
    run_dbs: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if run_ids:
        wanted_ids = {value.strip() for value in run_ids}
        rows = [row for row in rows if str(row.get("question_id") or "").strip() in wanted_ids]
    if run_dbs:
        wanted_dbs = {value.strip() for value in run_dbs}
        rows = [row for row in rows if str(row.get("db") or "").strip() in wanted_dbs]
    if limit > 0:
        rows = rows[:limit]
    return rows


def resolve_duckdb_path(db: str, db_dir: Path) -> Path:
    """Resolve <db>.duckdb inside db_dir, including a case-insensitive fallback."""
    wanted = str(db or "").strip()
    direct = db_dir / f"{wanted}.duckdb"
    if direct.is_file():
        return direct
    if db_dir.is_dir():
        for path in db_dir.iterdir():
            if path.suffix.lower() == ".duckdb" and path.stem.lower() == wanted.lower():
                return path
    raise FileNotFoundError(f"duckdb not found for db={wanted!r} in {db_dir}")


def load_db_tables(db_path: Path) -> dict[str, Any]:
    """Load every non-internal DuckDB table into a pandas DataFrame."""
    import duckdb

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        names = [row[0] for row in connection.execute("SHOW TABLES").fetchall()]
        return {
            name: connection.execute(f'SELECT * FROM "{name}"').fetchdf()
            for name in names
            if not str(name).startswith("sqlite_")
        }
    finally:
        connection.close()


def load_schema(db: str) -> list[dict[str, Any]]:
    base = REPO_ROOT / "swan" / "schema" / db
    records: list[dict[str, Any]] = []
    if not base.is_dir():
        return records
    for json_file in sorted(base.rglob("*.json")):
        try:
            value = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                records.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _short_table_name(full_name: str) -> str:
    return str(full_name or "").split(".")[-1]


def build_schema_prompt(db: str, tables: dict[str, Any], row_sample: int) -> str:
    """Build the schema and live row-sample prompt used by the planner."""
    schema_records = {
        _short_table_name(record.get("table_name", "")): record
        for record in load_schema(db)
    }
    parts: list[str] = [f"Database: {db}", ""]
    sample_max_chars = int(
        (os.getenv("E1_ROW_SAMPLE_MAX_CHARS") or "6000").strip() or "6000"
    )
    for table_name, dataframe in tables.items():
        record = schema_records.get(table_name, {})
        columns = list(dataframe.columns)
        recorded_names = list(record.get("column_names") or [])
        recorded_types = list(record.get("column_types") or [])
        type_by_name = dict(zip(recorded_names, recorded_types))
        types = [
            type_by_name.get(column, str(dataframe[column].dtype))
            for column in columns
        ]
        column_description = ", ".join(
            f"{column}:{column_type}" if column_type else str(column)
            for column, column_type in zip(columns, types)
        )
        parts.append(f"Table `{table_name}` ({len(dataframe)} rows)")
        parts.append(f"  columns: {column_description}")
        try:
            sample = dataframe.head(row_sample).to_dict(orient="records")
            serialized = json.dumps(sample, default=str)
            parts.append(f"  sample_rows: {serialized[:sample_max_chars]}")
        except Exception:
            pass
        parts.append("")
    return "\n".join(parts)


def clean_cell(value: Any) -> Any:
    """Conservatively remove answer prose and Markdown from semantic values."""
    import re

    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    markers = list(re.finditer(r"(?is)\banswer\s*:\s*", cleaned))
    if markers:
        cleaned = cleaned[markers[-1].end():].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) > 1:
        cleaned = lines[-1]
    cleaned = re.sub(r"^\s*[#>*\-•]+\s*", "", cleaned)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"[`*_]", "", cleaned)
    return cleaned.strip().strip('"').strip("'").strip()


def clean_result_dataframe(dataframe: Any) -> Any:
    try:
        cleaned = dataframe.copy()
        for column in cleaned.columns:
            if cleaned[column].dtype == object:
                cleaned[column] = cleaned[column].map(clean_cell)
        return cleaned
    except Exception:
        return dataframe
