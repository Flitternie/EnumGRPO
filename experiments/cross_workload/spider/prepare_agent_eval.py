#!/usr/bin/env python3
"""Add official Spider PK/FK metadata to an evaluation JSONL.

The generated file is an agent-facing view of the benchmark.  It does not
contain gold SQL beyond fields already present in the source JSONL, and the
runner continues to expose ordinary DuckDB schema inspection.  Relationships
are appended to ``hint`` because runners/agent.py already forwards that field;
no SWAN runner behavior needs to change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _qualified_column(
    column_names: list[list[Any]], table_names: list[str], column_index: int
) -> str:
    table_index, column_name = column_names[column_index]
    if int(table_index) < 0:
        return str(column_name)
    return f"{table_names[int(table_index)]}.{column_name}"


def _schema_hint(schema: dict[str, Any]) -> str:
    table_names = [str(x) for x in schema["table_names_original"]]
    column_names = schema["column_names_original"]
    primary_keys = [
        _qualified_column(column_names, table_names, int(index))
        for index in schema.get("primary_keys", [])
    ]
    foreign_keys = [
        (
            _qualified_column(column_names, table_names, int(source)),
            _qualified_column(column_names, table_names, int(target)),
        )
        for source, target in schema.get("foreign_keys", [])
    ]

    lines = ["Official Spider schema metadata:"]
    lines.append(
        "Primary keys: " + (", ".join(primary_keys) if primary_keys else "none declared")
    )
    lines.append(
        "Foreign keys: "
        + (
            "; ".join(f"{source} -> {target}" for source, target in foreign_keys)
            if foreign_keys
            else "none declared"
        )
    )
    lines.append(
        "Evaluation output requirement: always save the final CSV, including when the query result is empty."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--tables-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    schemas = {
        str(row["db_id"]): row
        for row in json.loads(args.tables_json.read_text(encoding="utf-8"))
    }
    output: list[dict[str, Any]] = []
    missing: set[str] = set()
    for line in args.input_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        db_id = str(row["db"])
        schema = schemas.get(db_id)
        if schema is None:
            missing.add(db_id)
            continue
        existing = str(row.get("hint") or "").strip()
        metadata = _schema_hint(schema)
        row["hint"] = f"{existing}\n\n{metadata}".strip() if existing else metadata
        output.append(row)

    if missing:
        raise RuntimeError(f"Missing tables.json schemas for: {', '.join(sorted(missing))}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(output)} records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
