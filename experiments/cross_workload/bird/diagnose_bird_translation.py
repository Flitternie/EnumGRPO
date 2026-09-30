#!/usr/bin/env python3
"""Run differential SQLite/DuckDB diagnostics for selected BIRD queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

from prepare_bird_duckdb import (
    _bird_translate,
    _audited_boundary_tie_answers,
    _equivalence_mode,
    _execute_duckdb,
    _sqlite_rows,
    evaluate_tables,
    requires_ordering_from_text,
)


def _selected_ids(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("failures", [])
    result: set[int] = set()
    for item in payload:
        value: Any = item.get("question_id") if isinstance(item, dict) else item[0]
        result.add(int(str(value).removeprefix("bird_dev_")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sqlite-root", type=Path, required=True)
    parser.add_argument("--duckdb-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wanted = _selected_ids(args.selection)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    rows = [
        row
        for row in annotations
        if row["difficulty"] == "challenging"
        and (wanted is None or int(row["question_id"]) in wanted)
    ]
    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        qid = int(row["question_id"])
        db_id = str(row["db_id"])
        sql = str(row["SQL"])
        translated = ""
        item: dict[str, Any] = {
            "question_id": f"bird_dev_{qid:04d}",
            "db": db_id,
            "question": row["question"],
            "sql": sql,
        }
        try:
            expected = _sqlite_rows(
                args.sqlite_root / db_id / f"{db_id}_template.sqlite", sql
            )
            translated = _bird_translate(sql)
            connection = duckdb.connect(
                str(args.duckdb_dir / f"{db_id}.duckdb"), read_only=True
            )
            try:
                actual, translated = _execute_duckdb(connection, translated)
                valid_answers = _audited_boundary_tie_answers(
                    qid, connection, translated, expected
                )
            finally:
                connection.close()
            ordered = requires_ordering_from_text(str(row["question"]), sql)
            metrics = evaluate_tables(expected, actual, ordered=ordered)
            if valid_answers:
                sqlite_valid = any(
                    evaluate_tables(expected, answer, ordered=ordered)["sr"]
                    for answer in valid_answers
                )
                duckdb_valid = any(
                    evaluate_tables(actual, answer, ordered=ordered)["sr"]
                    for answer in valid_answers
                )
                equivalent = bool(sqlite_valid and duckdb_valid)
                equivalence = "boundary_tie_variant" if equivalent else "mismatch"
            else:
                equivalent, equivalence = _equivalence_mode(
                    qid, expected, actual, ordered
                )
            item.update(
                {
                    "status": "match" if equivalent else "mismatch",
                    "equivalence": equivalence,
                    "ordered": ordered,
                    "metrics": metrics,
                    "sqlite_row_count": len(expected),
                    "duckdb_row_count": len(actual),
                    "sqlite_sample": expected[:5],
                    "duckdb_sample": actual[:5],
                    "translated_sql": translated,
                }
            )
        except Exception as exc:
            item.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "translated_sql": translated,
                }
            )
        diagnostics.append(item)
        print(f"[{index}/{len(rows)}] {item['question_id']}: {item['status']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = {
        status: sum(item["status"] == status for item in diagnostics)
        for status in ("match", "mismatch", "error")
    }
    print(json.dumps(counts, sort_keys=True))
    return 0 if counts["mismatch"] == 0 and counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
