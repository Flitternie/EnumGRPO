#!/usr/bin/env python3
"""Audit Spider gold-SQL compatibility after SQLite-to-DuckDB conversion.

Translation alone is not sufficient evidence of compatibility.  This script
executes the original query on SQLite, executes either the original or
sqlglot-transpiled query on DuckDB, and then compares the result tables with
the same ordering and canonicalization rules used by ``eval_swan.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb

from prepare_spider import _json_value, _nested_sql, _spider_hardness


REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swan.evaluation.utils import evaluate_tables, requires_ordering_from_text


def _fetch_sqlite(db_path: Path, sql: str) -> list[list[Any]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        return [
            [_json_value(value) for value in row]
            for row in con.execute(sql).fetchall()
        ]
    finally:
        con.close()


def _fetch_duckdb(db_path: Path, sql: str) -> list[list[Any]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return [
            [_json_value(value) for value in row]
            for row in con.execute(sql).fetchall()
        ]
    finally:
        con.close()


def _translate(sql: str, dialect: str) -> str:
    if dialect == "raw":
        return sql
    import sqlglot
    from sqlglot import exp

    if dialect == "normalized":
        # Spider 1.0 annotations use double quotes for string literals, while
        # DuckDB follows the SQL standard and treats them as identifiers.  A
        # corpus audit confirms that all double-quoted tokens in Spider dev are
        # values, not quoted schema identifiers.
        sql = re.sub(
            r'"([^"\\]*(?:\\.[^"\\]*)*)"',
            lambda match: "'" + match.group(1).replace("'", "''") + "'",
            sql,
        )

        # SQLite permits a selected column to be omitted from GROUP BY.  Make
        # the query standard SQL by adding every non-aggregate projection to
        # GROUP BY.  This preserves the intended result when the projected
        # label is functionally determined by the original grouping key.  If
        # it is not, the differential check rejects the rewrite instead of
        # reproducing SQLite's arbitrary-row behavior.
        tree = sqlglot.parse_one(sql, read="sqlite")
        for select in list(tree.find_all(exp.Select)):
            group = select.args.get("group")
            if not group:
                continue
            grouped = {
                item.sql(dialect="duckdb").casefold()
                for item in group.expressions
            }
            additions = []
            for item in select.expressions:
                core = item.this if isinstance(item, exp.Alias) else item
                core_sql = core.sql(dialect="duckdb").casefold()
                if (
                    not isinstance(core, exp.Star)
                    and core.find(exp.AggFunc) is None
                    and core_sql not in grouped
                ):
                    additions.append(core.copy())
                    grouped.add(core_sql)
            if additions:
                group.set("expressions", list(group.expressions) + additions)
        return tree.sql(dialect="duckdb")

    translated = sqlglot.transpile(sql, read="sqlite", write="duckdb")
    if len(translated) != 1:
        raise ValueError(f"Expected one translated statement, got {len(translated)}")
    return translated[0]


def _features(parsed: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    if len(parsed["from"]["table_units"]) > 1:
        out.add("join")
    if _nested_sql(parsed):
        out.add("nested")
    if parsed["groupBy"]:
        out.add("group_by")
    if parsed["orderBy"]:
        out.add("order_by")
    if any(parsed[key] is not None for key in ("intersect", "except", "union")):
        out.add("set_op")
    if len(parsed["select"][1]) > 1:
        out.add("multi_select")
    return out


def _failure_kind(exc: Exception | None, sql: str) -> str:
    if exc is None:
        return "result_mismatch"
    message = str(exc).casefold()
    if "must appear in the group by" in message:
        return "strict_group_by"
    if "referenced column" in message and '"' in sql:
        return "double_quoted_literal_or_identifier"
    if (
        "no function matches" in message
        or "conversion error" in message
        or "cannot compare values of type" in message
    ):
        return "sqlite_dynamic_typing"
    if "parser error" in message or "syntax error" in message:
        return "syntax"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--duckdb-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument(
        "--hardness",
        action="append",
        choices=("easy", "medium", "hard", "extra"),
        default=[],
        help="Hardness to include; repeatable. Defaults to hard and extra.",
    )
    parser.add_argument(
        "--dialect",
        choices=("raw", "sqlglot", "normalized"),
        default="raw",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    duckdb_dir = args.duckdb_dir.resolve()
    wanted = set(args.hardness or ("hard", "extra"))
    rows = json.loads((source_root / "dev.json").read_text(encoding="utf-8"))

    selected: list[dict[str, Any]] = []
    compatible: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    feature_total: Counter[str] = Counter()
    feature_compatible: Counter[str] = Counter()

    for index, source_row in enumerate(rows):
        hardness = _spider_hardness(source_row["sql"])
        if hardness not in wanted:
            continue
        db_id = str(source_row["db_id"])
        sql = str(source_row["query"])
        question = str(source_row["question"])
        qid = f"spider_dev_{index:04d}"
        feats = _features(source_row["sql"])
        feature_total.update(feats)
        selected.append({"question_id": qid, "db": db_id, "hardness": hardness})

        sqlite_path = source_root / "database" / db_id / f"{db_id}.sqlite"
        duckdb_path = duckdb_dir / f"{db_id}.duckdb"
        translated_sql = sql
        error: Exception | None = None
        duckdb_rows: list[list[Any]] | None = None
        try:
            sqlite_rows = _fetch_sqlite(sqlite_path, sql)
            translated_sql = _translate(sql, args.dialect)
            duckdb_rows = _fetch_duckdb(duckdb_path, translated_sql)
        except Exception as exc:
            error = exc

        is_compatible = False
        if error is None and duckdb_rows is not None:
            ordered = requires_ordering_from_text(question, sql)
            metrics = evaluate_tables(sqlite_rows, duckdb_rows, ordered=ordered)
            is_compatible = bool(metrics["sr"])

        if is_compatible:
            feature_compatible.update(feats)
            compatible.append(
                {
                    "question_id": qid,
                    "db": db_id,
                    "query": question,
                    "hint": "",
                    "sql": sql,
                    "duckdb_gold_sql": translated_sql,
                    "answer": sqlite_rows,
                    "complexity": hardness,
                    "spider_index": index,
                }
            )
        else:
            failures.append(
                {
                    "question_id": qid,
                    "db": db_id,
                    "hardness": hardness,
                    "category": _failure_kind(error, sql),
                    "error": str(error) if error is not None else None,
                    "sql": sql,
                    "translated_sql": translated_sql,
                }
            )

    all_dbs = sorted({row["db"] for row in selected})
    compatible_dbs = sorted({row["db"] for row in compatible})
    report = {
        "benchmark": "Spider 1.0 dev",
        "dialect_mode": args.dialect,
        "hardness": sorted(wanted),
        "selected_count": len(selected),
        "compatible_count": len(compatible),
        "compatible_rate": len(compatible) / len(selected) if selected else 0.0,
        "all_database_count": len(all_dbs),
        "compatible_database_count": len(compatible_dbs),
        "missing_databases": sorted(set(all_dbs) - set(compatible_dbs)),
        "compatible_by_hardness": dict(Counter(row["complexity"] for row in compatible)),
        "feature_counts_all": dict(sorted(feature_total.items())),
        "feature_counts_compatible": dict(sorted(feature_compatible.items())),
        "failure_categories": dict(sorted(Counter(row["category"] for row in failures).items())),
        "failures": failures,
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in compatible:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    printable = {key: value for key, value in report.items() if key != "failures"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
