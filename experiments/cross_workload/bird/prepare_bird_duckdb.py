#!/usr/bin/env python3
"""Convert BIRD dev SQLite assets to DuckDB and build agent evaluation JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SPIDER_DIR = REPO_ROOT / "experiments" / "cross_workload" / "spider"
for path in (str(REPO_ROOT), str(SPIDER_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from prepare_spider import _convert_sqlite_to_duckdb, _json_value
from swan.evaluation.utils import evaluate_tables, requires_ordering_from_text


# These queries have been inspected with their explicit ORDER BY values
# projected.  SQLite and DuckDB produce the same rows and the same sequence of
# sort-key tuples; they differ only within groups whose stated keys are equal.
AUDITED_ORDER_TIE_VARIANTS = {
    11,
    66,
    96,
    127,
    144,
    146,
    153,
    155,
    175,
    1041,
}

# GROUP_CONCAT has no ordering guarantee unless its input is explicitly
# ordered.  These cells contain the same elements in a different engine scan
# order.  Record both engine outputs as valid answers rather than rewriting the
# official SQL.
AUDITED_CONCAT_ORDER_VARIANTS = {
    21: (9, ","),
    92: (10, ","),
    195: (5, ","),
    207: (4, ", "),
}


def _rows(connection: Any, sql: str) -> list[list[Any]]:
    return [[_json_value(value) for value in row] for row in connection.execute(sql).fetchall()]


def _sqlite_rows(path: Path, sql: str) -> list[list[Any]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        return _rows(connection, sql)
    finally:
        connection.close()


def _normalize_concat_cell(value: Any, separator: str) -> Any:
    if not isinstance(value, str):
        return value
    return separator.join(sorted(value.split(separator)))


def _equivalence_mode(
    question_id: int,
    expected: list[list[Any]],
    actual: list[list[Any]],
    ordered: bool,
) -> tuple[bool, str]:
    """Compare results, including only manually audited nondeterministic order."""
    if evaluate_tables(expected, actual, ordered=ordered)["sr"]:
        return True, "exact"

    if question_id in AUDITED_ORDER_TIE_VARIANTS:
        if evaluate_tables(expected, actual, ordered=False)["sr"]:
            return True, "order_tie_variant"

    concat = AUDITED_CONCAT_ORDER_VARIANTS.get(question_id)
    if concat is not None:
        column, separator = concat

        def normalized(rows: list[list[Any]]) -> list[list[Any]]:
            result = [list(row) for row in rows]
            for row in result:
                if column < len(row):
                    row[column] = _normalize_concat_cell(row[column], separator)
            return result

        if evaluate_tables(
            normalized(expected), normalized(actual), ordered=ordered
        )["sr"]:
            return True, "group_concat_order_variant"

    return False, "mismatch"


def _same_answer(
    left: list[list[Any]], right: list[list[Any]], ordered: bool
) -> bool:
    return bool(evaluate_tables(left, right, ordered=ordered)["sr"])


def _audited_boundary_tie_answers(
    question_id: int,
    connection: Any,
    sql: str,
    base_answer: list[list[Any]],
) -> list[list[list[Any]]]:
    """Enumerate all valid LIMIT-1 answers for audited nested top-1 ties."""
    if question_id not in {190, 212}:
        return []
    tree = sqlglot.parse_one(sql, read="duckdb")
    root = tree if isinstance(tree, exp.Select) else None
    candidates = [
        select
        for select in tree.find_all(exp.Select)
        if select.args.get("order") is not None and select.args.get("limit") is not None
    ]
    expected_outer_columns = 9 if question_id == 190 else 1
    if (
        root is None
        or len(root.expressions) != expected_outer_columns
        or len(candidates) != 1
    ):
        raise ValueError(f"Unexpected BIRD {question_id} top-1 query shape")
    target = candidates[0].copy()
    if tree.args.get("with_") is not None:
        target.set("with_", tree.args["with_"].copy())
    limit = target.args.get("limit")
    if limit is None or int(limit.expression.this) != 1:
        raise ValueError(f"Unexpected BIRD {question_id} LIMIT")
    output_columns = len(target.expressions)
    order = target.args["order"]
    for index, ordered_expression in enumerate(order.expressions):
        target.append(
            "expressions",
            exp.alias_(ordered_expression.this.copy(), f"__tie_key_{index}"),
        )
    target.set("limit", None)
    expanded = _rows(connection, target.sql(dialect="duckdb"))
    if not expanded:
        return [[]]
    boundary = expanded[0][output_columns:]
    answers: list[list[list[Any]]] = []
    for row in expanded:
        if row[output_columns:] != boundary:
            break
        if question_id == 190:
            if len(base_answer) != 1 or not base_answer[0]:
                raise ValueError("Unexpected BIRD 190 answer shape")
            answer_row = list(base_answer[0])
            answer_row[-1] = row[0]
            answer = [answer_row]
        else:
            answer = [[row[0]]]
        if not any(_same_answer(answer, existing, ordered=True) for existing in answers):
            answers.append(answer)
    return answers


def _schema_hint(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        primary: list[str] = []
        foreign: list[str] = []
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            for row in connection.execute(f"PRAGMA table_info({quoted})"):
                if int(row[5] or 0) > 0:
                    primary.append(f"{table}.{row[1]}")
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted})"):
                foreign.append(f"{table}.{row[3]} -> {row[2]}.{row[4]}")
    finally:
        connection.close()
    return "\n".join(
        [
            "Official BIRD schema metadata:",
            "Primary keys: " + (", ".join(primary) if primary else "none declared"),
            "Foreign keys: " + ("; ".join(foreign) if foreign else "none declared"),
            "Evaluation output requirement: always save the final CSV, including when the query result is empty.",
        ]
    )


def _install_sqlite_compatibility_macros(path: Path) -> None:
    """Install deterministic helpers that reproduce SQLite coercion rules."""
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            r"""
            CREATE OR REPLACE MACRO sqlite_real(x) AS
              CASE WHEN x IS NULL THEN NULL ELSE
                COALESCE(
                  TRY_CAST(
                    regexp_extract(
                      trim(CAST(x AS VARCHAR)),
                      '^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',
                      0
                    ) AS DOUBLE
                  ),
                  0.0
                )
              END
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE MACRO sqlite_int(x) AS
              CAST(trunc(sqlite_real(x)) AS BIGINT)
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE MACRO sqlite_julianday(x) AS
              CASE
                WHEN lower(CAST(x AS VARCHAR)) = 'now' THEN julian(current_timestamp)
                ELSE julian(TRY_CAST(x AS TIMESTAMP))
              END
            """
        )
    finally:
        connection.close()


def _bird_translate(sql: str) -> str:
    """Translate SQLite SQL while retaining SQLite's documented coercions."""
    tree = sqlglot.parse_one(sql, read="sqlite")

    def rewrite_temporal(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.TsOrDsToTimestamp):
            value = node.this
            if isinstance(value, exp.Literal) and value.is_string and value.this.casefold() == "now":
                return exp.CurrentTimestamp()
            return exp.TryCast(
                this=value.copy(),
                to=exp.DataType.build(exp.DataType.Type.TIMESTAMP),
            )
        if isinstance(node, exp.Anonymous):
            name = str(node.this).casefold()
            if name == "julianday":
                return exp.Anonymous(
                    this="sqlite_julianday",
                    expressions=[item.copy() for item in node.expressions],
                )
            if name == "datetime" and not node.expressions:
                return exp.CurrentTimestamp()
        if isinstance(node, exp.Date):
            value = node.this
            if isinstance(value, exp.Literal) and value.is_string and value.this.casefold() == "now":
                return exp.CurrentDate()
        return node

    tree = tree.transform(rewrite_temporal)
    tree = tree.transform(rewrite_temporal)

    # This is deliberately a separate pass: replacing an outer numeric CAST
    # must not prevent traversal into an inner STRFTIME/JULIANDAY expression.
    def rewrite_numeric_cast(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Cast):
            return node
        target = node.args.get("to")
        kind = target.this if isinstance(target, exp.DataType) else None
        if kind in {exp.DataType.Type.FLOAT, exp.DataType.Type.DOUBLE}:
            return exp.Anonymous(this="sqlite_real", expressions=[node.this.copy()])
        if kind in {
            exp.DataType.Type.TINYINT,
            exp.DataType.Type.SMALLINT,
            exp.DataType.Type.INT,
            exp.DataType.Type.BIGINT,
        }:
            return exp.Anonymous(this="sqlite_int", expressions=[node.this.copy()])
        return node

    tree = tree.transform(rewrite_numeric_cast)

    # DuckDB's STRFTIME returns VARCHAR, as SQLite does.  Add numeric coercion
    # only where SQLite would apply it implicitly; casting a projected year
    # would incorrectly change the answer type from text to integer.
    def cast_year(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.TimeToStr):
            fmt = node.args.get("format")
            if isinstance(fmt, exp.Literal) and fmt.this == "%Y":
                parent = node.parent
                needs_number = isinstance(
                    parent, (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)
                )
                if isinstance(parent, exp.Between):
                    bounds = (parent.args.get("low"), parent.args.get("high"))
                    needs_number = all(
                        isinstance(value, exp.Literal) and not value.is_string
                        for value in bounds
                    )
                if isinstance(parent, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                    other = parent.expression if parent.this is node else parent.this
                    needs_number = isinstance(other, exp.Literal) and not other.is_string
                if needs_number:
                    return exp.Cast(
                        this=node.copy(),
                        to=exp.DataType.build(exp.DataType.Type.BIGINT),
                    )
        return node

    tree = tree.transform(cast_year)

    # SQLite truncates division when both operands have integer storage
    # classes, while DuckDB's `/` always returns a fractional value.  BIRD's
    # aggregate percentage query guards an integer SUM denominator with CASE;
    # retain SQLite's division before the subsequent multiplication by 100.
    def preserve_integer_aggregate_division(node: exp.Expression) -> exp.Expression:
        if (
            isinstance(node, exp.Div)
            and isinstance(node.this, exp.Sum)
            and isinstance(node.expression, exp.Case)
            and node.find(exp.Anonymous) is None
        ):
            return exp.Anonymous(this="TRUNC", expressions=[node.copy()])
        return node

    tree = tree.transform(preserve_integer_aggregate_division)

    # SQLite represents predicates as 0/1 in arithmetic expressions, whereas
    # DuckDB keeps BOOLEAN distinct. Make only that coercion explicit; broad
    # arithmetic wrapping can obscure aggregate expressions from GROUP BY.
    predicates = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between)

    def numeric_operand(value: exp.Expression) -> exp.Expression:
        if isinstance(value, exp.Anonymous) and str(value.this).casefold() in {
            "sqlite_real",
            "sqlite_int",
            "sqlite_julianday",
        }:
            return value
        if isinstance(value, predicates):
            value = exp.Case().when(value.copy(), exp.Literal.number(1)).else_(
                exp.Literal.number(0)
            )
        return exp.Anonymous(this="sqlite_real", expressions=[value.copy()])

    for node in tree.find_all(exp.Sub):
        right = node.expression
        predicate = right.this if isinstance(right, exp.Paren) else right
        if isinstance(predicate, predicates):
            node.set("expression", numeric_operand(predicate))
        if isinstance(node.this, (exp.CurrentTimestamp, exp.CurrentDate)):
            node.set("this", numeric_operand(node.this))
            node.set("expression", numeric_operand(node.expression))

    # A numeric BETWEEN over a text-affinity SQLite column also invokes
    # coercion. This occurs in BIRD's YYYYMM fields.
    for node in tree.find_all(exp.Between):
        low = node.args.get("low")
        high = node.args.get("high")
        if isinstance(low, exp.Literal) and not low.is_string and isinstance(high, exp.Literal) and not high.is_string:
            node.set("this", numeric_operand(node.this))

    # SQLite permits aggregate queries without GROUP BY to project arbitrary
    # row values. ANY_VALUE is DuckDB's explicit equivalent. Queries that do
    # have GROUP BY are repaired from DuckDB's precise binder diagnostics in
    # _execute_duckdb below, avoiding speculative grouping changes.
    def has_local_unwindowed_aggregate(select: exp.Select) -> bool:
        """Return whether this SELECT owns a relational aggregate.

        SQLGlot models ranking functions as aggregate-function nodes, and a
        recursive ``find`` also sees aggregates owned by scalar subqueries.
        Neither makes the surrounding SELECT an aggregate query.
        """
        for node in select.find_all(exp.AggFunc):
            if node.find_ancestor(exp.Select) is not select:
                continue
            ancestor = node.parent
            windowed = False
            while ancestor is not None and ancestor is not select:
                if isinstance(ancestor, exp.Window):
                    windowed = True
                    break
                ancestor = ancestor.parent
            if not windowed:
                return True
        return False

    for select in list(tree.find_all(exp.Select)):
        group = select.args.get("group")
        has_aggregate = has_local_unwindowed_aggregate(select)
        if not group and has_aggregate:
            for index, item in enumerate(list(select.expressions)):
                core = item.this if isinstance(item, exp.Alias) else item
                if (
                    core.find(exp.AggFunc) is None
                    and core.find(exp.Window) is None
                    and core.find(exp.Star) is None
                ):
                    wrapped = exp.Anonymous(this="ANY_VALUE", expressions=[core.copy()])
                    if isinstance(item, exp.Alias):
                        wrapped = exp.alias_(wrapped, item.alias)
                    select.expressions[index] = wrapped
            continue
        if not group:
            continue

    translated = tree.sql(dialect="duckdb")
    # AT is a DuckDB keyword but occurs as a table alias in BIRD financial SQL.
    translated = re.sub(r"\bAS at\b", 'AS "at"', translated, flags=re.IGNORECASE)
    translated = re.sub(r"(?<![\w\"])(at)\.", '"at".', translated, flags=re.IGNORECASE)
    return translated


_MISSING_GROUP_COLUMN = re.compile(
    r'column "([^"]+)" must appear in the GROUP BY clause', re.IGNORECASE
)


def _repair_missing_group_column(sql: str, error: str) -> str | None:
    """Make SQLite's loose GROUP BY projection explicit with ANY_VALUE.

    Adding the missing column to GROUP BY changes group cardinality and is not
    equivalent to SQLite.  SQLite chooses one value from each existing group;
    DuckDB's ANY_VALUE expresses that behavior directly.
    """
    match = _MISSING_GROUP_COLUMN.search(error)
    if not match:
        return None
    missing = match.group(1).casefold()
    tree = sqlglot.parse_one(sql, read="duckdb")
    for select in reversed(list(tree.find_all(exp.Select))):
        group = select.args.get("group")
        if not group:
            continue
        grouped = {
            item.sql(dialect="duckdb").casefold() for item in group.expressions
        }
        replacements: list[exp.Column] = []
        for column in list(select.find_all(exp.Column)):
            if column.name.casefold() != missing:
                continue
            if column.find_ancestor(exp.Select) is not select:
                continue
            rendered = column.sql(dialect="duckdb").casefold()
            if rendered in grouped:
                continue
            clause = column
            while clause.parent is not None and clause.parent is not select:
                clause = clause.parent
            # Binder diagnostics name a column but not its clause.  Rewriting
            # the same-named column in JOIN/WHERE changes row selection and can
            # even make aggregates illegal there; SQLite's loose grouping is
            # relevant only to projected, ordered, or post-group expressions.
            if clause.arg_key in {"from_", "joins", "where", "group"}:
                continue
            ancestor = column.parent
            already_aggregated = False
            while ancestor is not None and ancestor is not select:
                if isinstance(ancestor, exp.AggFunc):
                    already_aggregated = True
                    break
                ancestor = ancestor.parent
            if not already_aggregated:
                replacements.append(column)
        if replacements:
            for column in replacements:
                column.replace(
                    exp.Anonymous(this="ANY_VALUE", expressions=[column.copy()])
                )
            return tree.sql(dialect="duckdb")
    return None


def _execute_duckdb(connection: Any, sql: str) -> tuple[list[list[Any]], str]:
    current = sql
    for _ in range(20):
        try:
            return _rows(connection, current), current
        except duckdb.BinderException as exc:
            repaired = _repair_missing_group_column(current, str(exc))
            if repaired is None or repaired == current:
                raise
            current = repaired
    raise RuntimeError("Exceeded GROUP BY compatibility repair limit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sqlite-root", type=Path, required=True)
    parser.add_argument("--duckdb-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--difficulty", default="challenging")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    selected = [row for row in source if row["difficulty"] == args.difficulty]
    db_ids = sorted({str(row["db_id"]) for row in selected})
    args.duckdb_dir.mkdir(parents=True, exist_ok=True)
    schema_hints: dict[str, str] = {}
    for db_id in db_ids:
        sqlite_path = args.sqlite_root / db_id / f"{db_id}_template.sqlite"
        _convert_sqlite_to_duckdb(
            sqlite_path,
            args.duckdb_dir / f"{db_id}.duckdb",
            args.force,
        )
        _install_sqlite_compatibility_macros(args.duckdb_dir / f"{db_id}.duckdb")
        schema_hints[db_id] = _schema_hint(sqlite_path)

    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    semantic_variants: list[dict[str, Any]] = []
    for row in selected:
        numeric_qid = int(row["question_id"])
        qid = f"bird_dev_{numeric_qid:04d}"
        db_id = str(row["db_id"])
        sql = str(row["SQL"])
        sqlite_path = args.sqlite_root / db_id / f"{db_id}_template.sqlite"
        translated = sql
        valid_answers: list[list[list[Any]]] = []
        try:
            expected = _sqlite_rows(sqlite_path, sql)
            translated = _bird_translate(sql)
            connection = duckdb.connect(
                str(args.duckdb_dir / f"{db_id}.duckdb"), read_only=True
            )
            try:
                actual, translated = _execute_duckdb(connection, translated)
                valid_answers = _audited_boundary_tie_answers(
                    numeric_qid, connection, translated, expected
                )
            finally:
                connection.close()
            ordered = requires_ordering_from_text(str(row["question"]), sql)
            if valid_answers:
                sqlite_valid = any(
                    _same_answer(expected, answer, ordered) for answer in valid_answers
                )
                duckdb_valid = any(
                    _same_answer(actual, answer, ordered) for answer in valid_answers
                )
                equivalent = sqlite_valid and duckdb_valid
                equivalence = "boundary_tie_variant" if equivalent else "mismatch"
            else:
                equivalent, equivalence = _equivalence_mode(
                    numeric_qid, expected, actual, ordered
                )
            if not equivalent:
                raise ValueError("SQLite and DuckDB results differ")
        except Exception as exc:
            failures.append(
                {
                    "question_id": qid,
                    "db": db_id,
                    "error": str(exc),
                    "sql": sql,
                    "translated_sql": translated,
                }
            )
            continue

        evidence = str(row.get("evidence") or "").strip()
        hint = schema_hints[db_id]
        if evidence:
            hint = f"BIRD evidence: {evidence}\n\n{hint}"
        record = {
            "question_id": qid,
            "db": db_id,
            "query": str(row["question"]),
            "hint": hint,
            "sql": sql,
            "duckdb_gold_sql": translated,
            "answer": expected,
            "complexity": str(row["difficulty"]),
            "bird_question_id": numeric_qid,
        }
        if valid_answers:
            answers = [expected] + [
                answer
                for answer in valid_answers
                if not _same_answer(expected, answer, ordered)
            ]
            for answer_index, answer in enumerate(answers[1:], start=1):
                suffix = chr(ord("a") + answer_index)
                record[f"answer_{suffix}"] = answer
        elif equivalence != "exact":
            record["answer_b"] = actual
        if equivalence != "exact":
            semantic_variants.append(
                {
                    "question_id": qid,
                    "kind": equivalence,
                    "sqlite_rows": len(expected),
                    "duckdb_rows": len(actual),
                    "valid_answer_count": (
                        len(valid_answers) if valid_answers else 2
                    ),
                }
            )
        output.append(record)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    report = {
        "benchmark": "BIRD-SQL dev 2025-11-06",
        "difficulty": args.difficulty,
        "candidate_count": len(selected),
        "included_count": len(output),
        "failure_count": len(failures),
        "exact_match_count": len(output) - len(semantic_variants),
        "semantic_variant_count": len(semantic_variants),
        "semantic_variants": semantic_variants,
        "databases": dict(Counter(row["db"] for row in output)),
        "failures": failures,
    }
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if len(output) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
