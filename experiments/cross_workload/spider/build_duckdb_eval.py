#!/usr/bin/env python3
"""Build a DuckDB-native Spider hard/extra evaluation set.

The policy mirrors the SWAN data shipped with this repository:

* adapt SQL to the target engine explicitly;
* preserve every valid boundary-tie result as answer/answer_b/...;
* exclude questions whose requested answer is not semantically determined.

No model output is consulted while constructing the benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

from audit_duckdb_compat import _translate
from prepare_spider import _json_value, _spider_hardness


REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swan.evaluation.utils import evaluate_tables, requires_ordering_from_text


# These corrections are intentionally explicit and reviewable.  They cover
# only SQLite/DuckDB typing differences; they do not change task semantics.
SQL_OVERRIDES: dict[int, tuple[str, str]] = {
    24: ("type_cast", "SELECT T2.name, T2.capacity FROM concert T1 JOIN stadium T2 ON TRY_CAST(T1.stadium_id AS BIGINT)=TRY_CAST(T2.stadium_id AS BIGINT) WHERE TRY_CAST(T1.year AS INTEGER)>=2014 GROUP BY T2.stadium_id,T2.name,T2.capacity ORDER BY count(*) DESC LIMIT 1"),
    25: ("type_cast", "SELECT T2.name, T2.capacity FROM concert T1 JOIN stadium T2 ON TRY_CAST(T1.stadium_id AS BIGINT)=TRY_CAST(T2.stadium_id AS BIGINT) WHERE TRY_CAST(T1.year AS INTEGER)>2013 GROUP BY T2.stadium_id,T2.name,T2.capacity ORDER BY count(*) DESC LIMIT 1"),
    28: ("fk_type_cast", "SELECT name FROM stadium WHERE TRY_CAST(stadium_id AS BIGINT) NOT IN (SELECT TRY_CAST(stadium_id AS BIGINT) FROM concert)"),
    29: ("fk_type_cast", "SELECT name FROM stadium WHERE TRY_CAST(stadium_id AS BIGINT) NOT IN (SELECT TRY_CAST(stadium_id AS BIGINT) FROM concert)"),
    281: ("fk_type_cast", "SELECT name FROM employee WHERE TRY_CAST(Employee_ID AS BIGINT) NOT IN (SELECT TRY_CAST(Employee_ID AS BIGINT) FROM evaluation)"),
    282: ("fk_type_cast", "SELECT name FROM employee WHERE TRY_CAST(Employee_ID AS BIGINT) NOT IN (SELECT TRY_CAST(Employee_ID AS BIGINT) FROM evaluation)"),
    418: ("type_cast", "SELECT name FROM museum WHERE num_of_staff > (SELECT min(num_of_staff) FROM museum WHERE TRY_CAST(open_year AS INTEGER)>2010)"),
    426: ("type_cast", "SELECT t1.name FROM visitor t1 JOIN visit t2 ON TRY_CAST(t1.id AS BIGINT)=TRY_CAST(t2.visitor_id AS BIGINT) JOIN museum t3 ON t3.Museum_ID=t2.Museum_ID WHERE TRY_CAST(t3.open_year AS INTEGER)<2009 INTERSECT SELECT t1.name FROM visitor t1 JOIN visit t2 ON TRY_CAST(t1.id AS BIGINT)=TRY_CAST(t2.visitor_id AS BIGINT) JOIN museum t3 ON t3.Museum_ID=t2.Museum_ID WHERE TRY_CAST(t3.open_year AS INTEGER)>2011"),
    427: ("type_cast", "SELECT count(*) FROM visitor WHERE TRY_CAST(id AS BIGINT) NOT IN (SELECT TRY_CAST(t2.visitor_id AS BIGINT) FROM museum t1 JOIN visit t2 ON t1.Museum_ID=t2.Museum_ID WHERE TRY_CAST(t1.open_year AS INTEGER)>2010)"),
    920: ("type_cast", "SELECT avg(TRY_CAST(age AS DOUBLE)) FROM Dogs WHERE dog_id IN (SELECT dog_id FROM Treatments)"),
    921: ("type_cast", "SELECT avg(TRY_CAST(age AS DOUBLE)) FROM Dogs WHERE dog_id IN (SELECT dog_id FROM Treatments)"),
    974: ("type_cast", "SELECT count(*) FROM Dogs WHERE TRY_CAST(age AS DOUBLE) < (SELECT avg(TRY_CAST(age AS DOUBLE)) FROM Dogs)"),
    975: ("type_cast", "SELECT count(*) FROM Dogs WHERE TRY_CAST(age AS DOUBLE) < (SELECT avg(TRY_CAST(age AS DOUBLE)) FROM Dogs)"),
}


SEMANTIC_SQL_OVERRIDES: dict[int, str] = {
    500: "SELECT T2.id, T2.name FROM death T1 JOIN ship T2 ON T1.caused_by_ship_id=T2.id GROUP BY T2.id,T2.name ORDER BY SUM(COALESCE(T1.injured,0)) DESC LIMIT 1",
    944: "SELECT DISTINCT T1.first_name,T1.last_name FROM Professionals T1 JOIN Treatments T2 ON T1.professional_id=T2.professional_id WHERE cost_of_treatment < (SELECT avg(cost_of_treatment) FROM Treatments)",
    945: "SELECT DISTINCT T1.first_name,T1.last_name FROM Professionals T1 JOIN Treatments T2 ON T1.professional_id=T2.professional_id WHERE cost_of_treatment < (SELECT avg(cost_of_treatment) FROM Treatments)",
}


# DuckDB spellings that preserve the official Spider SQL behavior, including
# known annotation mistakes.  These are used by --semantic-policy preserve;
# they are dialect adaptations, not corrections to the benchmark semantics.
PRESERVE_SQL_OVERRIDES: dict[int, str] = {
    500: "SELECT T2.id, T2.name FROM death T1 JOIN ship T2 ON T1.caused_by_ship_id=T2.id GROUP BY T2.id,T2.name ORDER BY COUNT(*) DESC LIMIT 1",
    944: "SELECT DISTINCT T1.first_name,T1.last_name FROM Professionals T1 CROSS JOIN Treatments T2 WHERE cost_of_treatment < (SELECT avg(cost_of_treatment) FROM Treatments)",
    945: "SELECT DISTINCT T1.first_name,T1.last_name FROM Professionals T1 CROSS JOIN Treatments T2 WHERE cost_of_treatment < (SELECT avg(cost_of_treatment) FROM Treatments)",
}


SEMANTIC_ISSUES: dict[int, str] = {
    500: "official SQL counts incident rows but the question asks for total injuries",
    944: "official SQL omits the Professionals-Treatments join condition",
    945: "official SQL omits the Professionals-Treatments join condition",
}


# winner_rank_points changes from match to match, while the question specifies
# neither a date nor which observation to report.  This cannot be repaired by
# dialect conversion or by finite boundary-tie handling without inventing a
# new task definition.
EXCLUSIONS: dict[int, str] = {
    463: "winner_rank_points is not functionally determined by winner_name",
    464: "winner_rank_points is not functionally determined by winner_name",
}


def _rows(conn: Any, sql: str) -> list[list[Any]]:
    return [[_json_value(value) for value in row] for row in conn.execute(sql).fetchall()]


def _same(a: list[list[Any]], b: list[list[Any]], *, ordered: bool) -> bool:
    return bool(evaluate_tables(a, b, ordered=ordered)["sr"])


def _dedupe_answers(
    answers: list[list[list[Any]]], *, ordered: bool
) -> list[list[list[Any]]]:
    out: list[list[list[Any]]] = []
    for answer in answers:
        if not any(_same(existing, answer, ordered=ordered) for existing in out):
            out.append(answer)
    return out


def _answer_sort_key(answer: list[list[Any]]) -> str:
    return json.dumps(answer, ensure_ascii=True, sort_keys=True, default=str)


def _top1_answers(conn: Any, sql: str) -> list[list[list[Any]]] | None:
    """Return all top-1 boundary-tie answers, or None for a non-top-1 query."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    if not isinstance(tree, exp.Select):
        return None
    order = tree.args.get("order")
    limit = tree.args.get("limit")
    if not order or not limit:
        return None
    try:
        if int(limit.expression.this) != 1:
            return None
    except Exception:
        return None

    output_columns = len(tree.expressions)
    for index, ordered_expr in enumerate(order.expressions):
        tree.append(
            "expressions",
            exp.alias_(ordered_expr.this.copy(), f"__tie_key_{index}"),
        )
    tree.set("limit", None)
    expanded = _rows(conn, tree.sql(dialect="duckdb"))
    if not expanded:
        return [[]]
    boundary = expanded[0][output_columns:]
    candidates: list[list[list[Any]]] = []
    for row in expanded:
        if row[output_columns:] != boundary:
            break
        candidates.append([row[:output_columns]])
    return _dedupe_answers(candidates, ordered=True)


def _ordered_tie_answers(index: int, rows: list[list[Any]]) -> list[list[list[Any]]]:
    if index not in {401, 402}:
        return [rows]
    # Only teacher name is ordered.  Rows for the same teacher may appear in
    # any order; enumerate those permutations exactly as SWAN stores answer_b.
    groups: list[list[list[Any]]] = []
    for _, group in itertools.groupby(rows, key=lambda row: row[0]):
        groups.append(list(group))
    choices = [list(itertools.permutations(group)) for group in groups]
    return [
        [list(row) for group_variant in variant for row in group_variant]
        for variant in itertools.product(*choices)
    ]


def _quoted_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _rarest_breed_answers(conn: Any) -> list[list[list[Any]]]:
    breeds = _rows(
        conn,
        "SELECT breed_code, COUNT(*) AS n FROM Dogs GROUP BY breed_code ORDER BY n ASC",
    )
    if not breeds:
        return [[]]
    minimum = breeds[0][1]
    answers = []
    for breed, count in breeds:
        if count != minimum:
            break
        answers.append(
            _rows(
                conn,
                "SELECT T1.name, T2.date_of_treatment FROM Dogs T1 "
                "JOIN Treatments T2 ON T1.dog_id=T2.dog_id "
                f"WHERE T1.breed_code={_quoted_literal(breed)}",
            )
        )
    return answers


def _winner_rank_answers(conn: Any) -> list[list[list[Any]]]:
    counts = _rows(
        conn,
        "SELECT winner_name, COUNT(*) AS n FROM matches "
        "GROUP BY winner_name ORDER BY n DESC",
    )
    if not counts:
        return [[]]
    maximum = counts[0][1]
    answers: list[list[list[Any]]] = []
    for winner, count in counts:
        if count != maximum:
            break
        points = _rows(
            conn,
            "SELECT DISTINCT winner_rank_points FROM matches "
            f"WHERE winner_name={_quoted_literal(winner)}",
        )
        for point_row in points:
            answers.append([[winner, point_row[0]]])
    return answers


def _sqlite_answer(db_path: Path, sql: str) -> list[list[Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        return _rows(conn, sql)
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--duckdb-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-official-jsonl", type=Path)
    parser.add_argument(
        "--hardness",
        action="append",
        choices=("easy", "medium", "hard", "extra"),
        default=[],
        help="Hardness to include; repeatable. Defaults to hard and extra.",
    )
    parser.add_argument(
        "--semantic-policy",
        choices=("exclude", "repair", "preserve"),
        default="exclude",
        help=(
            "Exclude known bad official SQL, repair its semantics, or preserve "
            "its original behavior with DuckDB-compatible syntax."
        ),
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    duckdb_dir = args.duckdb_dir.resolve()
    source_rows = json.loads((source_root / "dev.json").read_text(encoding="utf-8"))
    wanted = set(args.hardness or ("hard", "extra"))
    output: list[dict[str, Any]] = []
    official_output: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    adaptations: Counter[str] = Counter()
    alternative_histogram: Counter[int] = Counter()

    for index, source in enumerate(source_rows):
        hardness = _spider_hardness(source["sql"])
        if hardness not in wanted:
            continue
        qid = f"spider_dev_{index:04d}"
        db_id = str(source["db_id"])
        original = _sqlite_answer(
            source_root / "database" / db_id / f"{db_id}.sqlite",
            str(source["query"]),
        )
        official_output.append(
            {
                "question_id": qid,
                "db": db_id,
                "query": str(source["question"]),
                "hint": "",
                "sql": str(source["query"]),
                "answer": original,
                "complexity": hardness,
                "spider_index": index,
            }
        )
        if index in EXCLUSIONS and args.semantic_policy != "preserve":
            excluded.append(
                {
                    "question_id": qid,
                    "db": source["db_id"],
                    "complexity": hardness,
                    "reason": EXCLUSIONS[index],
                }
            )
            continue

        if index in SEMANTIC_ISSUES and args.semantic_policy == "exclude":
            excluded.append(
                {
                    "question_id": qid,
                    "db": db_id,
                    "complexity": hardness,
                    "reason": SEMANTIC_ISSUES[index],
                }
            )
            continue

        if index in EXCLUSIONS and args.semantic_policy == "preserve":
            adaptation = "nonfunctional_projection_alternatives"
            sql = (
                "SELECT winner_name, MIN(winner_rank_points) AS winner_rank_points "
                "FROM matches GROUP BY winner_name "
                "ORDER BY COUNT(*) DESC, winner_name ASC LIMIT 1"
            )
        elif index in SEMANTIC_SQL_OVERRIDES and args.semantic_policy == "repair":
            adaptation, sql = "semantic_gold_fix", SEMANTIC_SQL_OVERRIDES[index]
        elif index in PRESERVE_SQL_OVERRIDES and args.semantic_policy == "preserve":
            adaptation, sql = "official_semantics_preserved", PRESERVE_SQL_OVERRIDES[index]
        else:
            adaptation, sql = SQL_OVERRIDES.get(
                index,
                ("dialect_normalization", _translate(str(source["query"]), "normalized")),
            )
        duck = duckdb.connect(str(duckdb_dir / f"{db_id}.duckdb"), read_only=True)
        try:
            primary = _rows(duck, sql)
            if index in {463, 464} and args.semantic_policy == "preserve":
                answers = _winner_rank_answers(duck)
            elif index in {954, 955}:
                answers = _rarest_breed_answers(duck)
                adaptation = "nested_boundary_ties"
            else:
                answers = _top1_answers(duck, sql) or _ordered_tie_answers(index, primary)
        except Exception as exc:
            excluded.append(
                {
                    "question_id": qid,
                    "db": db_id,
                    "complexity": hardness,
                    "reason": f"duckdb execution failed: {exc}",
                }
            )
            duck.close()
            continue
        finally:
            try:
                duck.close()
            except Exception:
                pass

        ordered = requires_ordering_from_text(str(source["question"]), str(source["query"]))
        if not ordered:
            answers = [
                sorted(answer, key=lambda row: json.dumps(row, ensure_ascii=True, default=str))
                for answer in answers
            ]
        answers = _dedupe_answers(answers, ordered=ordered)
        answers.sort(key=_answer_sort_key)
        semantic_fix = adaptation == "semantic_gold_fix"
        if not semantic_fix and not any(_same(original, answer, ordered=ordered) for answer in answers):
            excluded.append(
                {
                    "question_id": qid,
                    "db": db_id,
                    "complexity": hardness,
                    "reason": "SQLite answer is not among deterministic DuckDB valid answers",
                    "duckdb_gold_sql": sql,
                }
            )
            continue

        if not semantic_fix:
            # Keep the official SQLite result as the primary answer and add
            # only the other valid tied results as SWAN-style alternatives.
            answers = [original] + [
                answer for answer in answers
                if not _same(original, answer, ordered=ordered)
            ]

        adaptations[adaptation] += 1
        if len(answers) > 1:
            adaptations["multiple_valid_answers"] += 1
        alternative_histogram[len(answers)] += 1

        record = {
            "question_id": qid,
            "db": db_id,
            "query": str(source["question"]),
            "hint": "",
            "sql": str(source["query"]),
            "duckdb_gold_sql": sql,
            "answer": answers[0],
            "complexity": hardness,
            "spider_index": index,
            "adaptation": adaptation,
        }
        for answer_index, answer in enumerate(answers[1:], start=1):
            suffix = chr(ord("a") + answer_index)
            record[f"answer_{suffix}"] = answer
        output.append(record)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    if args.output_official_jsonl:
        args.output_official_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_official_jsonl.open("w", encoding="utf-8") as handle:
            for row in official_output:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    report = {
        "benchmark": "Spider 1.0 dev adapted for DuckDB",
        "hardness": sorted(wanted),
        "semantic_policy": args.semantic_policy,
        "candidate_count": len(official_output),
        "included_count": len(output),
        "excluded_count": len(excluded),
        "included_by_hardness": dict(Counter(row["complexity"] for row in output)),
        "excluded_by_hardness": dict(Counter(row["complexity"] for row in excluded)),
        "adaptations": dict(sorted(adaptations.items())),
        "answer_set_count_histogram": {
            str(key): value for key, value in sorted(alternative_histogram.items())
        },
        "excluded": excluded,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not any("execution failed" in row["reason"] for row in excluded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
