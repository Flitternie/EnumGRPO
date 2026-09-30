#!/usr/bin/env python3
"""Prepare the complete text-only SemBench Movie scenario for the DB agent."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


QUERY_COLUMNS: dict[int, list[str]] = {
    1: ["reviewId"],
    2: ["reviewId"],
    3: ["positive_review_cnt"],
    4: ["positivity_ratio"],
    5: ["id", "reviewId1", "reviewId2"],
    6: ["id", "reviewId1", "reviewId2"],
    7: ["id", "reviewId1", "reviewId2"],
    8: ["scoreSentiment", "count"],
    9: ["reviewId", "reviewScore"],
    10: ["movieId", "movieScore"],
}

HIDDEN_REVIEW_COLUMNS = {"originalScore", "reviewState", "scoreSentiment"}
HIDDEN_MOVIE_COLUMNS = {"audienceScore", "tomatoMeter"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sembench-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _source_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except Exception:
        return None


def _read_queries(root: Path) -> list[dict[str, Any]]:
    query_root = root / "files" / "movie" / "query" / "natural_language"
    records: list[dict[str, Any]] = []
    for qid in range(1, 11):
        path = query_root / f"Q{qid}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty query: {path}")
        records.append(
            {
                "question_id": f"Q{qid}",
                "db": "sembench_movie",
                "query": text,
                "hint": "",
                "required_columns": QUERY_COLUMNS[qid],
            }
        )
    return records


def _write_agent_db(data_root: Path, db_path: Path) -> dict[str, Any]:
    movies_csv = data_root / "Movies.csv"
    reviews_csv = data_root / "Reviews.csv"
    if not movies_csv.is_file() or not reviews_csv.is_file():
        raise FileNotFoundError(f"Missing Movie sf_2000 data under {data_root}")

    movies = pd.read_csv(movies_csv)
    reviews = pd.read_csv(reviews_csv)
    missing_reviews = sorted(HIDDEN_REVIEW_COLUMNS - set(reviews.columns))
    missing_movies = sorted(HIDDEN_MOVIE_COLUMNS - set(movies.columns))
    if missing_reviews or missing_movies:
        raise ValueError(
            "Expected SemBench columns are missing: "
            f"Reviews={missing_reviews}, Movies={missing_movies}"
        )

    agent_movies = movies.drop(columns=sorted(HIDDEN_MOVIE_COLUMNS))
    agent_reviews = reviews.drop(columns=sorted(HIDDEN_REVIEW_COLUMNS))

    con = duckdb.connect(str(db_path))
    try:
        con.register("agent_movies", agent_movies)
        con.register("agent_reviews", agent_reviews)
        con.execute("CREATE TABLE Movies AS SELECT * FROM agent_movies")
        con.execute("CREATE TABLE Reviews AS SELECT * FROM agent_reviews")
        con.execute("CREATE INDEX reviews_movie_id ON Reviews(id)")
        con.execute("CREATE INDEX reviews_review_id ON Reviews(reviewId)")
        con.execute("CREATE INDEX movies_id ON Movies(id)")
        counts = {
            "Movies": int(con.execute("SELECT COUNT(*) FROM Movies").fetchone()[0]),
            "Reviews": int(con.execute("SELECT COUNT(*) FROM Reviews").fetchone()[0]),
        }
        schemas = {
            table: [row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
            for table in ("Movies", "Reviews")
        }
    finally:
        con.close()

    leaked = {
        "Reviews": sorted(HIDDEN_REVIEW_COLUMNS & set(schemas["Reviews"])),
        "Movies": sorted(HIDDEN_MOVIE_COLUMNS & set(schemas["Movies"])),
    }
    if leaked["Reviews"] or leaked["Movies"]:
        raise RuntimeError(f"Ground-truth/proxy columns leaked into agent DB: {leaked}")
    return {"row_counts": counts, "agent_columns": schemas, "leaked_columns": leaked}


def _write_ground_truth(root: Path, data_root: Path, target: Path) -> dict[str, int]:
    movies = pd.read_csv(data_root / "Movies.csv")
    reviews = pd.read_csv(data_root / "Reviews.csv")
    sql_root = root / "files" / "movie" / "query" / "gold_sql"
    target.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    row_counts: dict[str, int] = {}
    try:
        con.register("Movies", movies)
        con.register("Reviews", reviews)
        for qid in range(1, 11):
            sql_path = sql_root / f"Q{qid}.sql"
            if not sql_path.is_file():
                raise FileNotFoundError(sql_path)
            frame = con.execute(sql_path.read_text(encoding="utf-8")).fetchdf()
            frame.to_csv(target / f"Q{qid}.csv", index=False)
            row_counts[f"Q{qid}"] = int(len(frame))
    finally:
        con.close()
    return row_counts


def main() -> int:
    args = _args()
    source = args.sembench_root.resolve()
    output = args.output_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    if output.exists():
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    data_root = source / "files" / "movie" / "data" / "sf_2000"
    db_dir = output / "database"
    db_dir.mkdir()
    db_path = db_dir / "sembench_movie.duckdb"
    audit = _write_agent_db(data_root, db_path)

    records = _read_queries(source)
    query_file = output / "evaluation_movie_agent.jsonl"
    query_file.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    gt_counts = _write_ground_truth(source, data_root, output / "ground_truth")
    report = {
        "benchmark": "SemBench",
        "scenario": "movie",
        "scale_factor": 2000,
        "source_revision": _source_revision(source),
        "query_count": len(records),
        "query_ids": [record["question_id"] for record in records],
        "visibility_policy": {
            "hidden_review_columns": sorted(HIDDEN_REVIEW_COLUMNS),
            "hidden_movie_columns": sorted(HIDDEN_MOVIE_COLUMNS),
            "reason": "Prevent ground-truth and direct-proxy leakage under free natural-language planning.",
        },
        "ground_truth_row_counts": gt_counts,
        **audit,
    }
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
