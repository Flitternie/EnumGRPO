"""Held-out DB (heldout) split builder.

Minimal helper: reads learning.jsonl and evaluation.jsonl, filters by
the 'db' field, and writes per-fold JSONL split files.

Used by heldout experiment scripts before calling python -m learning.cli for each fold.
"""

from __future__ import annotations

import json
from pathlib import Path

ALL_DATABASES: tuple[str, ...] = (
    "california_schools",
    "european_football_2",
    "formula_1",
    "superhero",
)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_heldout_splits(
    train_path: Path,
    test_path: Path,
    held_out_db: str,
    splits_dir: Path,
) -> tuple[Path, Path]:
    """Write filtered train/eval JSONL files for one fold and return their paths.

    train file: all records in train_path whose 'db' != held_out_db
    eval  file: all records in test_path  whose 'db' == held_out_db
    """
    train_all = _read_jsonl(train_path)
    test_all = _read_jsonl(test_path)

    train_records = [r for r in train_all if r.get("db") != held_out_db]
    eval_records = [r for r in test_all if r.get("db") == held_out_db]

    if not train_records:
        raise ValueError(
            f"No training records remain after excluding db='{held_out_db}'. "
            f"Available DBs: {sorted({r.get('db') for r in train_all})}"
        )
    if not eval_records:
        raise ValueError(
            f"No eval records found for db='{held_out_db}' in {test_path}."
        )

    fold_train_path = splits_dir / f"train_{held_out_db}.jsonl"
    fold_eval_path = splits_dir / f"eval_{held_out_db}.jsonl"
    _write_jsonl(train_records, fold_train_path)
    _write_jsonl(eval_records, fold_eval_path)

    return fold_train_path, fold_eval_path
