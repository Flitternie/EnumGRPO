"""
preprocess.py -- Adds required_columns to multi-column query entries, then
splits the dataset into train/test splits with a fixed seed.

Usage:
    python swan/preprocess.py [--seed SEED] [--input INPUT] [--output-dir OUTPUT_DIR]

Outputs (written to --output-dir, default: swan/):
    query_with_required_columns.jsonl  -- full dataset with required_columns added
    evaluation.jsonl                  -- 80 randomly sampled entries
    learning.jsonl                 -- remaining 40 entries

Constraints:
    - evaluation.jsonl will not contain any question_id already present in
      the existing learning.jsonl (swan/learning.jsonl), so those
      entries are pinned to train before random sampling occurs.
"""

import argparse
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Column extraction
# ---------------------------------------------------------------------------

# Manual overrides for queries whose SELECT clause cannot be parsed cleanly.
COLUMN_OVERRIDES: dict[str, list[str]] = {
    "formula_1_28": ["age", "forename", "surname"],
    "formula_1_30": ["circuit_name", "location", "race_name"],
    "superhero_22": ["bad_alignment_percentage", "marvel_bad_count"],
}


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split *text* on *delimiter* while ignoring delimiters inside
    parentheses or quoted strings."""
    parts: list[str] = []
    depth = 0
    in_quote = False
    quote_char = ""
    current: list[str] = []

    for ch in text:
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
        elif ch in ('"', "'", "`"):
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        parts.append("".join(current).strip())
    return parts


def _name_for_token(token: str) -> str:
    """Derive a human-readable column name from a single SELECT token."""
    token = token.strip()

    # Explicit AS alias -- find the last top-level AS
    # Split on AS at depth 0
    as_parts = re.split(r"\bAS\b", token, flags=re.IGNORECASE)
    if len(as_parts) >= 2:
        alias = as_parts[-1].strip().strip("`\"'")
        if re.match(r"^\w+$", alias):
            return alias

    # Table.column like T2.City or "T2"."City"
    dot_match = re.match(r'^(?:\w+\.)([`"]?)([\w ]+)\1$', token)
    if dot_match:
        return dot_match.group(2).strip()

    # Simple quoted identifier
    quoted = re.match(r'^[`"]([\w ]+)[`"]$', token)
    if quoted:
        return quoted.group(1).strip()

    # Simple bare identifier (optional table prefix)
    simple = re.match(r'^(?:\w+\.)?(\w+)$', token)
    if simple:
        return simple.group(1)

    # Expression: grab the last table.col reference
    refs = re.findall(r'\b\w+\.(\w+)\b', token)
    if refs:
        return refs[-1]

    # Fallback: strip non-identifier characters
    clean = re.sub(r"[^a-zA-Z0-9_ ]", "", token).strip()
    return clean if clean else token[:40]


def extract_required_columns(sql: str) -> list[str] | None:
    """Return column name list parsed from the SELECT clause, or None."""
    m = re.match(
        r"\s*SELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\s+",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    tokens = _split_top_level(m.group(1))
    return [_name_for_token(t) for t in tokens]


def add_required_columns(records: list[dict]) -> list[dict]:
    """Return a new list of records with *required_columns* added where
    the answer contains more than one column per row."""
    out: list[dict] = []
    for rec in records:
        rec = dict(rec)
        answer = rec.get("answer", [])
        is_multi = answer and any(
            isinstance(row, list) and len(row) > 1 for row in answer
        )
        if is_multi:
            qid = rec["question_id"]
            cols = COLUMN_OVERRIDES.get(qid) or extract_required_columns(rec["sql"])
            rec["required_columns"] = cols
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def split_dataset(
    records: list[dict],
    existing_train_ids: set[str],
    n_test: int = 80,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split *records* into (test, train) lists.

    Rules:
    - Any record whose question_id is in *existing_train_ids* is pinned to
      train and never placed in test.
    - From the remaining pool, *n_test* entries are sampled (seeded) for test.
    - Everything else goes to train.
    """
    pinned_train = [r for r in records if r["question_id"] in existing_train_ids]
    pool = [r for r in records if r["question_id"] not in existing_train_ids]

    rng = random.Random(seed)
    rng.shuffle(pool)

    n_test = min(n_test, len(pool))
    test = pool[:n_test]
    train = pool[n_test:] + pinned_train

    return test, train


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records):>3} entries -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="swan/swan.jsonl",
        help="Source query JSONL file (default: swan/swan.jsonl)",
    )
    parser.add_argument(
        "--existing-train",
        default="swan/learning.jsonl",
        help="Existing train JSONL whose IDs must stay in train (default: swan/learning.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="swan",
        help="Directory for output files (default: swan/)",
    )
    parser.add_argument(
        "--n-test", type=int, default=80, help="Number of test samples (default: 80)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    input_path = Path(args.input)
    existing_train_path = Path(args.existing_train)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load source data
    records = read_jsonl(input_path)
    print(f"Loaded {len(records)} records from {input_path}")

    # 2. Add required_columns
    records = add_required_columns(records)
    enriched_path = output_dir / "query_with_required_columns.jsonl"
    write_jsonl(records, enriched_path)

    # 3. Load existing train IDs to pin
    existing_train_ids: set[str] = set()
    if existing_train_path.exists():
        existing = read_jsonl(existing_train_path)
        existing_train_ids = {r["question_id"] for r in existing}
        print(
            f"Pinning {len(existing_train_ids)} IDs from {existing_train_path} to train"
        )
    else:
        print(f"No existing train file found at {existing_train_path}, skipping pin")

    # 4. Split
    test, train = split_dataset(
        records,
        existing_train_ids=existing_train_ids,
        n_test=args.n_test,
        seed=args.seed,
    )

    # 5. Write splits
    write_jsonl(test, output_dir / "evaluation.jsonl")
    write_jsonl(train, output_dir / "learning.jsonl")

    # 6. Sanity checks
    test_ids = {r["question_id"] for r in test}
    overlap = test_ids & existing_train_ids
    assert not overlap, f"Test/train overlap: {overlap}"
    assert len(test) + len(train) == len(records), "Split sizes don't add up"
    print(
        f"\nDone. test={len(test)}, train={len(train)}, "
        f"total={len(test)+len(train)}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
