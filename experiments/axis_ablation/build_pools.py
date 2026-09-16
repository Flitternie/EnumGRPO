#!/usr/bin/env python3
"""Build reproducible leave-one-axis-out experience pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITION_ORDER = ("D", "A", "B", "C", "F")


def _load_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def build_pools(source_pool: Path, axis_map_path: Path, output_dir: Path) -> dict[str, Any]:
    pool_doc = _load_object(source_pool)
    map_doc = _load_object(axis_map_path)

    experiences = pool_doc.get("experiences")
    mapping = map_doc.get("mapping")
    axes = map_doc.get("axes")
    expected_counts = map_doc.get("expected_counts")
    if not isinstance(experiences, dict) or not experiences:
        raise ValueError(f"Missing or empty 'experiences' object in {source_pool}")
    if not isinstance(mapping, dict) or not isinstance(axes, dict):
        raise ValueError(f"Malformed mapping document: {axis_map_path}")

    pool_ids = set(experiences)
    mapped_ids = set(mapping)
    if pool_ids != mapped_ids:
        missing = sorted(pool_ids - mapped_ids)
        extra = sorted(mapped_ids - pool_ids)
        raise ValueError(f"Mapping/pool ID mismatch: missing={missing}, extra={extra}")

    counts: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for experience_id, assignment in mapping.items():
        if not isinstance(assignment, dict):
            raise ValueError(f"Mapping for {experience_id} must be an object")
        axis = assignment.get("axis")
        value = assignment.get("value")
        if axis not in CONDITION_ORDER or axis not in axes:
            raise ValueError(f"Unknown axis for {experience_id}: {axis!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing axis value for {experience_id}")
        counts[axis] += 1
        value_counts[axis][value] += 1

    normalized_expected = {str(k): int(v) for k, v in (expected_counts or {}).items()}
    actual_counts = {axis: counts[axis] for axis in CONDITION_ORDER}
    if actual_counts != normalized_expected:
        raise ValueError(
            f"Axis counts differ from the audited mapping: "
            f"actual={actual_counts}, expected={normalized_expected}"
        )
    if sum(actual_counts.values()) != len(experiences):
        raise ValueError("Axis counts do not cover the complete experience pool")

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions: dict[str, Any] = {}
    for axis in CONDITION_ORDER:
        axis_name = str(axes[axis])
        slug = f"without_{axis_name}"
        removed_ids = [eid for eid in experiences if mapping[eid]["axis"] == axis]
        kept = {eid: text for eid, text in experiences.items() if eid not in removed_ids}
        output = output_dir / f"{slug}.json"
        ablated_doc = {
            **{k: v for k, v in pool_doc.items() if k != "experiences"},
            "num_experiences": len(kept),
            "experiences": kept,
            "ablation": {
                "type": "leave_one_axis_out",
                "removed_axis": axis,
                "removed_axis_name": axis_name,
                "removed_ids": removed_ids,
                "source_pool": str(source_pool.resolve()),
                "axis_map": str(axis_map_path.resolve()),
            },
        }
        output.write_text(
            json.dumps(ablated_doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        conditions[slug] = {
            "axis": axis,
            "axis_name": axis_name,
            "pool_file": str(output.resolve()),
            "removed_ids": removed_ids,
            "removed_count": len(removed_ids),
            "retained_count": len(kept),
        }

    manifest = {
        "experiment": "experience_axis_leave_one_out",
        "source_pool": str(source_pool.resolve()),
        "source_experience_count": len(experiences),
        "axis_map": str(axis_map_path.resolve()),
        "axis_counts": actual_counts,
        "axis_value_counts": {
            axis: dict(sorted(value_counts[axis].items())) for axis in CONDITION_ORDER
        },
        "condition_order": [f"without_{axes[axis]}" for axis in CONDITION_ORDER],
        "conditions": conditions,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--axis-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_pools(
        args.source_pool.resolve(), args.axis_map.resolve(), args.output_dir.resolve()
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
