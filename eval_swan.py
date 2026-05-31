#!/usr/bin/env python3
"""Evaluate a SWAN agent run: compute Success Rate, Row-level F1, Item-level F1,
and aggregate token usage (Agent Planning + LLM Op).

Usage:
    python eval_swan.py --run_dir swan/logs/blendsql_agent_run_20260314_141756_536607

Expects:
  - <run_dir>/<question_id>.csv                             — predicted CSV per query
  - <run_dir>/agent_runs/*/logs/conversations/**/base_state.json  — agent planning tokens
  - <run_dir>/agent_runs/*/logs/mcp_server/session_*.jsonl  — MCP tool call logs
  - swan/swan.jsonl                                        — ground truth (configurable via --query_file)

Compatible with both:
  - DB agent runs: agent planning from base_state.json, LLM ops from llm_* MCP tools
  - Agentic BlendSQL runs: agent planning from base_state.json, LLM ops from
    run_blendsql result.token_usage in MCP session logs
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Pricing (per token)
# ---------------------------------------------------------------------------
# Agent planning — Claude Sonnet 4.6 (Anthropic list pricing, USD per token)
# https://docs.anthropic.com/en/docs/about-claude/pricing
_AGENT_INPUT_PRICE_PER_TOK       = 3.00  / 1_000_000   # non-cached input
_AGENT_CACHE_WRITE_PRICE_PER_TOK = 3.75  / 1_000_000   # 5-minute cache write (1.25x base)
_AGENT_CACHE_READ_PRICE_PER_TOK  = 0.30  / 1_000_000   # cache-hit input (0.1x base)
_AGENT_OUTPUT_PRICE_PER_TOK      = 15.00 / 1_000_000   # output

# LLM-op tools (llm_map / llm_reduce) — Claude Haiku 4.5 (Anthropic list pricing)
_LLMOP_INPUT_PRICE_PER_TOK       = 1.00 / 1_000_000
_LLMOP_CACHE_WRITE_PRICE_PER_TOK = 1.25 / 1_000_000   # 5-minute cache write (1.25x base)
_LLMOP_CACHE_READ_PRICE_PER_TOK  = 0.10 / 1_000_000   # cache-hit input (0.1x base)
_LLMOP_OUTPUT_PRICE_PER_TOK      = 5.00 / 1_000_000

from swan.evaluation.utils import (
    evaluate_ground_truth,
    load_ground_truth_jsonl,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RUN_DIR_RE = re.compile(r"^(?P<qid>.+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<suffix>\d{6,})$")


def _infer_qid_from_run_dirname(name: str) -> str:
    m = _RUN_DIR_RE.match(name)
    return m.group("qid") if m else name


def _infer_ts_from_run_dirname(name: str) -> int:
    m = _RUN_DIR_RE.match(name)
    if not m:
        return 0
    return int(f"{m.group('date')}{m.group('time')}")


def _to_int(x: Any) -> int:
    try:
        if x is None:
            return 0
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            return int(float(s)) if s else 0
        return int(x)
    except Exception:
        return 0


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, str):
            s = x.strip()
            return float(s) if s else 0.0
        return float(x)
    except Exception:
        return 0.0


def _load_csv_rows(csv_path: Path) -> Tuple[List[List[str]], List[str]]:
    """Load a CSV file and return (rows, header).

    Rows exclude the header row. Header is an empty list when the file is
    missing or has no content.
    """
    rows: List[List[str]] = []
    if not csv_path.is_file():
        return rows, []
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows, []
        for r in reader:
            rows.append(list(r))
    return rows, list(header)


def _project_to_required_columns(
    rows: List[List[str]],
    header: List[str],
    required_columns: List[str],
) -> List[List[str]]:
    """Project *rows* to *required_columns* in the order they are listed.

    Column matching is case-insensitive. Columns that appear in
    *required_columns* but are absent from *header* are silently skipped
    (the current column-permutation fallback in the evaluator then handles
    the mismatch).  Returns the original rows unchanged when *required_columns*
    is empty or no header columns can be matched.
    """
    if not required_columns or not header:
        return rows
    header_lower = [h.lower() for h in header]
    indices: List[int] = []
    for col in required_columns:
        try:
            indices.append(header_lower.index(col.lower()))
        except ValueError:
            pass  # column not found — let the evaluator handle it
    if not indices:
        return rows
    return [[row[i] if i < len(row) else "" for i in indices] for row in rows]


def _find_pred_csv(run_dir: Path, question_id: str) -> Optional[Path]:
    """Find the predicted CSV for a question_id in the run directory."""
    # Direct match: <question_id>.csv
    direct = run_dir / f"{question_id}.csv"
    if direct.is_file():
        return direct
    # Sanitized match (same as _csv_filename_from_question_id)
    cleaned = []
    for ch in question_id:
        if ch in {"/", "\\", "\x00"} or ord(ch) < 32:
            cleaned.append("_")
        else:
            cleaned.append(ch)
    name = "".join(cleaned).strip().strip(".")
    if not name.lower().endswith(".csv"):
        name += ".csv"
    cand = run_dir / name
    if cand.is_file():
        return cand
    return None


def _find_agent_run_dir(run_dir: Path, question_id: str) -> Optional[Path]:
    """Find the latest agent_runs/<question_id>_* subdirectory for a question."""
    agent_runs = run_dir / "agent_runs"
    if not agent_runs.is_dir():
        return None

    best: Optional[Path] = None
    best_ts = -1

    for d in agent_runs.iterdir():
        if not d.is_dir():
            continue
        inferred_qid = _infer_qid_from_run_dirname(d.name)
        if inferred_qid != question_id:
            continue
        ts = _infer_ts_from_run_dirname(d.name)
        if ts > best_ts:
            best_ts = ts
            best = d

    if best is not None:
        return best

    # Fallback: sanitized prefix match (handles special chars in question_id)
    cleaned = []
    for ch in question_id:
        if ch in {"/", "\\", "\x00"} or ord(ch) < 32:
            cleaned.append("_")
        else:
            cleaned.append(ch)
    prefix = "".join(cleaned).strip().strip(".")

    best = None
    best_ts = -1
    for d in agent_runs.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            ts = _infer_ts_from_run_dirname(d.name)
            if ts > best_ts:
                best_ts = ts
                best = d
    return best


# ---------------------------------------------------------------------------
# Agent planning token usage — from base_state.json (OpenHands)
# ---------------------------------------------------------------------------

def _find_base_state_json(run_dir: Path) -> Optional[Path]:
    """Locate base_state.json inside an agent run dir."""
    conv_root = run_dir / "logs" / "conversations"
    if not conv_root.exists():
        return None
    preferred = list(conv_root.glob("main_agent/**/base_state.json"))
    if preferred:
        return max(preferred, key=lambda p: p.stat().st_mtime)
    any_bs = list(conv_root.glob("**/base_state.json"))
    if any_bs:
        return max(any_bs, key=lambda p: p.stat().st_mtime)
    return None


def _extract_agent_usage_from_base_state(base_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract agent planning cost/token usage from OpenHands base_state.json.

    Reads stats.usage_to_metrics (keyed by usage_id like 'main', 'condenser') and
    sums accumulated_cost + accumulated_token_usage across all entries, including
    cache_read_tokens where available.
    """
    stats = base_state.get("stats") if isinstance(base_state, dict) else None
    usage_to_metrics = (stats.get("usage_to_metrics") if isinstance(stats, dict) else None) or {}

    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    total_cache_read = 0
    total_cache_write = 0
    by_usage_id: Dict[str, Dict[str, Any]] = {}

    if isinstance(usage_to_metrics, dict):
        for usage_id, m in usage_to_metrics.items():
            if not isinstance(usage_id, str) or not isinstance(m, dict):
                continue
            tu = m.get("accumulated_token_usage") if isinstance(m.get("accumulated_token_usage"), dict) else {}
            pt = _to_int(tu.get("prompt_tokens"))
            ct = _to_int(tu.get("completion_tokens"))
            cr = _to_int(tu.get("cache_read_tokens"))
            cw = _to_int(tu.get("cache_write_tokens"))
            # prompt_tokens = total billable input (includes cache reads + cache writes);
            # non-cached input = prompt - cache_read - cache_write
            non_cached_input = max(0, pt - cr - cw)
            cost = (non_cached_input * _AGENT_INPUT_PRICE_PER_TOK
                    + cr * _AGENT_CACHE_READ_PRICE_PER_TOK
                    + cw * _AGENT_CACHE_WRITE_PRICE_PER_TOK
                    + ct * _AGENT_OUTPUT_PRICE_PER_TOK)
            by_usage_id[usage_id] = {
                "cost": float(cost),
                "prompt_tokens": int(pt),
                "completion_tokens": int(ct),
                "cache_read_tokens": int(cr),
                "cache_write_tokens": int(cw),
                "total_tokens": int(pt + ct),
            }
            total_cost += float(cost)
            total_prompt += int(pt)
            total_completion += int(ct)
            total_cache_read += int(cr)
            total_cache_write += int(cw)

    by_usage_summary = ""
    if by_usage_id:
        parts = [f"{uid}: ${by_usage_id[uid]['cost']:.4f} / {by_usage_id[uid]['total_tokens']}t"
                 for uid in sorted(by_usage_id)]
        by_usage_summary = "; ".join(parts)

    return {
        "agent_cost_usd": float(total_cost),
        # prompt_tokens = total input tokens (includes cache hits, per OpenHands definition)
        "agent_prompt_tokens": int(total_prompt),
        "agent_completion_tokens": int(total_completion),
        # cache_read = tokens served from cache (0.1x input price)
        "agent_cache_read_tokens": int(total_cache_read),
        # cache_write = tokens written to cache (1.25x input price for 5m cache)
        "agent_cache_write_tokens": int(total_cache_write),
        "agent_total_tokens": int(total_prompt + total_completion),
        "agent_by_usage_id": by_usage_id,
        "agent_by_usage_id_summary": by_usage_summary,
    }


def _extract_agent_token_usage(run_dir: Path, question_id: str) -> Dict[str, Any]:
    """Extract agent planning token usage for a question_id.

    Source: base_state.json from the agent run's conversations log.
    Falls back to an empty dict if not found.
    """
    target_dir = _find_agent_run_dir(run_dir, question_id)
    if target_dir is None:
        return {}
    bs_path = _find_base_state_json(target_dir)
    if bs_path is None or not bs_path.exists():
        return {}
    try:
        base_state = json.loads(bs_path.read_text(encoding="utf-8"))
        if isinstance(base_state, dict):
            result = _extract_agent_usage_from_base_state(base_state)
            result["agent_has_metrics"] = True
            return result
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Stdout log parsing — cost + token summary line from .stdout.txt files
# ---------------------------------------------------------------------------

_TOKENS_LINE_RE = re.compile(
    r"Tokens:\s*\u2191\s*input\s+([\d.]+[KMkm]?)"
    r"\s*\u2022\s*cache hit\s+([\d.]+%|N/A)"
    r"\s*\u2022\s*\u2193\s*output\s+([\d.]+[KMkm]?)"
    r"\s*\u2022\s*\$\s*([\d.]+)"
)


def _parse_k_tokens(s: str) -> int:
    """Parse token count like '183.18K', '1.2M', or '1234' to integer."""
    s = s.strip()
    if s and s[-1] in ("K", "k"):
        return int(float(s[:-1]) * 1_000)
    if s and s[-1] in ("M", "m"):
        return int(float(s[:-1]) * 1_000_000)
    return _to_int(s)


def _parse_last_tokens_line(text: str) -> Optional[Dict[str, Any]]:
    """Extract the final accumulated token/cost line from a stdout log.

    Looks for lines matching:
        Tokens: ↑ input 183.18K • cache hit 90.22% • ↓ output 7.44K • $ 0.2511
    and returns the values from the last such occurrence (cumulative totals).
    Returns None if no matching line is found.
    """
    matches = _TOKENS_LINE_RE.findall(text)
    if not matches:
        return None
    inp_s, cache_s, out_s, cost_s = matches[-1]
    inp = _parse_k_tokens(inp_s)
    out = _parse_k_tokens(out_s)
    cost = _to_float(cost_s)
    cache_pct: Optional[float] = None
    if cache_s != "N/A":
        try:
            cache_pct = float(cache_s.rstrip("%")) / 100.0
        except Exception:
            pass
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
        "cost_usd": cost,
        "cache_hit_pct": cache_pct,
    }


def _find_stdout_log(run_dir: Path, question_id: str) -> Optional[Path]:
    """Return the .stdout.txt path for question_id using manifest.jsonl, if it exists."""
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("question_id") == question_id:
                    idx = entry.get("idx")
                    db = entry.get("db", "")
                    if idx is not None:
                        fname = f"{int(idx):06d}_{question_id}_{db}.stdout.txt"
                        p = run_dir / fname
                        if p.is_file():
                            return p
    except Exception:
        pass
    return None


def _extract_stdout_log_usage(run_dir: Path, question_id: str) -> Optional[Dict[str, Any]]:
    """Parse token/cost summary from the stdout log for question_id, if available."""
    log_path = _find_stdout_log(run_dir, question_id)
    if log_path is None:
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return _parse_last_tokens_line(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MCP LLM-operator token usage — compatible with both agent types
# ---------------------------------------------------------------------------

def _extract_llm_token_usage_from_record(rec: Dict[str, Any], tool: str) -> Tuple[int, int, int, int, int]:
    """Best-effort extraction of (input, output, total, cache_read, cache_write) tokens from one MCP log record.

    Tries multiple locations in order of preference:
      1. Top-level rec["llm_token_usage"]           — used by llm_* tools (db agent)
      2. rec["result"]["llm_token_usage"]            — fallback nested location
      3. rec["result"][tool]["token_usage"]          — namespaced blob (e.g. {"llm_map": {...}})
      4. rec["result"]["token_usage"]                — run_blendsql (agentic blendsql)
      5. Top-level rec["partial_token_usage"]        — errored run_blendsql calls that consumed
                                                       tokens before the error (e.g. mid-LLMMap
                                                       502/timeout). Only used when the above
                                                       locations all yield zero tokens.
    """
    def _read_tu(tu: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        inp = _to_int(tu.get("input_tokens"))
        out = _to_int(tu.get("output_tokens"))
        tot = _to_int(tu.get("total_tokens"))
        cr  = _to_int(tu.get("cache_read_tokens"))
        cw  = _to_int(tu.get("cache_write_tokens"))
        return inp, out, tot if tot else inp + out, cr, cw

    # 1. Top-level llm_token_usage
    tu = rec.get("llm_token_usage")
    if isinstance(tu, dict):
        return _read_tu(tu)

    res = rec.get("result")
    if isinstance(res, dict):
        # 2. result["llm_token_usage"]
        tu2 = res.get("llm_token_usage")
        if isinstance(tu2, dict):
            return _read_tu(tu2)

        # 3. result[tool]["token_usage"] — namespaced blob
        if tool and isinstance(res.get(tool), dict):
            tblob = res[tool]
            if isinstance(tblob.get("token_usage"), dict):
                return _read_tu(tblob["token_usage"])

        # 4. result["token_usage"] — run_blendsql
        if isinstance(res.get("token_usage"), dict):
            return _read_tu(res["token_usage"])

    # 5. partial_token_usage — emitted on error records when BlendSQL consumed tokens
    #    before the failure (e.g. mid-LLMMap 502 / timeout). Fall back to this only
    #    when all structured locations above produced zero.
    ptu = rec.get("partial_token_usage")
    if isinstance(ptu, dict):
        inp, out, tot, cr, cw = _read_tu(ptu)
        if tot or inp or out:
            return inp, out, tot, cr, cw

    return 0, 0, 0, 0, 0


def _extract_mcp_tool_stats(run_dir: Path, question_id: str, llm_tool_prefix: str = "llm_") -> Dict[str, Any]:
    """Extract MCP tool call stats from server logs for a question_id.

    Handles both agent types:
    - DB agent: counts tools with names starting with ``llm_tool_prefix`` (e.g. "llm_map",
      "llm_qa") and sums their token usage.
    - Agentic BlendSQL: counts ``run_blendsql`` calls and sums token_usage from each
      result payload (ingredients produce LLM calls inside the MCP server).

    Returns:
        {
            "used_blendsql": bool,
            "blendsql_calls": int,
            "blendsql_errors": int,
            "llm_op_calls": int,           # total llm_* tool calls (db agent)
            "llm_op_token_usage": {        # aggregated LLM-op tokens (both agent types)
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
                "cache_read_tokens": int,
                "cache_write_tokens": int,
            },
            "llm_by_tool": dict,           # per-tool breakdown
            "llm_by_tool_summary": str,    # human-readable summary
        }
    """
    _empty: Dict[str, Any] = {
        "used_blendsql": False,
        "blendsql_calls": 0,
        "blendsql_errors": 0,
        "llm_op_calls": 0,
        "llm_op_token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        },
        "llm_by_tool": {},
        "llm_by_tool_summary": "",
        "num_cache_hits": 0,
        "workflow_length": 0,
    }

    target_dir = _find_agent_run_dir(run_dir, question_id)
    if target_dir is None:
        return _empty

    mcp_dir = target_dir / "logs" / "mcp_server"
    if not mcp_dir.is_dir():
        return _empty

    blendsql_calls = 0
    blendsql_errors = 0
    llm_op_calls = 0
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    total_cache_hits = 0
    workflow_length = 0   # total MCP tool call entries across all session files
    llm_by_tool: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
    )

    for f in mcp_dir.glob("session_*.jsonl"):
        try:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue

                    tool = entry.get("tool", "")
                    if not isinstance(tool, str):
                        continue

                    workflow_length += 1

                    if tool == "run_blendsql":
                        blendsql_calls += 1
                        if entry.get("error"):
                            blendsql_errors += 1
                        inp, out, tot, cr, cw = _extract_llm_token_usage_from_record(entry, tool)
                        if tot > 0:
                            total_input += inp
                            total_output += out
                            total_cache_read += cr
                            total_cache_write += cw
                            agg = llm_by_tool["run_blendsql"]
                            agg["calls"] += 1
                            agg["input_tokens"] += inp
                            agg["output_tokens"] += out
                            agg["total_tokens"] += tot
                            agg["cache_read_tokens"] += cr
                            agg["cache_write_tokens"] += cw
                        # Track cache hits (available after the fix to run_blendsql)
                        res = entry.get("result")
                        if isinstance(res, dict):
                            total_cache_hits += _to_int(res.get("num_cache_hits"))

                    elif tool.startswith(llm_tool_prefix):
                        llm_op_calls += 1
                        inp, out, tot, cr, cw = _extract_llm_token_usage_from_record(entry, tool)
                        total_input += inp
                        total_output += out
                        total_cache_read += cr
                        total_cache_write += cw
                        agg = llm_by_tool[tool]
                        agg["calls"] += 1
                        agg["input_tokens"] += inp
                        agg["output_tokens"] += out
                        agg["total_tokens"] += tot
                        agg["cache_read_tokens"] += cr
                        agg["cache_write_tokens"] += cw

        except Exception:
            continue

    by_tool_summary = ""
    if llm_by_tool:
        parts = [f"{t}:{llm_by_tool[t]['total_tokens']}t/{llm_by_tool[t]['calls']}c"
                 for t in sorted(llm_by_tool)]
        by_tool_summary = "; ".join(parts)

    total_tokens = total_input + total_output
    return {
        "used_blendsql": blendsql_calls > 0,
        "blendsql_calls": blendsql_calls,
        "blendsql_errors": blendsql_errors,
        "llm_op_calls": llm_op_calls,
        "workflow_length": workflow_length,
        "llm_op_token_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
        },
        "llm_by_tool": dict(llm_by_tool),
        "llm_by_tool_summary": by_tool_summary,
        "num_cache_hits": total_cache_hits,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a SWAN agent run.")
    p.add_argument("--run_dir", required=True, help="Path to the agent run output directory.")
    p.add_argument("--query_file", default=str((_REPO_ROOT / "swan" / "swan.jsonl").resolve()),
                   help="Path to SWAN ground truth JSONL.")
    p.add_argument("--num_decimals", type=int, default=2, help="Decimal precision for numeric comparison.")
    p.add_argument("--unordered", action="store_true",
                   help="Ignore ORDER BY / ordering heuristics and compare all results as unordered multisets.")
    p.add_argument("--json", action="store_true", help="Output results as JSON.")
    p.add_argument("--per_query", action="store_true", help="Also output per-query results.")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (_REPO_ROOT / run_dir).resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    query_file = Path(args.query_file).resolve()
    gt_map = load_ground_truth_jsonl(query_file)
    print(f"Loaded {len(gt_map)} ground truth entries from {query_file}", file=sys.stderr)

    # Load per-query wall-clock times from results.jsonl (produced by run_swan_main.py).
    # Gracefully absent for older runs or text2sql (empty file).
    elapsed_map: Dict[str, float] = {}
    results_jsonl = run_dir / "results.jsonl"
    if results_jsonl.exists():
        with results_jsonl.open(encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _rec = json.loads(_line)
                    _qid = _rec.get("job", {}).get("question_id", "")
                    _el  = _rec.get("elapsed_s")
                    if _qid and _el is not None:
                        elapsed_map[_qid] = float(_el)
                except (json.JSONDecodeError, KeyError):
                    pass

    per_query_results: List[Dict[str, Any]] = []
    n_matched = 0
    n_missing_csv = 0
    elapsed_total = 0.0
    n_elapsed = 0

    # Agent planning token accumulators (from base_state.json)
    agent_cost_total = 0.0
    agent_prompt_total = 0        # total input (cached + non-cached)
    agent_completion_total = 0
    agent_cache_read_total = 0    # tokens served from cache
    agent_cache_write_total = 0   # tokens written to cache (5m ephemeral)
    n_agent_token = 0  # queries with valid agent metrics

    # LLM-op token accumulators (from MCP session logs)
    llm_op_input_total = 0
    llm_op_output_total = 0
    llm_op_cache_read_total = 0
    llm_op_cache_write_total = 0
    llm_op_cost_total = 0.0
    n_llm_op_token = 0  # queries that had any LLM-op tokens

    # BlendSQL call tracking
    n_used_blendsql = 0
    total_blendsql_calls = 0
    total_blendsql_errors = 0
    total_cache_hits = 0
    workflow_length_total = 0     # sum of per-query MCP session entry counts

    # Stdout log cost accumulator (fallback when base_state.json absent)
    stdout_cost_total = 0.0
    n_stdout_cost = 0

    for qid, gt_rec in sorted(gt_map.items()):
        csv_path = _find_pred_csv(run_dir, qid)
        missing_csv = csv_path is None
        if missing_csv:
            n_missing_csv += 1
            metrics: Dict[str, Any] = {"sr": 0, "row_f1": 0.0, "item_f1": 0.0,
                                        "row_precision": 0.0, "row_recall": 0.0,
                                        "item_precision": 0.0, "item_recall": 0.0,
                                        "gt_rows": 0, "pred_rows": 0,
                                        "requires_order": False}
        else:
            n_matched += 1
            pred_rows, pred_header = _load_csv_rows(csv_path)
            # When the query specifies required output columns, project the
            # prediction to exactly those columns before scoring so that extra
            # context columns the agent included do not penalise it.  The
            # existing column-permutation search in the evaluator is kept as a
            # fallback for queries without required_columns.
            if gt_rec.required_columns:
                pred_rows = _project_to_required_columns(pred_rows, pred_header, gt_rec.required_columns)
            metrics = evaluate_ground_truth(gt_rec, pred_rows, num_decimals=int(args.num_decimals),
                                             force_unordered=bool(args.unordered))

        # Agent planning tokens (base_state.json) — extracted for all queries,
        # including those without a CSV output, so failed runs' costs are counted.
        agent_tu = _extract_agent_token_usage(run_dir, qid)
        has_agent_metrics = bool(agent_tu.get("agent_has_metrics"))
        agent_cost = _to_float(agent_tu.get("agent_cost_usd"))
        agent_prompt = _to_int(agent_tu.get("agent_prompt_tokens"))
        agent_completion = _to_int(agent_tu.get("agent_completion_tokens"))
        agent_cache_read = _to_int(agent_tu.get("agent_cache_read_tokens"))
        agent_cache_write = _to_int(agent_tu.get("agent_cache_write_tokens"))
        agent_total = _to_int(agent_tu.get("agent_total_tokens")) or (agent_prompt + agent_completion)
        agent_cache_hit_pct: Optional[float] = (
            agent_cache_read / agent_prompt if agent_prompt > 0 else None
        )
        if has_agent_metrics:
            agent_cost_total += agent_cost
            agent_prompt_total += agent_prompt
            agent_completion_total += agent_completion
            agent_cache_read_total += agent_cache_read
            agent_cache_write_total += agent_cache_write
            n_agent_token += 1

        # Stdout log — cost cross-check / fallback when base_state.json absent
        stdout_usage = _extract_stdout_log_usage(run_dir, qid)
        stdout_cost: Optional[float] = stdout_usage["cost_usd"] if stdout_usage else None
        if stdout_cost is not None and not has_agent_metrics:
            stdout_cost_total += stdout_cost
            n_stdout_cost += 1

        # MCP LLM-op tokens — extracted for all queries including missing-CSV ones.
        mcp_stats = _extract_mcp_tool_stats(run_dir, qid)
        used_blendsql = mcp_stats["used_blendsql"]
        if used_blendsql:
            n_used_blendsql += 1
            total_blendsql_calls += mcp_stats["blendsql_calls"]
            total_blendsql_errors += mcp_stats["blendsql_errors"]
        total_cache_hits += mcp_stats["num_cache_hits"]
        workflow_length_total += mcp_stats["workflow_length"]
        mcp_in = mcp_stats["llm_op_token_usage"]["input_tokens"]
        mcp_out = mcp_stats["llm_op_token_usage"]["output_tokens"]
        mcp_tot = mcp_stats["llm_op_token_usage"]["total_tokens"]
        mcp_cr  = mcp_stats["llm_op_token_usage"]["cache_read_tokens"]
        mcp_cw  = mcp_stats["llm_op_token_usage"]["cache_write_tokens"]
        mcp_non_cached_input = max(0, mcp_in - mcp_cr - mcp_cw)
        mcp_cost = (
            mcp_non_cached_input * _LLMOP_INPUT_PRICE_PER_TOK
            + mcp_cw * _LLMOP_CACHE_WRITE_PRICE_PER_TOK
            + mcp_cr * _LLMOP_CACHE_READ_PRICE_PER_TOK
            + mcp_out * _LLMOP_OUTPUT_PRICE_PER_TOK
        )
        if mcp_tot > 0:
            llm_op_input_total += mcp_in
            llm_op_output_total += mcp_out
            llm_op_cache_read_total += mcp_cr
            llm_op_cache_write_total += mcp_cw
            llm_op_cost_total += mcp_cost
            n_llm_op_token += 1

        elapsed_s: Optional[float] = elapsed_map.get(qid)
        if elapsed_s is not None:
            elapsed_total += elapsed_s
            n_elapsed += 1

        per_query_results.append({
            "question_id": qid,
            "db": gt_rec.db,
            "status": "missing_csv" if missing_csv else "evaluated",
            "elapsed_s": elapsed_s,
            "sr": metrics.get("sr", 0),
            "row_f1": metrics.get("row_f1", 0.0),
            "item_f1": metrics.get("item_f1", 0.0),
            "row_precision": metrics.get("row_precision", 0.0),
            "row_recall": metrics.get("row_recall", 0.0),
            "item_precision": metrics.get("item_precision", 0.0),
            "item_recall": metrics.get("item_recall", 0.0),
            "gt_rows": metrics.get("gt_rows", 0),
            "pred_rows": metrics.get("pred_rows", 0),
            "requires_order": metrics.get("requires_order", False),
            # BlendSQL MCP usage
            "used_blendsql": used_blendsql,
            "blendsql_calls": mcp_stats["blendsql_calls"],
            "blendsql_errors": mcp_stats["blendsql_errors"],
            # Agent planning (base_state.json)
            "agent_has_metrics": has_agent_metrics,
            "agent_cost_usd": agent_cost,
            "agent_prompt_tokens": agent_prompt,
            "agent_completion_tokens": agent_completion,
            "agent_cache_read_tokens": agent_cache_read,
            "agent_cache_write_tokens": agent_cache_write,
            "agent_cache_hit_pct": round(agent_cache_hit_pct * 100, 2) if agent_cache_hit_pct is not None else None,
            "agent_total_tokens": agent_total,
            "agent_by_usage_id_summary": str(agent_tu.get("agent_by_usage_id_summary", "")),
            # Stdout log cost (cross-check / fallback)
            "stdout_cost_usd": stdout_cost,
            # MCP LLM-op tokens
            "mcp_input_tokens": mcp_in,
            "mcp_output_tokens": mcp_out,
            "mcp_total_tokens": mcp_tot,
            "mcp_cache_read_tokens": mcp_cr,
            "mcp_cache_write_tokens": mcp_cw,
            "mcp_cost_usd": mcp_cost,
            "workflow_length": mcp_stats["workflow_length"],
            "llm_by_tool_summary": mcp_stats["llm_by_tool_summary"],
            "num_cache_hits": mcp_stats["num_cache_hits"],
            # Combined totals
            "total_input_tokens": agent_prompt + mcp_in,
            "total_output_tokens": agent_completion + mcp_out,
            "total_tokens": agent_total + mcp_tot,
        })

    # --- Aggregate metrics ---
    n_total = len(per_query_results)
    sr_values    = [r["sr"]       for r in per_query_results]
    row_f1_vals  = [r["row_f1"]   for r in per_query_results]
    item_f1_vals = [r["item_f1"]  for r in per_query_results]

    avg_sr       = sum(sr_values)    / n_total * 100 if n_total else 0.0
    avg_row_f1   = sum(row_f1_vals)  / n_total * 100 if n_total else 0.0
    avg_item_f1  = sum(item_f1_vals) / n_total * 100 if n_total else 0.0

    avg_agent_prompt      = agent_prompt_total      / n_agent_token if n_agent_token else 0.0
    avg_agent_completion  = agent_completion_total  / n_agent_token if n_agent_token else 0.0
    avg_agent_total_tok   = (agent_prompt_total + agent_completion_total) / n_agent_token if n_agent_token else 0.0
    avg_agent_cost        = agent_cost_total / n_agent_token if n_agent_token else 0.0
    avg_agent_cache_read  = agent_cache_read_total  / n_agent_token if n_agent_token else 0.0
    avg_agent_cache_write = agent_cache_write_total / n_agent_token if n_agent_token else 0.0
    overall_cache_hit_pct = agent_cache_read_total / agent_prompt_total if agent_prompt_total > 0 else None

    avg_mcp_in  = llm_op_input_total  / n_llm_op_token if n_llm_op_token else 0.0
    avg_mcp_out = llm_op_output_total / n_llm_op_token if n_llm_op_token else 0.0
    avg_mcp_tot = (llm_op_input_total + llm_op_output_total) / n_llm_op_token if n_llm_op_token else 0.0
    avg_mcp_cr  = llm_op_cache_read_total / n_llm_op_token if n_llm_op_token else 0.0
    avg_mcp_cw  = llm_op_cache_write_total / n_llm_op_token if n_llm_op_token else 0.0
    avg_mcp_cost = llm_op_cost_total / n_llm_op_token if n_llm_op_token else 0.0
    mcp_cache_hit_pct = llm_op_cache_read_total / llm_op_input_total if llm_op_input_total > 0 else None
    avg_workflow_length = workflow_length_total / n_total if n_total else 0.0
    avg_elapsed_s = elapsed_total / n_elapsed if n_elapsed else None

    summary = {
        "run_dir": str(run_dir),
        "n_total": n_total,
        "n_matched": n_matched,
        "n_missing_csv": n_missing_csv,
        "performance": {
            "success_rate": round(avg_sr, 2),
            "row_level_f1": round(avg_row_f1, 2),
            "item_level_f1": round(avg_item_f1, 2),
            "avg_elapsed_s": round(avg_elapsed_s, 2) if avg_elapsed_s is not None else None,
            "n_elapsed": n_elapsed,
        },
        "blendsql_usage": {
            "n_queries_used_blendsql": n_used_blendsql,
            "n_queries_total": n_total,
            "pct_used_blendsql": round(n_used_blendsql / n_total * 100, 1) if n_total else 0.0,
            "total_blendsql_calls": total_blendsql_calls,
            "total_blendsql_errors": total_blendsql_errors,
            "total_cache_hits": total_cache_hits,
        },
        "costs": {
            "agent_planning": {
                "n_queries": n_agent_token,
                "total_cost_usd": round(agent_cost_total, 4),
                "avg_cost_usd": round(avg_agent_cost, 4),
                "avg_total_tokens": round(avg_agent_total_tok),
                "avg_input_tokens": round(avg_agent_prompt),
                "avg_output_tokens": round(avg_agent_completion),
                "avg_cache_read_tokens": round(avg_agent_cache_read),
                "avg_cache_write_tokens": round(avg_agent_cache_write),
                "overall_cache_hit_pct": round(overall_cache_hit_pct * 100, 2) if overall_cache_hit_pct is not None else None,
            },
            "llm_op": {
                "n_queries_with_llm_op": n_llm_op_token,
                "avg_workflow_length": round(avg_workflow_length, 2),
                "avg_total_tokens": round(avg_mcp_tot),
                "avg_input_tokens": round(avg_mcp_in),
                "avg_output_tokens": round(avg_mcp_out),
                "avg_cache_read_tokens": round(avg_mcp_cr),
                "avg_cache_write_tokens": round(avg_mcp_cw),
                "avg_cost_usd": round(avg_mcp_cost, 4),
                "total_cost_usd": round(llm_op_cost_total, 4),
                "overall_cache_hit_pct": round(mcp_cache_hit_pct * 100, 2) if mcp_cache_hit_pct is not None else None,
            },
            "stdout_log_fallback": {
                "n_queries": n_stdout_cost,
                "total_cost_usd": round(stdout_cost_total, 4),
            },
        },
    }

    if args.json:
        out = dict(summary)
        if args.per_query:
            out["per_query"] = per_query_results
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print()
        print("=" * 64)
        print("  SWAN Evaluation Results")
        print("=" * 64)
        print(f"  Run:     {run_dir.name}")
        print(f"  Queries: {n_total} total, {n_matched} matched, {n_missing_csv} missing CSV")
        if n_total:
            print(f"  BlendSQL usage: {n_used_blendsql}/{n_total} queries ({n_used_blendsql / n_total * 100:.1f}%)")
        if total_blendsql_calls:
            print(f"    Total calls: {total_blendsql_calls}, errors: {total_blendsql_errors}, cache hits: {total_cache_hits}")
        print()
        print("  Performance:")
        print(f"    Success Rate:   {avg_sr:6.2f}%")
        print(f"    Row-level F1:   {avg_row_f1:6.2f}%")
        print(f"    Item-level F1:  {avg_item_f1:6.2f}%")
        if avg_elapsed_s is not None:
            print(f"    Avg elapsed:    {avg_elapsed_s:,.1f}s  ({n_elapsed} queries with timing)")
        print()
        print(f"  Agent planning tokens  ({n_agent_token} queries with base_state.json):")
        print(f"    Total cost:            ${agent_cost_total:.4f}")
        print(f"    Avg cost/query:        ${avg_agent_cost:.4f}")
        print(f"    Avg total tokens:      {avg_agent_total_tok:,.0f}")
        print(f"      Avg input tokens:    {avg_agent_prompt:,.0f}")
        cache_hit_str = f"  ({overall_cache_hit_pct*100:.1f}% cache hit)" if overall_cache_hit_pct is not None else ""
        print(f"        cache read:        {avg_agent_cache_read:,.0f}{cache_hit_str}")
        print(f"      Avg output tokens:   {avg_agent_completion:,.0f}")
        if n_stdout_cost > 0:
            print(f"  Stdout log cost ({n_stdout_cost} queries, no base_state.json): ${stdout_cost_total:.4f}")
        print()
        if n_llm_op_token > 0:
            print(f"  LLM-op tokens  ({n_llm_op_token} queries with LLM ingredient/tool calls):")
            print(f"    Avg total tokens:      {avg_mcp_tot:,.0f}")
            print(f"      Avg input tokens:    {avg_mcp_in:,.0f}")
            mcp_cache_hit_str = f"  ({mcp_cache_hit_pct*100:.1f}% cache hit)" if mcp_cache_hit_pct is not None else ""
            print(f"        cache read:        {avg_mcp_cr:,.0f}{mcp_cache_hit_str}")
            print(f"      Avg output tokens:   {avg_mcp_out:,.0f}")
        else:
            print(f"  LLM-op tokens:  none recorded (no llm_* / run_blendsql ingredient calls)")
        print(f"  Avg workflow length:   {avg_workflow_length:.2f} MCP tool calls/query")
        print("=" * 64)

        if args.per_query:
            print()
            print("Per-query results:")
            hdr = f"{'question_id':<30} {'SR':>4} {'Row-F1':>8} {'Item-F1':>8} {'BSQL':>5} {'AgentTok':>10} {'MCPTok':>8} {'Status'}"
            print(hdr)
            print("-" * len(hdr))
            for r in per_query_results:
                bsql = "Y" if r.get("used_blendsql") else ""
                a_tok = r.get("agent_total_tokens", 0)
                m_tok = r.get("mcp_total_tokens", 0)
                print(f"{r['question_id']:<30} {r['sr']:>4} {r['row_f1']:>8.4f} {r['item_f1']:>8.4f}"
                      f" {bsql:>5} {a_tok:>10,} {m_tok:>8,} {r['status']}")

    # Save outputs
    detail_path = run_dir / "eval_results.jsonl"
    with detail_path.open("w", encoding="utf-8") as fh:
        for r in per_query_results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDetailed results saved to: {detail_path}", file=sys.stderr)

    summary_path = run_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
