import os
import re
import json
import asyncio
import copy
from pathlib import Path
import sys
import signal
import time
import threading
import argparse
from typing import Any, Callable, TypeVar, Optional
import contextlib
import io
from dataclasses import dataclass

from blendsql import BlendSQL
from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem, GenerationResult
from blendsql.ingredients import LLMJoin as _LLMJoin  # type: ignore
from blendsql.ingredients import LLMMap as _BaseLLMMap  # type: ignore
from blendsql.ingredients import LLMQA as _BaseLLMQA  # type: ignore

from dotenv import load_dotenv

# Optional progress bars.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# Load repo-local env regardless of current working directory.
load_dotenv((Path(__file__).resolve().parents[2] / ".env"))

# Ensure repo-root imports (e.g. `tools.*`) work when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

T = TypeVar("T")


def _with_heartbeat_tqdm(*, desc: str, fn: Callable[[], T], interval_s: float = 1.0) -> T:
    """
    Show a simple time-elapsed progress indicator while `fn()` runs.

    This is useful when BlendSQL/LLMMap is busy but not producing logs yet.
    """
    if tqdm is None:
        return fn()

    hb_stop = threading.Event()
    hb_pbar = None
    hb_thread: threading.Thread | None = None
    try:
        hb_pbar = tqdm(
            total=None,
            desc=str(desc or "Executing"),
            unit="s",
            dynamic_ncols=True,
            leave=False,
        )

        def _hb() -> None:
            while not hb_stop.is_set():
                time.sleep(float(interval_s))
                try:
                    hb_pbar.update(1)  # type: ignore[union-attr]
                except Exception:
                    pass

        hb_thread = threading.Thread(target=_hb, daemon=True)
        hb_thread.start()
        return fn()
    finally:
        try:
            hb_stop.set()
        except Exception:
            pass
        if hb_thread is not None:
            try:
                hb_thread.join(timeout=2.0)
            except Exception:
                pass
        if hb_pbar is not None:
            try:
                hb_pbar.close()
            except Exception:
                pass


def build_model(*, model_name: str, caching: bool) -> tuple[ModelBase, str]:
    return LLMOpHostedModel(model_name, caching=bool(caching)), "llmop"


@dataclass(frozen=True)
class ExecuteBlendSQLResult:
    smoothie: Any
    df: Any
    exec_stdout: str
    exec_stderr: str
    token_usage: dict[str, int]
    num_generation_calls: int
    num_cache_hits: int = 0


@dataclass(frozen=True)
class TokenEstimate:
    ingredient: str
    question: str
    n_total_items: int
    n_sample_items: int
    sample_token_usage: dict[str, int]
    predicted_token_usage: dict[str, int]


_TOKEN_ESTIMATES: list[TokenEstimate] = []


def _clear_token_estimates() -> None:
    try:
        _TOKEN_ESTIMATES.clear()
    except Exception:
        pass


def get_token_estimates() -> list[TokenEstimate]:
    return list(_TOKEN_ESTIMATES)


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


class LLMQA(_BaseLLMQA):
    """
    Wrapper around BlendSQL's builtin LLMQA to support token-estimation sampling.

    Enabled when BLENDSQL_TOKEN_ESTIMATE_SAMPLE > 0.
    """

    async def run(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        sample_n = _env_int("BLENDSQL_TOKEN_ESTIMATE_SAMPLE", 0)
        if sample_n <= 0:
            return await super().run(*args, **kwargs)

        model = kwargs.get("model") or (args[0] if args else None)
        q = str(kwargs.get("question") or (args[1] if len(args) > 1 else "") or "")
        if model is None:
            return await super().run(*args, **kwargs)

        before_in = int(getattr(model, "prompt_tokens", 0) or 0)
        before_out = int(getattr(model, "completion_tokens", 0) or 0)
        out = await super().run(*args, **kwargs)
        after_in = int(getattr(model, "prompt_tokens", 0) or 0)
        after_out = int(getattr(model, "completion_tokens", 0) or 0)
        d_in = max(0, after_in - before_in)
        d_out = max(0, after_out - before_out)
        sample_usage = {"input_tokens": int(d_in), "output_tokens": int(d_out), "total_tokens": int(d_in + d_out)}
        _TOKEN_ESTIMATES.append(
            TokenEstimate(
                ingredient="LLMQA",
                question=q,
                n_total_items=1,
                n_sample_items=1,
                sample_token_usage=sample_usage,
                predicted_token_usage=sample_usage,
            )
        )
        return out


class LLMMap(_BaseLLMMap):
    """
    Wrapper around BlendSQL's builtin LLMMap to support token-estimation sampling.

    Strategy:
    - Observe full `values` length (n_total_items).
    - Run the underlying LLMMap on only the first N items (value_limit=N).
    - Use model-reported token usage deltas over that sample to extrapolate totals linearly.
    """

    async def run(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        sample_n = _env_int("BLENDSQL_TOKEN_ESTIMATE_SAMPLE", 0)
        if sample_n <= 0:
            return await super().run(*args, **kwargs)

        # LLMMap.run signature: (model, question, values, additional_args, list_options_in_prompt, context_formatter, ...)
        call_kwargs = dict(kwargs)
        arg_names = ["model", "question", "values", "additional_args", "list_options_in_prompt", "context_formatter"]
        for i, name in enumerate(arg_names):
            if name not in call_kwargs and i < len(args):
                call_kwargs[name] = args[i]

        model = call_kwargs.get("model")
        question = call_kwargs.get("question") or ""
        values = call_kwargs.get("values")
        additional_args = call_kwargs.get("additional_args")
        if model is None or not isinstance(values, list):
            return await super().run(*args, **kwargs)

        n_total = int(len(values))
        n_sample = max(0, min(int(sample_n), n_total))
        if n_sample <= 0:
            return await super().run(*args, **kwargs)

        before_in = int(getattr(model, "prompt_tokens", 0) or 0)
        before_out = int(getattr(model, "completion_tokens", 0) or 0)

        sampled_indices: list[int]
        import random

        rng = random.Random(42)
        sampled_indices = rng.sample(range(n_total), k=int(n_sample))
        sampled_indices.sort()

        values_sample = [values[i] for i in sampled_indices]
        call_kwargs["values"] = values_sample

        if isinstance(additional_args, list) and additional_args:
            add_args_sample = []
            for a in additional_args:
                try:
                    b = copy.copy(a)
                    if hasattr(a, "values") and isinstance(getattr(a, "values"), list):
                        b.values = [a.values[i] for i in sampled_indices]  # type: ignore[attr-defined]
                    add_args_sample.append(b)
                except Exception:
                    add_args_sample.append(a)
            call_kwargs["additional_args"] = add_args_sample
        call_kwargs.pop("value_limit", None)

        # Call the base ingredient using only kwargs (avoid duplicating positional args).
        out_sample = await super().run(**call_kwargs)

        after_in = int(getattr(model, "prompt_tokens", 0) or 0)
        after_out = int(getattr(model, "completion_tokens", 0) or 0)
        d_in = max(0, after_in - before_in)
        d_out = max(0, after_out - before_out)
        sample_usage = {"input_tokens": int(d_in), "output_tokens": int(d_out), "total_tokens": int(d_in + d_out)}

        per_in = float(d_in) / float(n_sample) if n_sample > 0 else 0.0
        per_out = float(d_out) / float(n_sample) if n_sample > 0 else 0.0
        pred_in = int(round(per_in * float(n_total)))
        pred_out = int(round(per_out * float(n_total)))
        pred_usage = {
            "input_tokens": int(pred_in),
            "output_tokens": int(pred_out),
            "total_tokens": int(pred_in + pred_out),
        }

        _TOKEN_ESTIMATES.append(
            TokenEstimate(
                ingredient="LLMMap",
                question=str(question or ""),
                n_total_items=int(n_total),
                n_sample_items=int(n_sample),
                sample_token_usage=sample_usage,
                predicted_token_usage=pred_usage,
            )
        )

        # Return full-length vector to let query continue cheaply (fill sampled indices only).
        full = [None] * n_total
        try:
            if isinstance(out_sample, list):
                for j, i in enumerate(sampled_indices):
                    if j < len(out_sample):
                        full[int(i)] = out_sample[j]
        except Exception:
            return [None] * n_total
        return full


class _TeeTextIO:
    """
    Minimal file-like object that mirrors writes to two streams.

    Used to simultaneously capture tool stdout/stderr and stream it to console.
    """

    def __init__(self, a: Any, b: Any):
        self._a = a
        self._b = b

    def write(self, s: str) -> int:  # type: ignore[override]
        n1 = 0
        n2 = 0
        try:
            n1 = int(self._a.write(s))
        except Exception:
            n1 = 0
        try:
            n2 = int(self._b.write(s))
        except Exception:
            n2 = 0
        return max(n1, n2)

    def flush(self) -> None:  # type: ignore[override]
        try:
            self._a.flush()
        except Exception:
            pass
        try:
            self._b.flush()
        except Exception:
            pass

    def isatty(self) -> bool:  # tqdm checks this
        try:
            return bool(self._a.isatty())
        except Exception:
            return False

    @property
    def encoding(self) -> str:  # type: ignore[override]
        try:
            return str(getattr(self._a, "encoding", "utf-8") or "utf-8")
        except Exception:
            return "utf-8"


def execute_blendsql(
    *,
    db_path: str,
    blendsql: str,
    model: ModelBase,
    verbose: bool,
    timeout_s: int,
    capture_logs: bool,
    tee_logs_to_console: bool = False,
    duckdb_readonly: bool = False,
    ingredients: Optional[set[type]] = None,
    heartbeat_desc: str | None = None,
) -> ExecuteBlendSQLResult:
    """
    Execute one BlendSQL string against a db, with optional timeout/log capture.
    """

    class _Timeout(Exception):
        pass

    def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]  # pragma: no cover
        raise _Timeout(f"BlendSQL timed out after {timeout_s}s")

    # IMPORTANT: Under multiprocessing, multiple processes may open the same DuckDB file.
    # Opening the file as read-only avoids "Conflicting lock is held" errors and is safe
    # for BlendSQL, since it uses temp tables for intermediate results.
    bsql_db: Any = db_path
    try:
        if bool(duckdb_readonly) and str(db_path).strip().lower().endswith(".duckdb"):
            import duckdb  # type: ignore
            from blendsql.db.duckdb import DuckDB as _BlendDuckDB  # type: ignore

            con = duckdb.connect(str(db_path), read_only=True)
            bsql_db = _BlendDuckDB(con=con, db_url=str(Path(str(db_path)).resolve()))
    except Exception:
        # Fall back to BlendSQL's internal db inference if anything goes wrong.
        bsql_db = db_path

    # Wrap model.reset_stats() so we can capture counters before
    # BlendSQL's finally block zeroes them out — both on success and on error.
    _orig_reset_stats = model.reset_stats

    def _patched_reset_stats():
        model._pre_reset_cache_hits = getattr(model, "num_cache_hits", 0)
        model._pre_reset_prompt_tokens = getattr(model, "prompt_tokens", 0)
        model._pre_reset_completion_tokens = getattr(model, "completion_tokens", 0)
        model._pre_reset_generation_calls = getattr(model, "num_generation_calls", 0)
        _orig_reset_stats()

    model.reset_stats = _patched_reset_stats  # type: ignore[method-assign]

    bsql_kwargs: dict = dict(model=model, verbose=bool(verbose))
    if ingredients is not None:
        bsql_kwargs["ingredients"] = ingredients
    bsql = BlendSQL(bsql_db, **bsql_kwargs)
    bsql_str = normalize_blendsql_query(blendsql)

    exec_stdout_buf = io.StringIO()
    exec_stderr_buf = io.StringIO()

    if int(timeout_s) > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))

    try:
        ctx = contextlib.ExitStack()
        with ctx:
            if capture_logs:
                if tee_logs_to_console:
                    # Keep a handle to the original streams so we can still print to console.
                    tee_out = _TeeTextIO(sys.stdout, exec_stdout_buf)
                    tee_err = _TeeTextIO(sys.stderr, exec_stderr_buf)
                    ctx.enter_context(contextlib.redirect_stdout(tee_out))
                    ctx.enter_context(contextlib.redirect_stderr(tee_err))
                else:
                    ctx.enter_context(contextlib.redirect_stdout(exec_stdout_buf))
                    ctx.enter_context(contextlib.redirect_stderr(exec_stderr_buf))

            run_fn = lambda: bsql.execute(bsql_str)
            if heartbeat_desc:
                smoothie = _with_heartbeat_tqdm(desc=str(heartbeat_desc), fn=run_fn)
            else:
                smoothie = run_fn()
    except Exception as _exec_exc:
        # BlendSQL's finally block already called _patched_reset_stats, which saved
        # the accumulated counters before zeroing them. Attach partial token usage to
        # the exception so callers can log costs even for failed runs.
        _p_in = max(0, int(getattr(model, "_pre_reset_prompt_tokens", 0) or 0))
        _p_out = max(0, int(getattr(model, "_pre_reset_completion_tokens", 0) or 0))
        _p_calls = max(0, int(getattr(model, "_pre_reset_generation_calls", 0) or 0))
        _exec_exc.partial_token_usage = {  # type: ignore[attr-defined]
            "input_tokens": _p_in,
            "output_tokens": _p_out,
            "total_tokens": _p_in + _p_out,
        }
        _exec_exc.partial_num_generation_calls = _p_calls  # type: ignore[attr-defined]
        raise
    finally:
        if int(timeout_s) > 0:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            except Exception:
                pass

    df = None
    if hasattr(smoothie, "df") and callable(getattr(smoothie, "df")):
        df = smoothie.df()
    elif hasattr(smoothie, "to_csv") and callable(getattr(smoothie, "to_csv")):
        df = smoothie

    # BlendSQL calls model.reset_stats() in its finally block before returning,
    # so the model counters are always 0 by the time we read them.
    # smoothie.meta is populated from the model *before* reset_stats(), so read
    # token usage from there instead.
    meta = getattr(smoothie, "meta", None)
    if meta is not None:
        d_prompt = max(0, int(getattr(meta, "prompt_tokens", 0) or 0))
        d_completion = max(0, int(getattr(meta, "completion_tokens", 0) or 0))
        d_calls = max(0, int(getattr(meta, "num_generation_calls", 0) or 0))
    else:
        d_prompt = 0
        d_completion = 0
        d_calls = 0

    # num_cache_hits is tracked on the model but not exposed via SmoothieMeta.
    # We capture it via _pre_reset_stats saved before BlendSQL's reset_stats().
    d_cache_hits = max(0, int(getattr(model, "_pre_reset_cache_hits", 0) or 0))

    return ExecuteBlendSQLResult(
        smoothie=smoothie,
        df=df,
        exec_stdout=exec_stdout_buf.getvalue(),
        exec_stderr=exec_stderr_buf.getvalue(),
        token_usage={
            "input_tokens": d_prompt,
            "output_tokens": d_completion,
            "total_tokens": d_prompt + d_completion,
        },
        num_generation_calls=d_calls,
        num_cache_hits=d_cache_hits,
    )


class LLMOpHostedModel(ModelBase):
    """
    BlendSQL model wrapper backed by ``utils.llm.llmop_call_async``.

    Configure via:
      - LLMOP_MODEL=<model id>  (e.g. "bedrock/us.anthropic.claude-sonnet-4-6" or "gpt-4o")
      - For Bedrock: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ optional AWS_SESSION_TOKEN / AWS_REGION)
      - For OpenAI-compatible: LLMOP_API_KEY (+ optional LLMOP_BASE_URL)
    """

    def __init__(self, model_name_or_path: str | None = None, caching: bool = False, **kwargs):
        from utils.llm import get_llmop_model  # deferred to keep deps optional

        model = (model_name_or_path or "").strip() or get_llmop_model()
        super().__init__(
            model_name_or_path=model,
            caching=caching,
            _allows_parallel_requests=True,
            **kwargs,
        )

    async def generate(self, item: GenerationItem, cancel_event: asyncio.Event | None = None):
        from utils.llm import llmop_call_async

        prompt = item.prompt
        if item.assistant_continuation:
            prompt = f"{prompt}\n{item.assistant_continuation}"

        out_text, usage = await llmop_call_async(model=self.model_name_or_path, prompt=prompt)
        self.num_generation_calls += 1
        if isinstance(usage, dict):
            self.prompt_tokens += int(usage.get("input_tokens") or 0)
            self.completion_tokens += int(usage.get("output_tokens") or 0)

        return GenerationResult(item.identifier, out_text, completed=not (cancel_event and cancel_event.is_set()))


def load_jsonl_from_dir(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def db_path_from_query(db_name: str, repo_root: Path) -> str:
    """
    Resolve the on-disk DB path for a SWAN `db` id.

    Prefer the repo's `swan/database/<db>.duckdb` (what `run_swan_blendsql.py` uses),
    and fall back to an older sqlite layout if present.
    """
    name = str(db_name or "").strip()
    if not name:
        raise ValueError("db name is empty")

    # Preferred: repo-local DuckDB layout.
    duck = (repo_root / "swan" / "database" / f"{name}.duckdb").resolve()
    if duck.is_file():
        return str(duck)

    # Legacy: sqlite layout (kept for compatibility with earlier experiments).
    sqlite = (repo_root / "SWAN" / "databases" / "dev_databases" / f"{name}" / f"{name}.sqlite").resolve()
    if sqlite.is_file():
        return str(sqlite)

    # Last resort: return the duckdb path (BlendSQL/DuckDB will raise a useful error).
    return str(duck)


def normalize_blendsql_query(q: str) -> str:
    """Normalize SWAN BlendSQL strings for this BlendSQL version.

    - SWAN JSONL encodes LLMMap options as a semicolon-delimited *string*.
      BlendSQL expects an options *tuple*, e.g. `options=('a','b')`.
    - SWAN uses `table::column` for value references; this BlendSQL build expects
      `table.column`, so we convert `::` to `.`.
    """

    def _options_repl(m: re.Match) -> str:
        raw = m.group(1)
        opts = [o.strip() for o in raw.split(";") if o.strip()]
        return "options=(" + ", ".join(repr(o) for o in opts) + ")"

    # options='a;b;c' or options="a;b;c"
    q = re.sub(r"options\s*=\s*'([^']*)'", _options_repl, q)
    q = re.sub(r'options\s*=\s*"([^"]*)"', _options_repl, q)

    # table::column -> table.column
    q = q.replace("::", ".")

    # Fix common SWAN formatting issues that crash BlendSQL/sqlglot:
    #
    # 1) Missing commas between adjacent string-literal args inside BlendSQL calls, e.g.
    #    LLMMap('q' 't.col') -> LLMMap('q', 't.col')
    q = re.sub(r"('(?:[^'\\\\]|\\\\.)*')(\s+)(')", r"\1, \3", q)

    # 2) Trailing/extra whitespace inside value-reference strings, e.g.
    #    'T1.col ' -> 'T1.col'  and  'T1 . col' -> 'T1.col'
    def _trim_ref_in_str(m: re.Match) -> str:
        a = (m.group(1) or "").strip()
        b = (m.group(2) or "").strip()
        return f"'{a}.{b}'"

    q = re.sub(r"'([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*'", _trim_ref_in_str, q)

    # 3) options=(table.column) is meant to refer to a set of DB-backed options.
    #    BlendSQL expects a subquery-like value (so it can execute and turn into a Python list).
    #    Rewrite to: options=(SELECT DISTINCT column FROM table)
    def _options_col_to_subquery(m: re.Match) -> str:
        tbl = (m.group(1) or "").strip()
        col = (m.group(2) or "").strip()
        return f"options=(SELECT DISTINCT {col} FROM {tbl})"

    q = re.sub(
        r"options\s*=\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        _options_col_to_subquery,
        q,
    )

    # 4) DuckDB strftime signature is strftime(timestamp/date, format_string).
    #    SWAN sometimes uses SQLite order: strftime('%Y', col). Flip + cast defensively.
    def _fix_strftime(m: re.Match) -> str:
        fmt = m.group(1)
        expr = (m.group(2) or "").strip()
        return f"strftime(CAST({expr} AS TIMESTAMP), '{fmt}')"

    q = re.sub(
        r"(?i)\bstrftime\s*\(\s*'([^']*)'\s*,\s*([A-Za-z_\"].*?)\s*\)",
        _fix_strftime,
        q,
    )

    # 4a) Fix a known SWAN pattern (european_football_2_21) where LLMMap is applied over a CTE name
    # (`player_key`) which BlendSQL's ingredient executor then tries to query as a real table,
    # causing: Catalog Error: Table with name player_key does not exist!
    #
    # Rewrite to a single self-contained `temp` CTE that maps over the real `Player` table, and
    # uses player_api_id to pick the tallest/shortest by inferred height.
    if re.search(r"(?is)\bwith\s+player_key\s+as\s*\(", q) and ("Provide the height in cm" in q):
        q = (
            "WITH temp AS (\n"
            "    SELECT T1.player_api_id,\n"
            "           {{\n"
            "               LLMMap(\n"
            "                   'Provide the height in cm.',\n"
            "                   'Player::player_name',\n"
            "                   return_type='int'\n"
            "               )\n"
            "           }} AS height\n"
            "    FROM Player AS T1\n"
            ")\n"
            "SELECT A\n"
            "FROM (\n"
            "    SELECT AVG(T3.finishing) AS result, 'Max' AS A\n"
            "    FROM Player AS T1\n"
            "    INNER JOIN Player_Attributes AS T3 ON T1.player_api_id = T3.player_api_id\n"
            "    WHERE T1.player_api_id IN (\n"
            "        SELECT player_api_id FROM temp WHERE height = (SELECT MAX(height) FROM temp)\n"
            "    )\n"
            "    UNION\n"
            "    SELECT AVG(T3.finishing) AS result, 'Min' AS A\n"
            "    FROM Player AS T1\n"
            "    INNER JOIN Player_Attributes AS T3 ON T1.player_api_id = T3.player_api_id\n"
            "    WHERE T1.player_api_id IN (\n"
            "        SELECT player_api_id FROM temp WHERE height = (SELECT MIN(height) FROM temp)\n"
            "    )\n"
            ") AS result_table\n"
            "ORDER BY result DESC\n"
            "LIMIT 1;\n"
        )

    # 5) Some SWAN queries include `, WITH <cte> AS (...)` which is invalid; it should be `, <cte> AS (...)`.
    #    (Single WITH introduces all CTEs.)
    q = re.sub(r"(?i),\s*with\s+", ", ", q)

    # 6) Recursive CTEs need `WITH RECURSIVE` in DuckDB when the CTE self-references.
    #    Heuristic: if we see `WITH <name> AS (` and later `FROM <name>` anywhere in the query,
    #    assume recursion is intended and upgrade the first WITH.
    if re.search(r"(?i)\bwith\s+(?!recursive\b)", q):
        cte_names = {
            m.group(1)
            for m in re.finditer(r"(?i)\bwith\s+([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(", q)
        }
        if not cte_names:
            cte_names = {
                m.group(1)
                for m in re.finditer(r"(?i)(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(", q)
            }
        for name in cte_names:
            if re.search(rf"(?i)\bfrom\s+{re.escape(name)}\b", q):
                q = re.sub(r"(?i)\bwith\b", "WITH RECURSIVE", q, count=1)
                break

    # Backtick-quoted identifiers (MySQL-style) -> double quotes (DuckDB/sqlglot-friendly)
    q = q.replace("`", '"')
    return q


def run_one(query: dict, model: ModelBase, row_limit: int, timeout_s: int):
    qid = str(query.get("question_id") or "").strip()
    db = str(query.get("db") or "").strip()
    print(f"Running question_id={qid} db={db}", flush=True)
    blendsql = normalize_blendsql_query(query["blendsql"])
    if row_limit > 0:
        # Debug helper: if the SWAN query uses a `WITH temp AS (<select ...>)` CTE,
        # inject a LIMIT into that materialization to keep LLMMap from mapping over
        # thousands of rows.
        m = re.search(r"(?is)(\bwith\s+temp\s+as\s*\(\s*)(select[\s\S]*?)(\)\s*select\b)", blendsql)
        if m and (" limit " not in m.group(2).lower()):
            inner = m.group(2).rstrip().rstrip(";")
            blendsql = blendsql[: m.start(2)] + (inner + f"\nLIMIT {row_limit}\n") + blendsql[m.end(2) :]
            print(f"Injected LIMIT {row_limit} into temp CTE for debugging", flush=True)
    t0 = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    _clear_token_estimates()
    res = execute_blendsql(
        db_path=db_path_from_query(db, repo_root),
        blendsql=blendsql,
        model=model,
        ingredients={LLMQA, LLMMap, _LLMJoin} if _env_int("BLENDSQL_TOKEN_ESTIMATE_SAMPLE", 0) > 0 else None,
        verbose=True,
        timeout_s=int(timeout_s),
        capture_logs=False,
        duckdb_readonly=True,
        heartbeat_desc=f"Executing {qid or 'query'}",
    )
    print(f"BlendSQL execute done in {time.time() - t0:.2f}s", flush=True)
    if res.df is not None and hasattr(res.df, "__repr__"):
        print(res.df)
    ests = get_token_estimates()
    if ests:
        try:
            pred_in = sum(int(e.predicted_token_usage.get("input_tokens") or 0) for e in ests)
            pred_out = sum(int(e.predicted_token_usage.get("output_tokens") or 0) for e in ests)
            pred_total = sum(int(e.predicted_token_usage.get("total_tokens") or 0) for e in ests)
            print(
                f"\nToken estimate (sample={_env_int('BLENDSQL_TOKEN_ESTIMATE_SAMPLE', 0)}): "
                f"pred_input={pred_in} pred_output={pred_out} pred_total={pred_total}",
                flush=True,
            )
        except Exception:
            pass


def __main__():
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Run a single (or a few) SWAN BlendSQL items for debugging.")
    ap.add_argument(
        "--query_file",
        default=str((repo_root / "swan" / "swan.jsonl").resolve()),
        help="Path to SWAN swan.jsonl (default: repo_root/swan/swan.jsonl).",
    )
    ap.add_argument(
        "--run_id",
        action="append",
        default=[],
        help="Run only these question_id values (repeatable). Exact match.",
    )
    ap.add_argument(
        "--run_db",
        action="append",
        default=[],
        help="Run only these db values (repeatable).",
    )
    ap.add_argument("--limit", type=int, default=1, help="Max number of matched items to run (default: 1).")
    ap.add_argument("--timeout_s", type=int, default=180, help="Per-item timeout (default: 180).")
    ap.add_argument("--estimate_tokens", action="store_true", help="Estimate token usage by sampling LLMMap/LLMQA calls.")
    ap.add_argument("--sample_n", type=int, default=10, help="Sample size per LLMMap call when --estimate_tokens is set.")
    args = ap.parse_args()

    data = load_jsonl_from_dir(str(Path(args.query_file).resolve()))
    print(f"Loaded {len(data)} items from {args.query_file}", flush=True)

    model_env = (os.getenv("LLMOP_MODEL") or "").strip()
    # Bridge LLMOP_* → BlendSQL internal env vars so callers only need LLMOP_* names.
    if os.getenv("LLMOP_CONCURRENCY") and not os.getenv("BLENDSQL_ASYNC_LIMIT"):
        os.environ["BLENDSQL_ASYNC_LIMIT"] = os.environ["LLMOP_CONCURRENCY"]
    if os.getenv("LLMOP_ROW_LIMIT") and not os.getenv("BLENDSQL_ROW_LIMIT"):
        os.environ["BLENDSQL_ROW_LIMIT"] = os.environ["LLMOP_ROW_LIMIT"]
    async_limit = (os.getenv("BLENDSQL_ASYNC_LIMIT") or "").strip()
    row_limit = int((os.getenv("BLENDSQL_ROW_LIMIT") or "0").strip() or "0")
    print(
        f"LLMOP_MODEL={model_env or '<unset>'}  "
        f"LLMOP_CONCURRENCY={async_limit or '<default>'}  LLMOP_ROW_LIMIT={row_limit}",
        flush=True,
    )

    model = LLMOpHostedModel(caching=True)

    want_qids = [str(x).strip() for x in (args.run_id or []) if str(x).strip()]
    want_dbs = [str(x).strip() for x in (args.run_db or []) if str(x).strip()]
    if want_qids:
        qid_set = set(want_qids)
        data = [q for q in data if str(q.get("question_id") or "").strip() in qid_set]
    elif want_dbs:
        db_set = set(want_dbs)
        data = [q for q in data if str(q.get("db") or "").strip() in db_set]

    if not data:
        print("No matching items found.", flush=True)
        return

    lim = max(1, int(args.limit))
    timeout_s = max(1, int(args.timeout_s))
    if bool(args.estimate_tokens):
        os.environ["BLENDSQL_TOKEN_ESTIMATE_SAMPLE"] = str(max(1, int(args.sample_n)))
    for query in data[:lim]:
        run_one(query, model, row_limit, timeout_s)


if __name__ == "__main__":
    __main__()
