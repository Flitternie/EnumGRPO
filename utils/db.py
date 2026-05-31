from __future__ import annotations

import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# General Python helpers
# ---------------------------------------------------------------------------

def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def nullsafe_str(v: Any) -> Optional[str]:
    """Coerce a nullable tool argument to str or None.

    LLMs using strict-mode schemas sometimes serialize JSON null as the
    string "null" before the value reaches the handler.  This helper treats
    that string (case-insensitive) as equivalent to None so tool code does
    not have to guard against it everywhere.
    """
    if v is None:
        return None
    s = str(v).strip()
    return None if not s or s.lower() == "null" else s


def _json_safe(v: Any) -> Any:
    return None if _is_nan(v) else v


# ---------------------------------------------------------------------------
# SQL identifier quoting
# ---------------------------------------------------------------------------

# DuckDB (and SQL standard) quoted identifiers use double-quotes and escape a
# literal double-quote by doubling it:  my"col  ->  "my""col"
_QUOTED_IDENT_RE = re.compile(r'^"(?:""|[^"])*"$')


def parse_ident(raw: str) -> str:
    """
    Parse an identifier that may already be double-quoted.

    Examples:
      District Name   -> District Name
      "District Name" -> District Name
      "a""b"          -> a"b
    """
    if not isinstance(raw, str):
        raise ValueError("identifier must be a string")
    s = raw.strip()
    if not s:
        raise ValueError("identifier is required")
    if "\x00" in s:
        raise ValueError("identifier must not contain NUL byte")
    if _QUOTED_IDENT_RE.fullmatch(s):
        return s[1:-1].replace('""', '"')
    return s


def quote_ident(raw: str) -> str:
    """
    Quote/escape an identifier safely for use in SQL.

    Accepts raw identifiers or already-quoted identifiers; always returns a
    double-quoted identifier (except for '*' which is returned unchanged).
    """
    s = parse_ident(raw)
    if s == "*":
        return s
    return '"' + s.replace('"', '""') + '"'


def parse_qualified_name(raw: str, *, max_parts: int = 2) -> List[str]:
    """
    Split a qualified name like schema.table into parts, stripping whitespace and
    allowing quoted parts.

    Note: this intentionally does not try to support dots inside quoted parts.
    """
    s = str(raw or "").strip()
    if not s:
        raise ValueError("name is required")
    parts = [p.strip() for p in s.split(".") if p.strip()]
    if len(parts) < 1 or len(parts) > max_parts:
        raise ValueError("name must be 'table' or 'schema.table'")
    return [parse_ident(p) for p in parts]


def quote_qualified_name(raw: str, *, max_parts: int = 2) -> str:
    parts = parse_qualified_name(raw, max_parts=max_parts)
    return ".".join(quote_ident(p) for p in parts)


def parse_schema_table(raw: str) -> Tuple[Optional[str], str]:
    """
    Return (schema, table) from a raw relation name.
    """
    parts = parse_qualified_name(raw, max_parts=2)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# SQL text processing
# ---------------------------------------------------------------------------

def strip_sql_comments(sql_text: str) -> str:
    """
    Remove SQL comments (--) and (/* */) while preserving quoted strings/identifiers.
    Used for safety/guardrail checks so header comments don't affect validation.
    """
    s = sql_text or ""
    out: List[str] = []
    i = 0
    in_single = False
    in_double = False
    n = len(s)

    while i < n:
        ch = s[i]

        if in_single:
            out.append(ch)
            if ch == "'":
                if i + 1 < n and s[i + 1] == "'":
                    out.append("'")
                    i += 1
                else:
                    in_single = False
            i += 1
            continue

        if in_double:
            out.append(ch)
            if ch == '"':
                if i + 1 < n and s[i + 1] == '"':
                    out.append('"')
                    i += 1
                else:
                    in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double = True
            out.append(ch)
            i += 1
            continue

        if ch == "-" and i + 1 < n and s[i + 1] == "-":
            i += 2
            while i < n and s[i] not in "\r\n":
                i += 1
            continue

        if ch == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i = i + 2 if i + 1 < n else n
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def is_readonly_sql(sql_text: str) -> bool:
    """
    Best-effort guardrail: allow only SELECT/WITH style queries.
    Also block common mutating / unsafe keywords.
    """
    s = strip_sql_comments(sql_text).strip()
    if not s:
        return False
    # Allow a single trailing semicolon, but not multiple statements.
    if ";" in s:
        s2 = s.rstrip()
        if s2.endswith(";"):
            s2 = s2[:-1]
        if ";" in s2:
            return False
        s = s2.strip()
    if re.search(
        r"\b(create|insert|update|delete|drop|alter|attach|detach|copy|export|import|pragma|load|install)\b",
        s,
        flags=re.IGNORECASE,
    ):
        return False
    head = s.lstrip().lower()
    return head.startswith("select") or head.startswith("with")


def resolve_duckdb_path(db_dir: str, db_name: str) -> Optional[str]:
    """
    Resolve a .duckdb file path under db_dir.
    Accepts either "EU_SOCCER" or "EU_SOCCER.duckdb" and matches case-insensitively.
    Returns the absolute path if found, else None.
    """
    if not db_name:
        return None
    name = db_name.strip()
    if name.lower().endswith(".duckdb"):
        name = name[: -len(".duckdb")]
    want = name.upper()
    if not os.path.isdir(db_dir):
        return None
    try:
        for fn in os.listdir(db_dir):
            if not fn.lower().endswith(".duckdb"):
                continue
            stem = fn[: -len(".duckdb")]
            if stem.upper() == want:
                return os.path.join(db_dir, fn)
    except Exception:
        return None
    cand = os.path.join(db_dir, f"{name}.duckdb")
    return cand if os.path.isfile(cand) else None


# ---------------------------------------------------------------------------
# DuckDB connection utilities
# ---------------------------------------------------------------------------

def list_attached_db_paths(conn: Any) -> List[str]:
    """
    Returns absolute paths from PRAGMA database_list (best-effort).
    """
    out: List[str] = []
    try:
        rows = conn.execute("PRAGMA database_list;").fetchall()
        for r in rows:
            # DuckDB returns: (seq, name, file)
            try:
                p = r[2]
            except Exception:
                p = None
            if isinstance(p, str) and p:
                out.append(os.path.abspath(p))
    except Exception:
        pass
    return out


def attach_extra_required_dbs(
    *,
    conn: Any,
    required_dbs: Sequence[str],
    db_files_dir: str,
    read_only: bool = True,
) -> List[str]:
    """
    Attach required_dbs[1:] into the existing connection, if not already attached.
    Returns the basenames of any files attached during this call.
    """
    attached_now: List[str] = []
    if not required_dbs:
        return attached_now
    db_dir = str(db_files_dir or "")
    if not os.path.isdir(db_dir):
        return attached_now

    existing_paths = set(list_attached_db_paths(conn))

    for extra in list(required_dbs)[1:]:
        if not isinstance(extra, str) or not extra.strip():
            continue
        name = extra.strip()
        if name.lower().endswith(".duckdb"):
            name = name[: -len(".duckdb")]
        want = name.upper()

        extra_path: Optional[str] = None
        try:
            for fn in os.listdir(db_dir):
                if not fn.lower().endswith(".duckdb"):
                    continue
                stem = fn[: -len(".duckdb")]
                if stem.upper() == want:
                    extra_path = os.path.join(db_dir, fn)
                    break
        except Exception:
            extra_path = None

        if not extra_path:
            continue
        extra_abs = os.path.abspath(extra_path)
        if extra_abs in existing_paths:
            continue

        alias = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(extra_abs))[0]) or "db"
        safe_path = extra_abs.replace("'", "''")
        ro = " (READ_ONLY)" if read_only else ""
        conn.execute(f"ATTACH DATABASE '{safe_path}' AS {alias}{ro};")
        existing_paths.add(extra_abs)
        attached_now.append(os.path.basename(extra_abs))

    return attached_now


def _check_no_external_access(sql_text: str) -> None:
    """Raise if the SQL tries to discover or attach external files.

    Blocks two cheat vectors:
    - glob() calls (filesystem discovery)
    - ATTACH of any path not already open on the connection
    """
    s = strip_sql_comments(sql_text)
    if re.search(r"\bglob\s*\(", s, flags=re.IGNORECASE):
        raise ValueError(
            "glob() is not allowed. Only the provided database file(s) may be queried."
        )
    if re.search(r"\battach\b", s, flags=re.IGNORECASE):
        raise ValueError(
            "ATTACH is not allowed. Only the provided database file(s) may be queried."
        )


def run_readonly_query_on_conn(*, conn: Any, sql_text: str, limit_rows: int = 200) -> Dict[str, Any]:
    if not is_readonly_sql(sql_text):
        raise ValueError(
            "Only read-only single-statement SQL starting with SELECT or WITH is allowed. "
            "Do not use PRAGMA (e.g., PRAGMA table_info) for schema inspection; use list_relations/describe_relation instead."
        )
    _check_no_external_access(sql_text)

    limit_rows = int(limit_rows or 0)
    if limit_rows <= 0:
        limit_rows = 200
    limit_rows = max(1, min(limit_rows, 1000))

    q = strip_sql_comments(sql_text).strip()
    if q.endswith(";"):
        q = q[:-1].strip()
    wrapped = f"SELECT * FROM ({q}) AS q LIMIT {int(limit_rows)}"
    df = conn.execute(wrapped).fetchdf()

    cols = [str(c) for c in list(df.columns)]
    rows = [[_json_safe(v) for v in r] for r in df.values.tolist()]
    return {
        "columns": cols,
        "rows": rows,
        "row_count": int(len(rows)),
        "limit_rows": int(limit_rows),
    }


def split_sql_statements(sql_text: str) -> List[str]:
    """
    Split SQL text into statements on semicolons, with basic handling of:
    - single quotes ('...'), doubled '' escapes
    - double quotes ("..."), doubled "" escapes
    - line comments (-- ...)
    - block comments (/* ... */)
    """
    s = sql_text or ""
    out: List[str] = []
    buf: List[str] = []
    i = 0
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    n = len(s)
    while i < n:
        ch = s[i]
        nxt = s[i + 1] if i + 1 < n else ""

        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                i += 2
                in_block_comment = False
                continue
            i += 1
            continue

        if not in_single and not in_double:
            if ch == "-" and nxt == "-":
                buf.append(ch)
                buf.append(nxt)
                i += 2
                in_line_comment = True
                continue
            if ch == "/" and nxt == "*":
                buf.append(ch)
                buf.append(nxt)
                i += 2
                in_block_comment = True
                continue

        if ch == "'" and not in_double:
            buf.append(ch)
            if in_single and nxt == "'":
                buf.append(nxt)
                i += 2
                continue
            in_single = not in_single
            i += 1
            continue

        if ch == '"' and not in_single:
            buf.append(ch)
            if in_double and nxt == '"':
                buf.append(nxt)
                i += 2
                continue
            in_double = not in_double
            i += 1
            continue

        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def exec_sql_steps_on_conn(*, conn: Any, sql_steps: List[Any]) -> Tuple[str, List[str]]:
    if not isinstance(sql_steps, list) or not sql_steps:
        raise ValueError("sql_steps must be a non-empty list of strings.")
    statements: List[str] = []
    for step in sql_steps:
        if not isinstance(step, str) or not step.strip():
            continue
        statements.extend(split_sql_statements(step))
    statements = [s.strip() for s in statements if isinstance(s, str) and s.strip()]
    if not statements:
        raise ValueError("No executable SQL statements found in sql_steps.")

    sql_log = "\n\n".join(s.rstrip(";").strip() + ";" for s in statements)

    conn.execute("BEGIN TRANSACTION;")
    try:
        for stmt in statements:
            conn.execute(stmt)
        conn.execute("COMMIT;")
    except Exception:
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
        raise

    return sql_log, statements


# ---------------------------------------------------------------------------
# SQL value literals and clause guardrails
# ---------------------------------------------------------------------------

def sql_literal(v: Any) -> str:
    """
    Render a Python value as a safe SQL literal.

    Supported types: None -> NULL, bool, int, finite float, str.
    NaN and ±inf are rejected; they have no standard SQL literal representation.
    """
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"{v!r} is not a supported SQL literal (NaN/inf)")
        return str(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    raise ValueError(f"Unsupported literal type: {type(v).__name__}")


def quote_table_name(table: str) -> str:
    """Convenience wrapper: quote an unqualified or schema.table name."""
    return quote_qualified_name(table, max_parts=2)


def validate_where_clause(where: Any) -> Optional[str]:
    """
    Validate and normalise a WHERE-clause expression.

    Accepts a single SQL expression (no semicolons, no statement keywords).
    Returns None for empty / null input.
    """
    if where is None:
        return None
    if not isinstance(where, str):
        raise ValueError("where must be a string or null")
    s = where.strip()
    if not s:
        return None
    if ";" in s:
        raise ValueError("where must not contain ';'")
    if re.search(
        r"\b(select|update|delete|insert|alter|drop|create|attach|detach|copy|pragma|load|install)\b",
        s, re.I,
    ):
        raise ValueError("where contains a forbidden keyword")
    return s


def validate_type_sql(type_sql: Any) -> str:
    """
    Validate a SQL type string for ADD COLUMN.

    Allows simple DuckDB type tokens: VARCHAR, INTEGER, DECIMAL(18,2), etc.
    """
    if not isinstance(type_sql, str) or not type_sql.strip():
        raise ValueError("new_column_type is required when create_column=true")
    s = type_sql.strip()
    if ";" in s:
        raise ValueError("new_column_type must not contain ';'")
    if re.search(
        r"\b(select|update|delete|insert|alter|drop|create|attach|detach|copy|pragma|load|install)\b",
        s, re.I,
    ):
        raise ValueError("new_column_type contains a forbidden keyword")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(\s*\(\s*\d+\s*(,\s*\d+\s*)?\))?", s):
        raise ValueError(
            "new_column_type must look like a SQL type (e.g. VARCHAR, INTEGER, DECIMAL(18,2))"
        )
    return s


def append_edits_log(*, db_edit_dir: str, edited_db_path: str, sql_log: str) -> None:
    """
    Best-effort append combined sql_log next to the edited DB.
    Only writes if edited_db_path is under db_edit_dir.
    """
    if not isinstance(sql_log, str) or not sql_log.strip():
        return
    edited_dir_abs = os.path.abspath(str(db_edit_dir or os.getenv("SANDBOX_DIR") or "./sandbox"))
    db_abs = os.path.abspath(str(edited_db_path or ""))
    if not db_abs or not (db_abs == edited_dir_abs or db_abs.startswith(edited_dir_abs + os.sep)):
        return
    stem = os.path.splitext(os.path.basename(db_abs))[0]
    log_path = os.path.join(edited_dir_abs, f"{stem}__edits.sql")
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"-- edits at {ts}\n")
            f.write(sql_log.strip() + ("\n" if not sql_log.endswith("\n") else ""))
            f.write("\n")
    except Exception:
        return
