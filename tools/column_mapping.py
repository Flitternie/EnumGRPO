from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.edit_duckdb import exec_edit_sql
from tools.tool_base import ToolPlugin, ToolRuntime
from utils.db import (
    attach_extra_required_dbs,
    append_edits_log,
    exec_sql_steps_on_conn,
    quote_ident,
    quote_table_name,
    run_readonly_query_on_conn,
    sql_literal,
    validate_type_sql,
    validate_where_clause,
)


def _parse_mapping(mapping: Any) -> List[Tuple[Any, Any]]:
    """
    Accept:
      - object: {"old": "new", ...}
      - list: [{"from": ..., "to": ...}, ...]
    """
    out: List[Tuple[Any, Any]] = []
    if mapping is None:
        raise ValueError("mapping is required")

    if isinstance(mapping, dict):
        for k, v in mapping.items():
            out.append((k, v))
    elif isinstance(mapping, list):
        for item in mapping:
            if not isinstance(item, dict):
                raise ValueError("mapping list items must be objects")
            if "from" not in item or "to" not in item:
                raise ValueError("mapping items must include 'from' and 'to'")
            out.append((item.get("from"), item.get("to")))
    else:
        raise ValueError("mapping must be an object or a list")

    if not out:
        raise ValueError("mapping must be non-empty")
    if len(out) > 5000:
        raise ValueError("mapping too large; send in chunks (max 5000 pairs)")

    # Validate supported literal types early
    for (k, v) in out:
        if not isinstance(k, (int, float, str, bool)) and k is not None:
            raise ValueError("mapping 'from' values must be JSON scalars or null")
        if not isinstance(v, (int, float, str, bool)) and v is not None:
            raise ValueError("mapping 'to' values must be JSON scalars or null")

    return out


def _chunk_pairs(pairs: List[Tuple[Any, Any]], n: int) -> List[List[Tuple[Any, Any]]]:
    return [pairs[i : i + n] for i in range(0, len(pairs), n)]


def _column_mapping_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "column_mapping",
        "description": (
            "Apply a value->value mapping to a column. Can either replace the original column values or create a new "
            "column and write mapped values there (optionally keeping original values for unmapped rows)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Target table: table or schema.table"},
                "source_column": {"type": "string", "description": "Column whose values are mapped."},
                "target_column": {
                    "type": ["string", "null"],
                    "description": "Column to write to. If null, replaces source_column.",
                },
                "create_column": {
                    "type": ["boolean", "null"],
                    "description": "If true, adds target_column first (requires new_column_type).",
                },
                "new_column_type": {
                    "type": ["string", "null"],
                    "description": "SQL type for new column (e.g. VARCHAR, INTEGER).",
                },
                "mapping": {
                    "type": ["object", "array"],
                    "description": (
                        "Either an object {from: to, ...} (object keys are strings) or a list of {from,to} objects. "
                        "'from' may be null to match NULL (supported in list form)."
                    ),
                    "anyOf": [
                        {
                            "type": "object",
                            "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
                        },
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": ["string", "number", "boolean", "null"]},
                                    "to": {"type": ["string", "number", "boolean", "null"]},
                                },
                                "required": ["from", "to"],
                                "additionalProperties": False,
                            },
                            "minItems": 1,
                        },
                    ],
                },
                "unmapped_behavior": {
                    "type": ["string", "null"],
                    "enum": ["keep", "null", None],
                    "description": (
                        "What to do for values not present in mapping. "
                        "keep (default): keep existing/source value. null: set to NULL."
                    ),
                },
                "where": {
                    "type": ["string", "null"],
                    "description": "Optional WHERE expression (single expression; no semicolons).",
                },
                "target_db_path": {
                    "type": ["string", "null"],
                    "description": "Same as edit_duckdb: existing editable DB under db_edit_dir to continue editing.",
                },
                "dest_db_name": {
                    "type": ["string", "null"],
                    "description": "Same as edit_duckdb: optional filename for new cloned edited DB (ignored if target_db_path provided).",
                },
                "preview_sql": {
                    "type": ["string", "null"],
                    "description": "Optional read-only SELECT/WITH to preview after apply.",
                },
                "limit_rows": {
                    "type": ["integer", "null"],
                    "description": "Max rows to return for preview_sql (1-1000). If null, defaults to 200.",
                },
            },
            "required": ["table", "source_column", "mapping"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _column_mapping_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        table = quote_table_name(args.get("table"))
        source_col_raw = str(args.get("source_column") or "").strip()
        source_col = quote_ident(source_col_raw)

        target_raw = args.get("target_column")
        if target_raw is None or (isinstance(target_raw, str) and not target_raw.strip()):
            target_col_raw = source_col_raw
        elif isinstance(target_raw, str):
            target_col_raw = target_raw.strip()
        else:
            raise ValueError("target_column must be a string or null")
        target_col = quote_ident(target_col_raw)

        create_col = args.get("create_column")
        create = bool(create_col) if create_col is not None else False
        if create and target_col_raw == source_col_raw:
            raise ValueError("create_column=true requires target_column different from source_column")
        type_sql = validate_type_sql(args.get("new_column_type")) if create else ""

        where = validate_where_clause(args.get("where"))

        unmapped = args.get("unmapped_behavior")
        if unmapped is None:
            unmapped_behavior = "keep"
        elif isinstance(unmapped, str):
            unmapped_behavior = unmapped.strip().lower() or "keep"
        else:
            raise ValueError("unmapped_behavior must be 'keep', 'null', or null")
        if unmapped_behavior not in ("keep", "null"):
            raise ValueError("unmapped_behavior must be 'keep', 'null', or null")

        pairs = _parse_mapping(args.get("mapping"))

        # Split NULL mapping (needs IS NULL) from non-NULL.
        null_to: Optional[Any] = None
        non_null_pairs: List[Tuple[Any, Any]] = []
        for k, v in pairs:
            if k is None:
                if null_to is not None:
                    raise ValueError("mapping may include at most one null 'from' entry")
                null_to = v
            else:
                non_null_pairs.append((k, v))

        sql_steps: List[str] = []

        if create:
            sql_steps.append(f"ALTER TABLE {table} ADD COLUMN {target_col} {type_sql};")

        # Initialize target for unmapped rows when writing into a different column.
        if target_col_raw != source_col_raw:
            if unmapped_behavior == "keep":
                init = f"UPDATE {table} SET {target_col} = {source_col}"
                if where:
                    init += f" WHERE {where}"
                init += ";"
                sql_steps.append(init)
            elif unmapped_behavior == "null" and not create:
                # Overwrite existing target column to NULL in-scope.
                init = f"UPDATE {table} SET {target_col} = NULL"
                if where:
                    init += f" WHERE {where}"
                init += ";"
                sql_steps.append(init)
        else:
            # Replacing the original column: null behavior means wipe first, then re-apply mapped values.
            if unmapped_behavior == "null":
                wipe = f"UPDATE {table} SET {target_col} = NULL"
                if where:
                    wipe += f" WHERE {where}"
                wipe += ";"
                sql_steps.append(wipe)

        # Apply NULL mapping if present.
        if null_to is not None:
            stmt = f"UPDATE {table} SET {target_col} = {sql_literal(null_to)} WHERE {source_col} IS NULL"
            if where:
                stmt += f" AND ({where})"
            stmt += ";"
            sql_steps.append(stmt)

        # Apply non-null mappings using VALUES join (chunked).
        for chunk in _chunk_pairs(non_null_pairs, 1000):
            values_rows = ",\n".join(f"({sql_literal(k)}, {sql_literal(v)})" for (k, v) in chunk)
            values_cte = f"(VALUES\n{values_rows}\n) AS m(src, dst)"
            stmt = (
                f"UPDATE {table} AS t SET {target_col} = m.dst "
                f"FROM {values_cte} WHERE t.{source_col} = m.src"
            )
            if where:
                stmt += f" AND ({where})"
            stmt += ";"
            sql_steps.append(stmt)

        limit_rows = args.get("limit_rows")
        limit = int(limit_rows) if limit_rows is not None else 200

        if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            if getattr(sess, "read_only", False):
                raise ValueError(
                    "column_mapping cannot apply updates on a read-only session. "
                    "Open the session with read_only=false, or use edit_duckdb with a DB under db_edit_dir."
                )
            with sess.lock:
                attach_extra_required_dbs(
                    conn=sess.conn,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                    read_only=True,
                )
                sql_log, statements = exec_sql_steps_on_conn(conn=sess.conn, sql_steps=sql_steps)
                append_edits_log(
                    db_edit_dir=str(rt.db_edit_dir or ""),
                    edited_db_path=str(getattr(sess, "db_path", "") or ""),
                    sql_log=sql_log,
                )
                preview_sql = args.get("preview_sql")
                preview = None
                if isinstance(preview_sql, str) and preview_sql.strip():
                    preview = run_readonly_query_on_conn(conn=sess.conn, sql_text=preview_sql, limit_rows=limit)
                return {
                    "ok": True,
                    "edited_db_path": str(getattr(sess, "db_path", "") or ""),
                    "cloned_from": None,
                    "attached_db_files": [],
                    "sql_log": sql_log,
                    "statements_executed": statements,
                    "preview": preview,
                    "note": "Session mode: updates were applied directly on the existing session connection.",
                }

        return exec_edit_sql(
            sql_steps=sql_steps,
            required_dbs=list(rt.required_dbs),
            db_files_dir=str(rt.db_files_dir or ""),
            db_edit_dir=str(rt.db_edit_dir or ""),
            target_db_path=args.get("target_db_path"),
            dest_db_name=args.get("dest_db_name"),
            preview_sql=args.get("preview_sql"),
            limit_rows=limit,
        )

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_column_mapping_tool(), handler_factory=_column_mapping_handler_factory)




