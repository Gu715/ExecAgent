from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _short_name(qualname: str) -> str:
    if not isinstance(qualname, str) or not qualname:
        return ""
    return qualname.split(".")[-1]


def _owner_class_qualname(pre: dict, qualname: str) -> str:
    if not isinstance(qualname, str) or not qualname:
        return ""
    callables = pre.get("callables", {}) or {}
    classes = pre.get("classes", {}) or {}
    if qualname in classes:
        return qualname
    record = callables.get(qualname)
    if isinstance(record, dict):
        owner = record.get("owner_class")
        return owner if isinstance(owner, str) else ""
    return ""


def resolve_candidates(pre: dict, short_name: str) -> list[str]:
    indexes = pre.get("indexes", {}) or {}
    callable_idx = indexes.get("callable_by_short_name", {}) or {}
    class_idx = indexes.get("class_by_short_name", {}) or {}

    out: list[str] = []
    seen = set()
    for pool in (callable_idx.get(short_name, []), class_idx.get(short_name, [])):
        for x in pool:
            if not isinstance(x, str):
                continue
            x = x.strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
    return out


_OBVIOUS_BUILTINS = {
    "print", "input", "int", "str", "len", "range", "list", "dict", "set", "tuple",
    "min", "max", "sum", "abs", "enumerate", "zip", "sorted", "map", "filter",
    "any", "all", "isinstance", "getattr", "setattr", "hasattr", "open", "super",
    "repr", "format", "type", "id", "iter", "next", "round", "reversed", "callable",
}

_OBVIOUS_METHODS = {
    "append", "extend", "items", "keys", "values", "get", "pop", "update", "split", "join",
    "strip", "lower", "upper", "startswith", "endswith", "format", "encode", "decode",
}

_OBVIOUS_EXCEPTION_CTORS = {
    "Exception", "TypeError", "ValueError", "RuntimeError", "KeyError", "IndexError", "AssertionError",
}


def _dedupe_keep_order(names: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in names:
        if not isinstance(x, str):
            continue
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _keyword_parts(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    parts = [p.strip().lower() for p in re.split(r"[^0-9A-Za-z_\.]+", text) if p.strip()]
    return _dedupe_keep_order(parts)


def _score_text_match(*, keywords: List[str], fields: List[str]) -> int:
    if not keywords:
        return 0
    score = 0
    lowered = [str(f or "").lower() for f in fields]
    for kw in keywords:
        if not kw:
            continue
        for idx, field in enumerate(lowered):
            if not field:
                continue
            if kw == field:
                score += 10 - min(idx, 4)
            elif kw in field:
                score += 6 - min(idx, 4)
            else:
                parts = _keyword_parts(field)
                if kw in parts:
                    score += 5 - min(idx, 4)
    return score


def get_unit_code(pre: dict, qualname: str, main_code: str = "", dep_sources: dict | None = None) -> str:
    if qualname == "<entry>":
        return str(((pre.get("project", {}) or {}).get("entry_script", "") or "")).rstrip() + "\n"

    callables = pre.get("callables", {}) or {}
    record = callables.get(qualname)
    if isinstance(record, dict):
        return str(record.get("code", "") or "").rstrip() + "\n"

    classes = pre.get("classes", {}) or {}
    cls = classes.get(qualname)
    if isinstance(cls, dict):
        return str(cls.get("code", "") or "").rstrip() + "\n"

    raise ValueError(f"unknown callable/class qualname: {qualname}")


def get_entry_code(pre: dict, main_code: str = "", entry_name: str = "", dep_sources: dict | None = None) -> dict:
    entry_script = str(((pre.get("project", {}) or {}).get("entry_script", "") or "")).strip()
    if entry_script:
        return {
            "mode": "single",
            "entry": {
                "name": "entry",
                "qualname": "<entry>",
                "code": entry_script,
            },
        }

    entry = pre.get("entry", {}) or {}
    if isinstance(entry.get("code"), str) and entry.get("code", "").strip():
        return {
            "mode": "single",
            "entry": {
                "name": "entry",
                "qualname": "<entry>",
                "code": str(entry.get("code", "")).strip(),
            },
        }

    return {"mode": "error", "error": "missing project.entry_script in pre_process_v2 output"}


def get_class_context(
    pre: dict,
    target_qualname: str,
    main_code: str = "",
    dep_sources: dict | None = None,
    *,
    include_class_statements: bool = True,
    include_init: bool = True,
    include_dunder_methods: bool = True,
    include_target_method: bool = False,
    extra_shared_method_names: Optional[list[str]] = None,
) -> dict:
    class_qn = _owner_class_qualname(pre, target_qualname)
    if not class_qn:
        return {
            "mode": "no_class",
            "target_qualname": target_qualname,
            "class_qualname": "",
            "class_name": "",
            "shared_context_code": "",
            "shared_items": [],
            "class_statements": [],
            "shared_methods": [],
            "all_methods": [],
        }

    class_info = ((pre.get("classes", {}) or {}).get(class_qn) or {})
    shared_context_code = str(class_info.get("shared_context_code", "") or "")
    return {
        "mode": "ok",
        "target_qualname": target_qualname,
        "class_name": _short_name(class_qn),
        "class_qualname": class_qn,
        "class_code": str(class_info.get("code", "") or ""),
        "class_statements": [],
        "shared_methods": [],
        "all_methods": list(class_info.get("methods", []) or []),
        "shared_items": [],
        "shared_context_code": shared_context_code,
    }


def get_frame_class_contexts(
    pre: dict,
    unit_qualnames: list[str],
    main_code: str = "",
    dep_sources: dict | None = None,
    **kwargs: Any,
) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for qualname in unit_qualnames or []:
        if not isinstance(qualname, str) or not qualname or qualname == "<entry>":
            continue
        ctx = get_class_context(pre, qualname, main_code, dep_sources=dep_sources, **kwargs)
        if ctx.get("mode") != "ok":
            continue
        class_qn = str(ctx.get("class_qualname", "") or "").strip()
        shared_code = str(ctx.get("shared_context_code", "") or "").strip()
        if not class_qn or not shared_code or class_qn in seen:
            continue
        seen.add(class_qn)
        out.append(ctx)
    return out


def get_frame_class_context_code(
    pre: dict,
    unit_qualnames: list[str],
    main_code: str = "",
    dep_sources: dict | None = None,
    **kwargs: Any,
) -> str:
    contexts = get_frame_class_contexts(pre, unit_qualnames, main_code, dep_sources=dep_sources, **kwargs)
    parts: list[str] = []
    for ctx in contexts:
        class_name = str(ctx.get("class_name", "") or "").strip() or _short_name(str(ctx.get("class_qualname", "")))
        shared_code = str(ctx.get("shared_context_code", "") or "").rstrip()
        if not shared_code:
            continue
        parts.extend([
            f"[CLASS {class_name} SHARED CONTEXT]",
            "```python",
            shared_code,
            "```",
            "",
        ])
    return "\n".join(parts).rstrip()


def _callsite_stmt(cs: dict) -> str:
    stmt = str(cs.get("stmt", "") or "").strip()
    if stmt:
        return stmt
    return str(cs.get("call_expr", "") or "").strip()


def _callsite_expr(cs: dict) -> str:
    return str(cs.get("call_expr", "") or _callsite_stmt(cs)).strip()


def _should_skip_for_route(cs: dict) -> bool:
    if not isinstance(cs, dict):
        return True

    resolved = cs.get("resolved", []) or []
    candidate_resolved = cs.get("candidate_resolved", []) or []
    if resolved or candidate_resolved:
        return False

    receiver = cs.get("receiver_expr")
    if isinstance(receiver, str) and receiver in {"self", "cls"}:
        return False

    callee_short = str(cs.get("callee_short", "") or "").strip()
    expr = _callsite_expr(cs)

    if callee_short in _OBVIOUS_BUILTINS:
        return True
    if callee_short in _OBVIOUS_EXCEPTION_CTORS:
        return True
    if callee_short in _OBVIOUS_METHODS:
        return True
    if expr.startswith("TypeError(") or expr.startswith("ValueError(") or expr.startswith("CommandError("):
        return True
    return False


def _lookup_unit_callsites(pre: dict, function_name: str) -> tuple[list[dict], set[str]]:
    if function_name == "<entry>":
        entry = pre.get("entry", {}) or {}
        decls = entry.get("decls", {}) or {}
        return list(entry.get("callsites", []) or []), set(decls.get("nested_defs", []) or [])

    callables = pre.get("callables", {}) or {}
    record = callables.get(function_name)
    if not isinstance(record, dict):
        cands = resolve_candidates(pre, function_name)
        if len(cands) == 1:
            record = callables.get(cands[0])

    if isinstance(record, dict):
        decls = record.get("decls", {}) or {}
        return list(record.get("callsites", []) or []), set(decls.get("nested_defs", []) or [])

    classes = pre.get("classes", {}) or {}
    if function_name in classes:
        # Class code may be materialized into the frame as context/support code.
        # It is not a normal callable body, so it has no routeable callsites here.
        return [], set()

    raise ValueError(f"unknown callable qualname: {function_name}")

def _canonicalize_constructor_target(pre: dict, qualname: str) -> str:
    if not isinstance(qualname, str) or not qualname.strip():
        return ""
    qualname = qualname.strip()

    callables = pre.get("callables", {}) or {}
    classes = pre.get("classes", {}) or {}

    if qualname in callables:
        return qualname

    if qualname in classes:
        init_qn = f"{qualname}.__init__"
        if init_qn in callables:
            return init_qn
        # Keep the class qualname materializable even when there is no explicit __init__.
        # Downstream code can still fetch the class body and include it in the frame/prompt.
        return qualname

    return qualname


def _canonicalize_route_qualnames(pre: dict, qualnames: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for qn in qualnames or []:
        new_qn = _canonicalize_constructor_target(pre, qn)
        if not new_qn or new_qn in seen:
            continue
        seen.add(new_qn)
        out.append(new_qn)
    return out

def build_route_items(
    pre: dict,
    function_name: str,
    source: str,
    dep_sources: Optional[Dict[str, str]] = None,
) -> List[dict]:
    callsites, nested_defs = _lookup_unit_callsites(pre, function_name)
    route_items: List[dict] = []

    for cs in callsites:
        if not isinstance(cs, dict):
            continue
        if _should_skip_for_route(cs):
            continue

        callee_short = str(cs.get("callee_short", "") or "").strip()
        if callee_short and callee_short in nested_defs:
            continue

        raw_resolved = _dedupe_keep_order(list(cs.get("resolved", []) or []))
        raw_candidate_resolved = _dedupe_keep_order(list(cs.get("candidate_resolved", []) or []))

        resolved = _canonicalize_route_qualnames(pre, raw_resolved)
        candidate_resolved = _canonicalize_route_qualnames(pre, raw_candidate_resolved)

        # remove duplicates across the two lists
        candidate_resolved = [x for x in candidate_resolved if x not in resolved]

        expr = _callsite_expr(cs)
        stmt = _callsite_stmt(cs)

        # If the only thing we had was a bare class constructor with no explicit __init__,
        # canonicalization turns it into empty -> do not route this callsite.
        if not resolved and not candidate_resolved:
            continue

        route_items.append(
            {
                "expr": expr,
                "stmt": stmt,
                "receiver_expr": cs.get("receiver_expr"),
                "resolved": resolved,
                "candidate_resolved": candidate_resolved,
                "extensions": {
                    "search": {
                        "enabled": True,
                        "queries": [],
                        "notes": "route-time callable-source keyword search is available",
                    }
                },
            }
        )

    return route_items

def search_callable_sources(
    pre: dict,
    keywords: List[str],
    *,
    preferred_qualnames: Optional[List[str]] = None,
    top_k: int = 3,
) -> List[dict]:
    callables = pre.get("callables", {}) or {}
    classes = pre.get("classes", {}) or {}
    preferred = set(preferred_qualnames or [])
    kws = _keyword_parts(" ".join(keywords or []))
    if not kws and not preferred:
        return []

    scored: List[tuple[int, str, dict]] = []
    for qualname, record in {**callables, **classes}.items():
        if not isinstance(record, dict):
            continue
        code = str(record.get("code", "") or "").strip()
        if not code:
            continue
        score = _score_text_match(keywords=kws, fields=[qualname, _short_name(qualname), code])
        if qualname in preferred:
            score += 8
        if score <= 0 and preferred and qualname in preferred:
            score = 8
        if score <= 0:
            continue
        scored.append((score, qualname, record))

    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[dict] = []
    seen = set()
    for score, qualname, record in scored:
        if qualname in seen:
            continue
        seen.add(qualname)
        out.append(
            {
                "qualname": qualname,
                "file_path": str(record.get("file_path", "") or ""),
                "code": str(record.get("code", "") or "").strip(),
                "score": score,
            }
        )
        if len(out) >= max(1, int(top_k)):
            break
    return out


def _binding_record_source(record: Any) -> str:
    if isinstance(record, str):
        return record.strip()
    if isinstance(record, dict):
        return str(record.get("source", "") or "").strip()
    return ""


def _iter_binding_records(pre: dict) -> List[dict]:
    out: List[dict] = []

    entry = pre.get("entry", {}) or {}
    entry_decls = entry.get("decls", {}) or {}
    entry_bindings = entry_decls.get("bindings", {}) or {}
    for name, records in entry_bindings.items():
        for record in records or []:
            src = _binding_record_source(record)
            if not src:
                continue
            out.append(
                {
                    "name": str(name),
                    "scope": "entry",
                    "owner": "<entry>",
                    "file_path": "<entry_script>",
                    "code": src,
                }
            )

    files = pre.get("files", {}) or {}
    for file_path, file_info in files.items():
        tlb = file_info.get("top_level_bindings", {}) or {}
        owner = str(file_info.get("module_name", "") or file_path)
        for name, records in tlb.items():
            for record in records or []:
                src = _binding_record_source(record)
                if not src:
                    continue
                out.append(
                    {
                        "name": str(name),
                        "scope": "module",
                        "owner": owner,
                        "file_path": str(file_path),
                        "code": src,
                    }
                )

    classes = pre.get("classes", {}) or {}
    for class_qn, class_info in classes.items():
        bindings = class_info.get("class_bindings", {}) or {}
        file_path = str(class_info.get("file_path", "") or "")
        for name, records in bindings.items():
            for record in records or []:
                src = _binding_record_source(record)
                if not src:
                    continue
                out.append(
                    {
                        "name": str(name),
                        "scope": "class",
                        "owner": str(class_qn),
                        "file_path": file_path,
                        "code": src,
                    }
                )

    dedup: List[dict] = []
    seen = set()
    for item in out:
        key = (item["scope"], item["owner"], item["name"], item["code"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def search_binding_sources(pre: dict, keywords: List[str], *, top_k: int = 5) -> List[dict]:
    kws = _keyword_parts(" ".join(keywords or []))
    if not kws:
        return []

    scored: List[tuple[int, dict]] = []
    for item in _iter_binding_records(pre):
        score = _score_text_match(
            keywords=kws,
            fields=[item.get("name", ""), item.get("owner", ""), item.get("file_path", ""), item.get("code", "")],
        )
        if score <= 0:
            continue
        scored.append((score, item))

    scored.sort(key=lambda x: (-x[0], x[1].get("name", ""), x[1].get("owner", "")))
    out: List[dict] = []
    for score, item in scored[: max(1, int(top_k))]:
        row = dict(item)
        row["score"] = score
        out.append(row)
    return out
