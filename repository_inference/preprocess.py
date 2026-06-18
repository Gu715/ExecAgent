from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import io
import textwrap
import tokenize

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _sanitize_empty_blocks(code: str, *, pass_indent: int = 4) -> str:
    """Insert ``pass`` into obviously empty suites so partial slices still parse."""
    lines = code.splitlines()
    block_re = re.compile(
        r"^(?P<indent>\s*)(?P<kw>async\s+def|def|class|if|elif|else|for|while|try|except|finally|with|match)\b.*:\s*(#.*)?$"
    )
    out: List[str] = []
    n = len(lines)

    def next_nonempty_idx(j: int) -> int:
        while j < n and lines[j].strip() == "":
            j += 1
        return j

    for i, line in enumerate(lines):
        out.append(line)
        m = block_re.match(line)
        if not m:
            continue
        header_indent = len(m.group("indent"))
        j = next_nonempty_idx(i + 1)
        if j >= n:
            out.append(" " * (header_indent + pass_indent) + "pass")
            continue
        next_line = lines[j]
        next_indent = len(next_line) - len(next_line.lstrip(" "))
        if next_indent <= header_indent:
            out.append(" " * (header_indent + pass_indent) + "pass")

    return "\n".join(out) + ("\n" if code.endswith("\n") else "")


@dataclass
class ParseResult:
    source: str
    tree: Optional[ast.AST]
    sanitized: bool
    parse_error: Optional[str]


@dataclass
class Span:
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int

    def to_dict(self) -> dict:
        return {
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "col_offset": self.col_offset,
            "end_col_offset": self.end_col_offset,
        }


@dataclass
class ImportAlias:
    kind: str  # module | symbol
    bound_name: str
    target: str


@dataclass
class ClassContext:
    class_qualname: str
    module_name: str
    file_path: str


def _parse_with_fallback(code: str) -> ParseResult:
    try:
        return ParseResult(source=code, tree=ast.parse(code), sanitized=False, parse_error=None)
    except SyntaxError as e:
        sanitized = _sanitize_empty_blocks(code)
        try:
            return ParseResult(
                source=sanitized,
                tree=ast.parse(sanitized),
                sanitized=True,
                parse_error=f"initial_parse_error: {e}",
            )
        except SyntaxError as e2:
            return ParseResult(
                source=sanitized,
                tree=None,
                sanitized=True,
                parse_error=f"initial_parse_error: {e}; sanitized_parse_error: {e2}",
            )


def _collect_docstring_line_spans(code: str) -> List[Tuple[int, int]]:
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return []

    spans: List[Tuple[int, int]] = []

    def visit_body(body: List[ast.stmt]) -> None:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            start = getattr(body[0], "lineno", None)
            end = getattr(body[0], "end_lineno", start)
            if isinstance(start, int) and isinstance(end, int):
                spans.append((start, end))

        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit_body(stmt.body)

    visit_body(tree.body)
    return spans


def _strip_python_comments(code: str) -> str:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        kept = [tok for tok in tokens if tok.type != tokenize.COMMENT]
        cleaned = tokenize.untokenize(kept)
    except Exception:
        cleaned = code

    lines = [line.rstrip() for line in cleaned.splitlines()]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _clean_extracted_code(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return code.strip()

    lines = code.splitlines()
    spans = _collect_docstring_line_spans(code)
    if spans:
        remove_lines: Set[int] = set()
        for start, end in spans:
            remove_lines.update(range(start, end + 1))
        lines = [line for idx, line in enumerate(lines, start=1) if idx not in remove_lines]

    cleaned = _strip_python_comments("\n".join(lines))
    return cleaned.strip()

# ---------------------------------------------------------------------------
# Basic AST helpers
# ---------------------------------------------------------------------------


def _span(node: ast.AST) -> Span:
    lineno = getattr(node, "lineno", -1)
    end_lineno = getattr(node, "end_lineno", lineno)
    col_offset = getattr(node, "col_offset", -1)
    end_col_offset = getattr(node, "end_col_offset", col_offset)
    return Span(lineno, end_lineno, col_offset, end_col_offset)


def _stringify_expr(node: Optional[ast.AST], source: str) -> Optional[str]:
    if node is None:
        return None
    seg = ast.get_source_segment(source, node)
    if isinstance(seg, str) and seg.strip():
        return seg.strip()
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:
        return node.__class__.__name__


def _stmt_text(source: str, stmt: Optional[ast.stmt], fallback_node: Optional[ast.AST] = None) -> str:
    node = stmt or fallback_node
    if node is None:
        return ""

    seg = ast.get_source_segment(source, node)
    if isinstance(seg, str) and seg.strip():
        return _clean_extracted_code(seg)

    lineno = getattr(node, "lineno", -1)
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return _clean_extracted_code(lines[lineno - 1])

    return ""


def _slice_source_lines(source: str, start_lineno: int, end_lineno: int) -> str:
    lines = source.splitlines()
    if start_lineno <= 0 or end_lineno < start_lineno or start_lineno > len(lines):
        return ""
    end_excl = min(end_lineno, len(lines))
    return "\n".join(lines[start_lineno - 1:end_excl]).rstrip()


def _node_code_with_decorators(node: ast.AST, source: str) -> str:
    lines = source.splitlines()

    start = getattr(node, "lineno", -1)
    end = getattr(node, "end_lineno", start)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(node, "decorator_list", None):
        deco_lines = [getattr(d, "lineno", start) for d in node.decorator_list if isinstance(getattr(d, "lineno", None), int)]
        if deco_lines:
            start = min(deco_lines)

    if not (1 <= start <= len(lines)) or end < start:
        seg = ast.get_source_segment(source, node) or ""
        return _clean_extracted_code(seg)

    seg = "\n".join(lines[start - 1:end])
    return _clean_extracted_code(seg)


def _extract_store_names(target: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.extend(_extract_store_names(elt))
    return names


def _module_name_from_path(file_path: str) -> str:
    path = file_path.replace("\\", "/").strip("/")
    if path.endswith(".py"):
        path = path[:-3]
    parts = [p for p in path.split("/") if p and p != "."]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    sanitized: List[str] = []
    for p in parts:
        q = re.sub(r"[^0-9A-Za-z_]", "_", p)
        if not q:
            q = "_"
        if q[0].isdigit():
            q = "_" + q
        sanitized.append(q)
    return ".".join(sanitized) if sanitized else "module"


def _module_package(module_name: str, file_path: str) -> str:
    norm = file_path.replace("\\", "/")
    if norm.endswith("/__init__.py") or norm == "__init__.py":
        return module_name
    parts = module_name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def _resolve_relative_module(current_module: str, file_path: str, module: Optional[str], level: int) -> str:
    if level <= 0:
        return module or ""
    package = _module_package(current_module, file_path)
    base_parts = [p for p in package.split(".") if p]
    up = max(level - 1, 0)
    if up:
        if up >= len(base_parts):
            base_parts = []
        else:
            base_parts = base_parts[:-up]
    if module:
        base_parts.extend([p for p in module.split(".") if p])
    return ".".join(base_parts)


def _short_name(qualname: str) -> str:
    return qualname.split(".")[-1] if qualname else ""


def _mangled_private_name(class_qualname: str, method_name: str) -> Optional[str]:
    cls = _short_name(class_qualname)
    if not cls:
        return None
    if method_name.startswith("__") and not method_name.endswith("__"):
        return f"_{cls}{method_name}"
    return None


def _qualname_join(*parts: str) -> str:
    return ".".join([p for p in parts if p])


def _safe_literal_eval(node: ast.AST) -> Optional[Any]:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _rhs_info(node: Optional[ast.AST], source: str) -> Tuple[str, Optional[Any], Optional[str]]:
    if node is None:
        return "none", None, None

    text = _stringify_expr(node, source)

    if isinstance(node, ast.Constant):
        return "literal", node.value, text

    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        value = _safe_literal_eval(node)
        if value is not None:
            return "literal", value, text

    if isinstance(node, ast.Dict):
        value = _safe_literal_eval(node)
        return "dict_literal", value, text

    if isinstance(node, ast.List):
        value = _safe_literal_eval(node)
        return "list_literal", value, text

    if isinstance(node, ast.Set):
        value = _safe_literal_eval(node)
        return "set_literal", value, text

    if isinstance(node, ast.Tuple):
        value = _safe_literal_eval(node)
        return "tuple_literal", value, text

    if isinstance(node, ast.Call):
        return "call", None, text

    if isinstance(node, ast.Name):
        return "name", None, text

    if isinstance(node, ast.Attribute):
        return "attribute", None, text

    if isinstance(node, ast.Lambda):
        return "lambda", None, text

    return "other", None, text


def _parameter_kind_from_ast(index_kind: str) -> str:
    mapping = {
        "posonly": "positional_only",
        "arg": "positional_or_keyword",
        "vararg": "var_positional",
        "kwonly": "keyword_only",
        "kwarg": "var_keyword",
    }
    return mapping.get(index_kind, index_kind)


def _signature_info(node: ast.AST, source: str) -> dict:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    params: List[dict] = []
    args = node.args

    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    positional_defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    for arg_node, default in zip(getattr(args, "posonlyargs", []), positional_defaults[: len(getattr(args, "posonlyargs", []))]):
        params.append(
            {
                "name": arg_node.arg,
                "kind": _parameter_kind_from_ast("posonly"),
                "annotation": _stringify_expr(arg_node.annotation, source),
                "default": _stringify_expr(default, source),
            }
        )

    start = len(getattr(args, "posonlyargs", []))
    for arg_node, default in zip(args.args, positional_defaults[start:]):
        params.append(
            {
                "name": arg_node.arg,
                "kind": _parameter_kind_from_ast("arg"),
                "annotation": _stringify_expr(arg_node.annotation, source),
                "default": _stringify_expr(default, source),
            }
        )

    if args.vararg is not None:
        params.append(
            {
                "name": args.vararg.arg,
                "kind": _parameter_kind_from_ast("vararg"),
                "annotation": _stringify_expr(args.vararg.annotation, source),
                "default": None,
            }
        )

    for arg_node, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            {
                "name": arg_node.arg,
                "kind": _parameter_kind_from_ast("kwonly"),
                "annotation": _stringify_expr(arg_node.annotation, source),
                "default": _stringify_expr(default, source),
            }
        )

    if args.kwarg is not None:
        params.append(
            {
                "name": args.kwarg.arg,
                "kind": _parameter_kind_from_ast("kwarg"),
                "annotation": _stringify_expr(args.kwarg.annotation, source),
                "default": None,
            }
        )

    pieces: List[str] = []
    posonly_count = len(getattr(args, "posonlyargs", []))
    positional_count = len(positional)
    inserted_kw_separator = False

    for idx, p in enumerate(params):
        piece = p["name"]
        if p["kind"] == "var_positional":
            piece = "*" + piece
            inserted_kw_separator = True
        elif p["kind"] == "var_keyword":
            piece = "**" + piece
        elif p["kind"] == "keyword_only" and not inserted_kw_separator:
            pieces.append("*")
            inserted_kw_separator = True

        if p.get("annotation"):
            piece += f": {p['annotation']}"
        if p.get("default") is not None:
            piece += f" = {p['default']}"
        pieces.append(piece)

        if idx + 1 == posonly_count and posonly_count > 0:
            pieces.append("/")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    text = f"{prefix} {node.name}(" + ", ".join(pieces) + ")"
    if node.returns is not None:
        text += f" -> {_stringify_expr(node.returns, source)}"

    return {
        "text": text,
        "return_annotation": _stringify_expr(node.returns, source),
        "parameters": params,
    }


# ---------------------------------------------------------------------------
# Binding extraction
# ---------------------------------------------------------------------------


def _binding_record(
    *,
    name: str,
    stmt: ast.stmt,
    value_node: Optional[ast.AST],
    binding_kind: str,
    source: str,
) -> dict:
    rhs_kind, static_value, rhs_text = _rhs_info(value_node, source)
    record = {
        "name": name,
        "binding_kind": binding_kind,
        "rhs_kind": rhs_kind,
        "rhs_text": rhs_text,
        "source": _stmt_text(source, stmt),
        "span": _span(stmt).to_dict(),
    }
    if static_value is not None:
        record["static_value"] = static_value
    return record


def _collect_named_bindings_from_stmt(stmt: ast.stmt, source: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}

    def add(name: str, record: dict) -> None:
        out.setdefault(name, []).append(record)

    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            for name in _extract_store_names(target):
                add(name, _binding_record(name=name, stmt=stmt, value_node=stmt.value, binding_kind="assign", source=source))
    elif isinstance(stmt, ast.AnnAssign):
        for name in _extract_store_names(stmt.target):
            add(name, _binding_record(name=name, stmt=stmt, value_node=stmt.value, binding_kind="annassign", source=source))
    elif isinstance(stmt, ast.AugAssign):
        for name in _extract_store_names(stmt.target):
            add(name, _binding_record(name=name, stmt=stmt, value_node=stmt.value, binding_kind="augassign", source=source))
    return out


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


def _import_record(stmt: ast.stmt, source: str, module_name: str, file_path: str) -> Tuple[dict, List[ImportAlias]]:
    aliases: List[ImportAlias] = []
    raw_code = _stmt_text(source, stmt)
    span = _span(stmt).to_dict()

    if isinstance(stmt, ast.Import):
        names: List[dict] = []
        for alias in stmt.names:
            bound_name = alias.asname or alias.name.split(".")[0]
            if alias.asname:
                target = alias.name
            else:
                target = alias.name.split(".")[0]
            aliases.append(ImportAlias(kind="module", bound_name=bound_name, target=target))
            names.append(
                {
                    "name": alias.name,
                    "asname": alias.asname,
                    "bound_name": bound_name,
                }
            )
        return (
            {
                "kind": "import",
                "module": None,
                "level": 0,
                "names": names,
                "raw_code": raw_code,
                "span": span,
            },
            aliases,
        )

    assert isinstance(stmt, ast.ImportFrom)
    resolved_module = _resolve_relative_module(module_name, file_path, stmt.module, stmt.level)
    names = []
    for alias in stmt.names:
        bound_name = alias.asname or alias.name
        target = _qualname_join(resolved_module, alias.name)
        aliases.append(ImportAlias(kind="symbol", bound_name=bound_name, target=target))
        names.append(
            {
                "name": alias.name,
                "asname": alias.asname,
                "bound_name": bound_name,
            }
        )
    return (
        {
            "kind": "from_import",
            "module": resolved_module,
            "level": stmt.level,
            "names": names,
            "raw_code": raw_code,
            "span": span,
        },
        aliases,
    )


# ---------------------------------------------------------------------------
# Callsite / decl analysis for callable bodies and entry script
# ---------------------------------------------------------------------------


class BlockAnalyzer(ast.NodeVisitor):
    def __init__(self, source: str, *, current_callable_name: Optional[str] = None) -> None:
        self.source = source
        self.current_callable_name = current_callable_name
        self.current_stmt: Optional[ast.stmt] = None
        self.site_role: str = "body"

        self.locals: Set[str] = set()
        self.nested_defs: Set[str] = set()
        self.callsites: List[dict] = []

        self.loop_depth = 0
        self.cond_depth = 0
        self.comprehension_depth = 0
        self.lambda_depth = 0

    def analyze_statements(self, stmts: Iterable[ast.stmt]) -> None:
        for stmt in stmts:
            self.visit(stmt)

    def analyze_header_exprs(self, exprs: Iterable[ast.AST], *, role: str, stmt_context: Optional[ast.stmt]) -> None:
        old_role = self.site_role
        old_stmt = self.current_stmt
        self.site_role = role
        self.current_stmt = None
        for expr in exprs:
            self.visit(expr)
        self.site_role = old_role
        self.current_stmt = old_stmt

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.stmt):
            prev = self.current_stmt
            self.current_stmt = node
            super().generic_visit(node)
            self.current_stmt = prev
        else:
            super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.locals.add(node.name)
        self.nested_defs.add(node.name)
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.locals.add(node.name)
        self.nested_defs.add(node.name)
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.locals.add(node.name)
        return

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.locals.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.locals.add(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name in _extract_store_names(target):
                self.locals.add(name)

            if isinstance(target, ast.Subscript):
                self._record_protocol_callsite(
                    op_name="__setitem__",
                    receiver_node=target.value,
                    full_node=target,
                    call_expr_override=_stringify_expr(target, self.source) or "",
                )

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        for name in _extract_store_names(node.target):
            self.locals.add(name)

        if isinstance(node.target, ast.Subscript):
            self._record_protocol_callsite(
                op_name="__setitem__",
                receiver_node=node.target.value,
                full_node=node.target,
                call_expr_override=_stringify_expr(node.target, self.source) or "",
            )

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in _extract_store_names(node.target):
            self.locals.add(name)

        if isinstance(node.target, ast.Subscript):
            self._record_protocol_callsite(
                op_name="__setitem__",
                receiver_node=node.target.value,
                full_node=node.target,
                call_expr_override=_stringify_expr(node.target, self.source) or "",
            )

        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                self._record_protocol_callsite(
                    op_name="__delitem__",
                    receiver_node=target.value,
                    full_node=target,
                    call_expr_override=_stringify_expr(target, self.source) or "",
                )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        for name in _extract_store_names(node.target):
            self.locals.add(name)
        old = self.loop_depth
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth = old

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        for name in _extract_store_names(node.target):
            self.locals.add(name)
        old = self.loop_depth
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth = old

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                for name in _extract_store_names(item.optional_vars):
                    self.locals.add(name)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                for name in _extract_store_names(item.optional_vars):
                    self.locals.add(name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.locals.add(node.name)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        old = self.loop_depth
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth = old

    def visit_If(self, node: ast.If) -> None:
        old = self.cond_depth
        self.cond_depth += 1
        self.generic_visit(node)
        self.cond_depth = old

    def visit_Try(self, node: ast.Try) -> None:
        old = self.cond_depth
        self.cond_depth += 1
        self.generic_visit(node)
        self.cond_depth = old

    def visit_TryStar(self, node: ast.TryStar) -> None:  # type: ignore[attr-defined]
        old = self.cond_depth
        self.cond_depth += 1
        self.generic_visit(node)
        self.cond_depth = old

    def visit_Match(self, node: ast.Match) -> None:
        old = self.cond_depth
        self.cond_depth += 1
        self.generic_visit(node)
        self.cond_depth = old

    def visit_ListComp(self, node: ast.ListComp) -> None:
        old = self.comprehension_depth
        self.comprehension_depth += 1
        self.generic_visit(node)
        self.comprehension_depth = old

    def visit_SetComp(self, node: ast.SetComp) -> None:
        old = self.comprehension_depth
        self.comprehension_depth += 1
        self.generic_visit(node)
        self.comprehension_depth = old

    def visit_DictComp(self, node: ast.DictComp) -> None:
        old = self.comprehension_depth
        self.comprehension_depth += 1
        self.generic_visit(node)
        self.comprehension_depth = old

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        old = self.comprehension_depth
        self.comprehension_depth += 1
        self.generic_visit(node)
        self.comprehension_depth = old

    def visit_Lambda(self, node: ast.Lambda) -> None:
        old = self.lambda_depth
        self.lambda_depth += 1
        self.generic_visit(node)
        self.lambda_depth = old

    def _record_protocol_callsite(
        self,
        *,
        op_name: str,
        receiver_node: ast.AST,
        full_node: ast.AST,
        call_expr_override: Optional[str] = None,
    ) -> None:
        receiver_expr = _stringify_expr(receiver_node, self.source)
        call_expr = call_expr_override or (_stringify_expr(full_node, self.source) or "")
        func_expr = f"{receiver_expr}.{op_name}" if receiver_expr else op_name

        lineno = getattr(full_node, "lineno", -1)
        col_offset = getattr(full_node, "col_offset", -1)
        end_lineno = getattr(full_node, "end_lineno", lineno)
        end_col_offset = getattr(full_node, "end_col_offset", col_offset)

        self.callsites.append(
            {
                "call_expr": call_expr,
                "func_expr": func_expr,
                "callee_text": func_expr,
                "callee_short": op_name,
                "call_kind": "attr",
                "receiver_expr": receiver_expr,
                "stmt_text": _stmt_text(self.source, self.current_stmt, full_node),
                "stmt_kind": type(self.current_stmt).__name__ if self.current_stmt is not None else "",
                "site_role": self.site_role,
                "loc": {
                    "lineno": lineno,
                    "col_offset": col_offset,
                    "end_lineno": end_lineno,
                    "end_col_offset": end_col_offset,
                },
                "resolved": [],
                "candidate_resolved": [],
                "context": {
                    "loop_depth": self.loop_depth,
                    "cond_depth": self.cond_depth,
                    "comprehension_depth": self.comprehension_depth,
                    "lambda_depth": self.lambda_depth,
                },
                "arg_count": 0,
                "kwarg_names": [],
                "has_starargs": False,
                "has_kwargs": False,
                "is_recursive": False,
            }
        )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        ctx = getattr(node, "ctx", None)
        if isinstance(ctx, ast.Load):
            self._record_protocol_callsite(
                op_name="__getitem__",
                receiver_node=node.value,
                full_node=node,
                call_expr_override=_stringify_expr(node, self.source) or "",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee_text = _stringify_call_target(node.func, self.source)
        callee_short = callee_text.split(".")[-1] if callee_text else ""
        if isinstance(node.func, ast.Name):
            call_kind = "name"
            receiver_expr = None
        elif isinstance(node.func, ast.Attribute):
            call_kind = "attr"
            receiver_expr = _stringify_expr(node.func.value, self.source)
        else:
            call_kind = "other"
            receiver_expr = None

        lineno = getattr(node, "lineno", -1)
        col_offset = getattr(node, "col_offset", -1)
        end_lineno = getattr(node, "end_lineno", lineno)
        end_col_offset = getattr(node, "end_col_offset", col_offset)

        kwarg_names = [kw.arg for kw in node.keywords if kw.arg is not None]
        has_starargs = any(isinstance(arg, ast.Starred) for arg in node.args)
        has_kwargs = any(kw.arg is None for kw in node.keywords)

        self.callsites.append(
            {
                "call_expr": _stringify_expr(node, self.source) or "",
                "func_expr": _stringify_expr(node.func, self.source) or "",
                "callee_text": callee_text,
                "callee_short": callee_short,
                "call_kind": call_kind,
                "receiver_expr": receiver_expr,
                "stmt_text": _stmt_text(self.source, self.current_stmt, node),
                "stmt_kind": type(self.current_stmt).__name__ if self.current_stmt is not None else "",
                "site_role": self.site_role,
                "loc": {
                    "lineno": lineno,
                    "col_offset": col_offset,
                    "end_lineno": end_lineno,
                    "end_col_offset": end_col_offset,
                },
                "resolved": [],
                "candidate_resolved": [],
                "context": {
                    "loop_depth": self.loop_depth,
                    "cond_depth": self.cond_depth,
                    "comprehension_depth": self.comprehension_depth,
                    "lambda_depth": self.lambda_depth,
                },
                "arg_count": len(node.args),
                "kwarg_names": kwarg_names,
                "has_starargs": has_starargs,
                "has_kwargs": has_kwargs,
                "is_recursive": bool(self.current_callable_name and callee_short == self.current_callable_name),
            }
        )
        self.generic_visit(node)


class _BodyShapeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_branch = False
        self.has_loop = False

    def visit_If(self, node: ast.If) -> None:
        self.has_branch = True
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.has_branch = True
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.has_branch = True
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:  # type: ignore[attr-defined]
        self.has_branch = True
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.has_branch = True
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.has_loop = True
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.has_loop = True
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.has_loop = True
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.has_loop = True
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.has_loop = True
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.has_loop = True
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.has_loop = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _body_shape(stmts: Iterable[ast.stmt]) -> Tuple[bool, bool]:
    v = _BodyShapeVisitor()
    for stmt in stmts:
        v.visit(stmt)
    return v.has_branch, v.has_loop


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _stringify_call_target(func: ast.AST, source: str) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _stringify_call_target(func.value, source)
        return f"{base}.{func.attr}" if base else func.attr
    text = _stringify_expr(func, source)
    return text or "<call>"


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_alias_chain(receiver_expr: Optional[str], alias_map: Dict[str, List[ImportAlias]]) -> Optional[str]:
    if not receiver_expr:
        return None
    parts = receiver_expr.split(".")
    if not parts:
        return None
    first = parts[0]
    aliases = alias_map.get(first, [])
    if not aliases:
        return None
    prefix = aliases[0].target
    if len(parts) == 1:
        return prefix
    return _qualname_join(prefix, *parts[1:])


def _callable_candidates_for_module_attr(module_path: str, method_short: str, callables: Dict[str, dict]) -> List[str]:
    exact = _qualname_join(module_path, method_short)
    return [exact] if exact in callables else []


def _method_candidates_for_class(class_qualname: str, method_short: str, methods_by_class: Dict[str, Dict[str, List[str]]]) -> List[str]:
    return list(methods_by_class.get(class_qualname, {}).get(method_short, []))


def _constructor_candidates_for_class(class_qualname: str, callables: Dict[str, dict]) -> List[str]:
    init_qn = _qualname_join(class_qualname, "__init__")
    return [init_qn] if init_qn in callables else []


def _same_module_candidates(module_name: str, short: str, callable_by_short_name: Dict[str, List[str]]) -> List[str]:
    prefix = module_name + "." if module_name else ""
    return [qn for qn in callable_by_short_name.get(short, []) if qn.startswith(prefix)]


def _resolve_callsite(
    *,
    cs: dict,
    owner_class: Optional[str],
    module_name: str,
    alias_map: Dict[str, List[ImportAlias]],
    callable_by_short_name: Dict[str, List[str]],
    class_by_short_name: Dict[str, List[str]],
    methods_by_class: Dict[str, Dict[str, List[str]]],
    callables: Dict[str, dict],
) -> Tuple[List[str], List[str]]:
    short = cs.get("callee_short") or ""
    call_kind = cs.get("call_kind")
    receiver_expr = cs.get("receiver_expr")

    resolved: List[str] = []
    candidates: List[str] = []

    if not short:
        return resolved, candidates

    if call_kind == "name":
        # direct imported symbol alias: from x import y as z; z(...)
        imported = alias_map.get(short, [])
        explicit: List[str] = []
        for alias in imported:
            if alias.kind != "symbol":
                continue
            if alias.target in callables:
                explicit.append(alias.target)
            if alias.target in class_by_short_name.get(_short_name(alias.target), []) or alias.target in methods_by_class:
                explicit.extend(_constructor_candidates_for_class(alias.target, callables))
                explicit.append(alias.target)
        explicit = _dedupe_keep_order(explicit)
        if len(explicit) == 1 and explicit[0] in callables:
            return explicit, explicit
        if explicit:
            candidates.extend(explicit)

        same_module = _same_module_candidates(module_name, short, callable_by_short_name)
        if len(same_module) == 1:
            return same_module, _dedupe_keep_order(candidates + same_module)
        if same_module:
            candidates.extend(same_module)

        global_funcs = callable_by_short_name.get(short, [])
        if len(global_funcs) == 1:
            return global_funcs, _dedupe_keep_order(candidates + global_funcs)
        if global_funcs:
            candidates.extend(global_funcs)

        classes = class_by_short_name.get(short, [])
        class_candidates: List[str] = []
        for cls in classes:
            ctor = _constructor_candidates_for_class(cls, callables)
            class_candidates.extend(ctor)
            class_candidates.append(cls)
        candidates.extend(class_candidates)
        return [], _dedupe_keep_order(candidates)

    if call_kind == "attr":
        if receiver_expr in {"self", "cls"} and owner_class:
            owner_methods = _method_candidates_for_class(owner_class, short, methods_by_class)
            if len(owner_methods) == 1:
                return owner_methods, owner_methods
            if owner_methods:
                candidates.extend(owner_methods)

        alias_target = _resolve_alias_chain(receiver_expr, alias_map)
        if alias_target:
            if alias_target in methods_by_class:
                cls_methods = _method_candidates_for_class(alias_target, short, methods_by_class)
                if len(cls_methods) == 1:
                    return cls_methods, cls_methods
                candidates.extend(cls_methods)
            module_attr = _callable_candidates_for_module_attr(alias_target, short, callables)
            if len(module_attr) == 1:
                return module_attr, _dedupe_keep_order(candidates + module_attr)
            candidates.extend(module_attr)

        receiver_tail = receiver_expr.split(".")[-1] if isinstance(receiver_expr, str) and receiver_expr else None
        if receiver_tail:
            classes = class_by_short_name.get(receiver_tail, [])
            class_method_candidates: List[str] = []
            for cls in classes:
                class_method_candidates.extend(_method_candidates_for_class(cls, short, methods_by_class))
            if len(class_method_candidates) == 1:
                return class_method_candidates, _dedupe_keep_order(candidates + class_method_candidates)
            candidates.extend(class_method_candidates)

        global_funcs = callable_by_short_name.get(short, [])
        candidates.extend(global_funcs)
        return [], _dedupe_keep_order(candidates)

    # other / dynamic call target
    candidates.extend(callable_by_short_name.get(short, []))
    return [], _dedupe_keep_order(candidates)


# ---------------------------------------------------------------------------
# File / class / callable analysis
# ---------------------------------------------------------------------------


def _analyze_callable(
    node: ast.AST,
    *,
    source: str,
    file_path: str,
    module_name: str,
    owner_class: Optional[str],
    class_ctx: Optional[ClassContext],
) -> Tuple[str, dict]:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    kind = "method" if owner_class else "function"
    qualname = _qualname_join(owner_class or module_name, node.name)

    analyzer = BlockAnalyzer(source, current_callable_name=node.name)
    if class_ctx and owner_class:
        if node.args.args:
            # keep self/cls inside locals for downstream state search convenience
            pass
    # Parameters are also local bindings.
    for p in _signature_info(node, source)["parameters"]:
        analyzer.locals.add(p["name"])

    analyzer.analyze_header_exprs(node.decorator_list, role="decorator", stmt_context=node)
    default_exprs = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
    analyzer.analyze_header_exprs(default_exprs, role="default", stmt_context=node)
    analyzer.analyze_statements(node.body)

    has_branch, has_loop = _body_shape(node.body)

    local_bindings: Dict[str, List[dict]] = {}
    for stmt in node.body:
        for name, recs in _collect_named_bindings_from_stmt(stmt, source).items():
            local_bindings.setdefault(name, []).extend(recs)

    record = {
        "kind": kind,
        "file_path": file_path,
        "owner_class": owner_class,
        "span": _span(node).to_dict(),
        "code": _node_code_with_decorators(node, source),
        "signature": _signature_info(node, source),
        "decorators": [_stringify_expr(d, source) or "" for d in node.decorator_list],
        "decls": {
            "locals": sorted(analyzer.locals),
            "nested_defs": sorted(analyzer.nested_defs),
            "bindings": local_bindings,
        },
        "callsites": analyzer.callsites,
        "has_branch": has_branch,
        "has_loop": has_loop,
    }
    return qualname, record


def _class_shared_context_code(node: ast.ClassDef, source: str) -> str:
    snippets: List[str] = []
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        text = _stmt_text(source, stmt)
        if text:
            snippets.append(text)
    return "\n\n".join(snippets)


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant) and isinstance(stmt.value.value, str)


def _analyze_file(file_path: str, code: str) -> Tuple[dict, Dict[str, dict], Dict[str, dict], Dict[str, List[ImportAlias]]]:
    module_name = _module_name_from_path(file_path)
    parsed = _parse_with_fallback(code)

    file_info = {
        "module_name": module_name,
        "imports": [],
        "top_level_bindings": {},
        "top_level_effect_snippets": [],
        "classes": [],
        "top_level_functions": [],
    }
    if parsed.parse_error:
        file_info["parse_error"] = parsed.parse_error
    if parsed.sanitized:
        file_info["sanitized"] = True

    classes: Dict[str, dict] = {}
    callables: Dict[str, dict] = {}
    alias_map: Dict[str, List[ImportAlias]] = {}

    tree = parsed.tree
    if tree is None:
        return file_info, classes, callables, alias_map

    assert isinstance(tree, ast.Module)
    source = parsed.source

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            record, aliases = _import_record(stmt, source, module_name, file_path)
            file_info["imports"].append(record)
            for alias in aliases:
                alias_map.setdefault(alias.bound_name, []).append(alias)
            continue

        bindings = _collect_named_bindings_from_stmt(stmt, source)
        if bindings:
            for name, recs in bindings.items():
                file_info["top_level_bindings"].setdefault(name, []).extend(recs)
            continue

        if isinstance(stmt, ast.ClassDef):
            class_qualname = _qualname_join(module_name, stmt.name)
            file_info["classes"].append(class_qualname)

            class_bindings: Dict[str, List[dict]] = {}
            methods: List[str] = []
            class_ctx = ClassContext(class_qualname=class_qualname, module_name=module_name, file_path=file_path)

            for class_stmt in stmt.body:
                for name, recs in _collect_named_bindings_from_stmt(class_stmt, source).items():
                    class_bindings.setdefault(name, []).extend(recs)
                if isinstance(class_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qn, record = _analyze_callable(
                        class_stmt,
                        source=source,
                        file_path=file_path,
                        module_name=module_name,
                        owner_class=class_qualname,
                        class_ctx=class_ctx,
                    )
                    methods.append(qn)
                    callables[qn] = record

            classes[class_qualname] = {
                "file_path": file_path,
                "span": _span(stmt).to_dict(),
                "code": _node_code_with_decorators(stmt, source),
                "bases": [_stringify_expr(base, source) or "" for base in stmt.bases],
                "decorators": [_stringify_expr(d, source) or "" for d in stmt.decorator_list],
                "class_bindings": class_bindings,
                "methods": methods,
                "shared_context_code": _class_shared_context_code(stmt, source),
            }
            continue

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qn, record = _analyze_callable(
                stmt,
                source=source,
                file_path=file_path,
                module_name=module_name,
                owner_class=None,
                class_ctx=None,
            )
            file_info["top_level_functions"].append(qn)
            callables[qn] = record
            continue

        if _is_docstring_expr(stmt):
            continue

        file_info["top_level_effect_snippets"].append(
            {
                "stmt_kind": type(stmt).__name__,
                "code": _stmt_text(source, stmt),
                "span": _span(stmt).to_dict(),
            }
        )

    file_info["classes"].sort()
    file_info["top_level_functions"].sort()
    return file_info, classes, callables, alias_map


# ---------------------------------------------------------------------------
# Entry analysis
# ---------------------------------------------------------------------------


def _entry_bindings(source: str, tree: ast.Module) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for stmt in tree.body:
        for name, recs in _collect_named_bindings_from_stmt(stmt, source).items():
            out.setdefault(name, []).extend(recs)
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            # imports also bind names in entry scope.
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    name = alias.asname or alias.name.split(".")[0]
                    out.setdefault(name, []).append(
                        {
                            "name": name,
                            "binding_kind": "import",
                            "rhs_kind": "module_import",
                            "rhs_text": alias.name,
                            "source": _stmt_text(source, stmt),
                            "span": _span(stmt).to_dict(),
                        }
                    )
            else:
                for alias in stmt.names:
                    name = alias.asname or alias.name
                    out.setdefault(name, []).append(
                        {
                            "name": name,
                            "binding_kind": "from_import",
                            "rhs_kind": "symbol_import",
                            "rhs_text": alias.name,
                            "source": _stmt_text(source, stmt),
                            "span": _span(stmt).to_dict(),
                        }
                    )
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = stmt.name
            out.setdefault(name, []).append(
                {
                    "name": name,
                    "binding_kind": type(stmt).__name__.lower(),
                    "rhs_kind": "definition",
                    "rhs_text": name,
                    "source": _stmt_text(source, stmt),
                    "span": _span(stmt).to_dict(),
                }
            )
    return out


def _entry_alias_map(tree: ast.Module, source: str) -> Dict[str, List[ImportAlias]]:
    alias_map: Dict[str, List[ImportAlias]] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _, aliases = _import_record(stmt, source, "__entry__", "__entry__.py")
            for alias in aliases:
                alias_map.setdefault(alias.bound_name, []).append(alias)
    return alias_map


def _analyze_entry(entry_script: str) -> Tuple[dict, Dict[str, List[ImportAlias]]]:
    parsed = _parse_with_fallback(entry_script)
    entry = {
        "code": entry_script,
        "callsites": [],
        "decls": {
            "locals": [],
            "nested_defs": [],
            "bindings": {},
        },
    }
    if parsed.parse_error:
        entry["parse_error"] = parsed.parse_error
    if parsed.sanitized:
        entry["sanitized"] = True
    if parsed.tree is None or not isinstance(parsed.tree, ast.Module):
        return entry, {}

    analyzer = BlockAnalyzer(parsed.source, current_callable_name=None)
    analyzer.analyze_statements(parsed.tree.body)

    entry["callsites"] = analyzer.callsites
    entry["decls"] = {
        "locals": sorted(analyzer.locals),
        "nested_defs": sorted(analyzer.nested_defs),
        "bindings": _entry_bindings(parsed.source, parsed.tree),
    }
    return entry, _entry_alias_map(parsed.tree, parsed.source)


# ---------------------------------------------------------------------------
# Index building and summary
# ---------------------------------------------------------------------------


def _build_indexes(files: Dict[str, dict], classes: Dict[str, dict], callables: Dict[str, dict]) -> dict:
    callable_by_short_name: Dict[str, List[str]] = {}
    class_by_short_name: Dict[str, List[str]] = {}
    methods_by_class: Dict[str, Dict[str, List[str]]] = {}
    bindings_by_name: Dict[str, List[dict]] = {}

    for qn, record in callables.items():
        short = _short_name(qn)
        callable_by_short_name.setdefault(short, []).append(qn)
        owner_class = record.get("owner_class")
        if owner_class:
            methods_by_class.setdefault(owner_class, {}).setdefault(short, []).append(qn)
            mangled = _mangled_private_name(owner_class, short)
            if mangled:
                callable_by_short_name.setdefault(mangled, []).append(qn)
                methods_by_class.setdefault(owner_class, {}).setdefault(mangled, []).append(qn)

    for qn in classes:
        class_by_short_name.setdefault(_short_name(qn), []).append(qn)
        methods_by_class.setdefault(qn, {})

    for file_path, file_info in files.items():
        for name, records in file_info.get("top_level_bindings", {}).items():
            for record in records:
                bindings_by_name.setdefault(name, []).append(
                    {
                        "file_path": file_path,
                        "scope": "module",
                        "owner": file_info.get("module_name"),
                        "binding_kind": record.get("binding_kind"),
                        "rhs_kind": record.get("rhs_kind"),
                        "span": record.get("span"),
                        "source": record.get("source"),
                    }
                )

    for class_qn, class_info in classes.items():
        for name, records in class_info.get("class_bindings", {}).items():
            for record in records:
                bindings_by_name.setdefault(name, []).append(
                    {
                        "file_path": class_info.get("file_path"),
                        "scope": "class",
                        "owner": class_qn,
                        "binding_kind": record.get("binding_kind"),
                        "rhs_kind": record.get("rhs_kind"),
                        "span": record.get("span"),
                        "source": record.get("source"),
                    }
                )

    for mapping in (callable_by_short_name, class_by_short_name):
        for name in list(mapping.keys()):
            mapping[name] = sorted(_dedupe_keep_order(mapping[name]))

    for class_qn, per_class in methods_by_class.items():
        for short, qns in list(per_class.items()):
            per_class[short] = sorted(_dedupe_keep_order(qns))

    for name in list(bindings_by_name.keys()):
        bindings_by_name[name] = bindings_by_name[name]

    return {
        "callable_by_short_name": callable_by_short_name,
        "class_by_short_name": class_by_short_name,
        "methods_by_class": methods_by_class,
        "bindings_by_name": bindings_by_name,
    }


def _resolve_all_callsites(
    *,
    files: Dict[str, dict],
    classes: Dict[str, dict],
    callables: Dict[str, dict],
    indexes: dict,
    file_alias_maps: Dict[str, Dict[str, List[ImportAlias]]],
) -> None:
    callable_by_short_name = indexes["callable_by_short_name"]
    class_by_short_name = indexes["class_by_short_name"]
    methods_by_class = indexes["methods_by_class"]

    for qn, record in callables.items():
        file_path = record["file_path"]
        module_name = files[file_path]["module_name"]
        alias_map = file_alias_maps.get(file_path, {})
        owner_class = record.get("owner_class")
        for cs in record.get("callsites", []):
            resolved, candidates = _resolve_callsite(
                cs=cs,
                owner_class=owner_class,
                module_name=module_name,
                alias_map=alias_map,
                callable_by_short_name=callable_by_short_name,
                class_by_short_name=class_by_short_name,
                methods_by_class=methods_by_class,
                callables=callables,
            )
            cs["resolved"] = resolved
            cs["candidate_resolved"] = [c for c in candidates if c not in resolved]


def _summarize(files: Dict[str, dict], classes: Dict[str, dict], callables: Dict[str, dict], entry: dict) -> dict:
    total_imports = sum(len(f.get("imports", [])) for f in files.values())
    total_bindings = sum(sum(len(v) for v in f.get("top_level_bindings", {}).values()) for f in files.values())
    total_top_level_functions = sum(len(f.get("top_level_functions", [])) for f in files.values())
    total_methods = sum(len(c.get("methods", [])) for c in classes.values())

    per_file = []
    for file_path, file_info in files.items():
        per_file.append(
            {
                "file_path": file_path,
                "num_imports": len(file_info.get("imports", [])),
                "num_top_level_bindings": sum(len(v) for v in file_info.get("top_level_bindings", {}).values()),
                "num_top_level_functions": len(file_info.get("top_level_functions", [])),
                "classes": [
                    {
                        "qualname": class_qn,
                        "num_methods": len(classes.get(class_qn, {}).get("methods", [])),
                    }
                    for class_qn in file_info.get("classes", [])
                ],
            }
        )

    return {
        "num_files": len(files),
        "num_imports": total_imports,
        "num_top_level_bindings": total_bindings,
        "num_top_level_functions": total_top_level_functions,
        "num_classes": len(classes),
        "num_methods": total_methods,
        "num_callables": len(callables),
        "num_entry_callsites": len(entry.get("callsites", [])),
        "files": per_file,
    }



# ---------------------------------------------------------------------------
# Output compaction
# ---------------------------------------------------------------------------


def _compact_binding_records(bindings: Dict[str, List[dict]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for name, records in bindings.items():
        compact_records: List[Any] = []
        for record in records:
            src = record.get("source")
            if isinstance(src, str) and src.strip():
                compact_records.append(src)
        if compact_records:
            out[name] = compact_records
    return out


def _compact_imports(imports: List[dict]) -> List[str]:
    out: List[str] = []
    for record in imports:
        raw = record.get("raw_code")
        if isinstance(raw, str) and raw.strip():
            out.append(raw)
    return out


def _compact_context(context: dict) -> Optional[dict]:
    if not isinstance(context, dict):
        return None
    item = {}
    if context.get("loop_depth"):
        item["loop"] = context["loop_depth"]
    if context.get("cond_depth"):
        item["cond"] = context["cond_depth"]
    if context.get("comprehension_depth"):
        item["comp"] = context["comprehension_depth"]
    if context.get("lambda_depth"):
        item["lambda"] = context["lambda_depth"]
    return item or None


def _compact_loc(loc: dict) -> Optional[List[int]]:
    if not isinstance(loc, dict):
        return None
    return [
        int(loc.get("lineno", -1)),
        int(loc.get("col_offset", -1)),
        int(loc.get("end_lineno", -1)),
        int(loc.get("end_col_offset", -1)),
    ]


def _compact_callsites(callsites: List[dict]) -> List[dict]:
    out: List[dict] = []
    for cs in callsites:
        item = {
            "call_expr": cs.get("call_expr"),
            "stmt": cs.get("stmt_text") or cs.get("call_expr"),
            "callee_short": cs.get("callee_short"),
            "call_kind": cs.get("call_kind"),
        }
        receiver_expr = cs.get("receiver_expr")
        if receiver_expr is not None:
            item["receiver_expr"] = receiver_expr
        resolved = cs.get("resolved") or []
        if resolved:
            item["resolved"] = resolved
        candidates = cs.get("candidate_resolved") or []
        if candidates:
            item["candidate_resolved"] = candidates
        out.append(item)
    return out


def _compact_signature(signature: dict) -> dict:
    item = {"parameters": signature.get("parameters", [])}
    if signature.get("return_annotation") is not None:
        item["return_annotation"] = signature["return_annotation"]
    return item


def _compact_file_info(file_info: dict) -> dict:
    item = {
        "module_name": file_info.get("module_name"),
        "imports": _compact_imports(file_info.get("imports", [])),
        "top_level_bindings": _compact_binding_records(file_info.get("top_level_bindings", {})),
        "classes": file_info.get("classes", []),
        "top_level_functions": file_info.get("top_level_functions", []),
    }
    if file_info.get("parse_error"):
        item["parse_error"] = file_info["parse_error"]
    return item


def _compact_class_info(class_info: dict) -> dict:
    item = {
        "file_path": class_info.get("file_path"),
        "span": class_info.get("span"),
        "code": class_info.get("code", ""),
        "class_bindings": _compact_binding_records(class_info.get("class_bindings", {})),
        "methods": class_info.get("methods", []),
    }
    shared_context_code = class_info.get("shared_context_code")
    if shared_context_code:
        item["shared_context_code"] = shared_context_code
    return item


def _compact_decls(decls: dict) -> dict:
    item = {}
    nested_defs = decls.get("nested_defs") or []
    bindings = decls.get("bindings") or {}
    if nested_defs:
        item["nested_defs"] = nested_defs
    if bindings:
        item["bindings"] = _compact_binding_records(bindings)
    return item


def _compact_callable_info(record: dict) -> dict:
    item = {
        "kind": record.get("kind"),
        "file_path": record.get("file_path"),
        "span": record.get("span"),
        "code": record.get("code", ""),
        "signature": _compact_signature(record.get("signature", {})),
        "callsites": _compact_callsites(record.get("callsites", [])),
    }
    owner_class = record.get("owner_class")
    if owner_class is not None:
        item["owner_class"] = owner_class
    decls = _compact_decls(record.get("decls", {}))
    if decls:
        item["decls"] = decls
    return item


def _compact_indexes(indexes: dict) -> dict:
    out = {
        "callable_by_short_name": indexes.get("callable_by_short_name", {}),
        "class_by_short_name": indexes.get("class_by_short_name", {}),
    }
    methods_by_class = indexes.get("methods_by_class") or {}
    if methods_by_class:
        out["methods_by_class"] = methods_by_class
    return out


def _compact_entry(entry: dict) -> dict:
    item = {
        "callsites": _compact_callsites(entry.get("callsites", [])),
    }
    decls = _compact_decls(entry.get("decls", {}))
    if decls:
        item["decls"] = decls
    if entry.get("parse_error"):
        item["parse_error"] = entry["parse_error"]
    return item


def compact_preprocess(pre: dict) -> dict:
    return {
        "project": pre.get("project", {}),
        "files": {k: _compact_file_info(v) for k, v in (pre.get("files", {}) or {}).items()},
        "classes": {k: _compact_class_info(v) for k, v in (pre.get("classes", {}) or {}).items()},
        "callables": {k: _compact_callable_info(v) for k, v in (pre.get("callables", {}) or {}).items()},
        "indexes": _compact_indexes(pre.get("indexes", {}) or {}),
        "entry": _compact_entry(pre.get("entry", {}) or {}),
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_project(files: Dict[str, str], entry_script: str) -> dict:
    """
    Build a high-availability multi-file static index for downstream route/state
    search/callee localization/slicing.

    Notes:
    - Focuses on reliable static extraction, not dynamic Python semantics.
    - Nested functions are recorded in owner callable decls but are not promoted to
      top-level callables and are not expanded as normal cross-frame targets.
    - If a file cannot be parsed even after basic sanitation, that file still
      appears in the output with a parse_error field and empty structured content.
    """
    files_out: Dict[str, dict] = {}
    classes_out: Dict[str, dict] = {}
    callables_out: Dict[str, dict] = {}
    alias_maps: Dict[str, Dict[str, List[ImportAlias]]] = {}

    for file_path in sorted(files.keys()):
        file_info, class_info, callable_info, alias_map = _analyze_file(file_path, files[file_path])
        files_out[file_path] = file_info
        classes_out.update(class_info)
        callables_out.update(callable_info)
        alias_maps[file_path] = alias_map

    indexes = _build_indexes(files_out, classes_out, callables_out)
    _resolve_all_callsites(
        files=files_out,
        classes=classes_out,
        callables=callables_out,
        indexes=indexes,
        file_alias_maps=alias_maps,
    )

    entry, entry_alias_map = _analyze_entry(entry_script)
    for cs in entry.get("callsites", []):
        resolved, candidates = _resolve_callsite(
            cs=cs,
            owner_class=None,
            module_name="",
            alias_map=entry_alias_map,
            callable_by_short_name=indexes["callable_by_short_name"],
            class_by_short_name=indexes["class_by_short_name"],
            methods_by_class=indexes["methods_by_class"],
            callables=callables_out,
        )
        cs["resolved"] = resolved
        cs["candidate_resolved"] = [c for c in candidates if c not in resolved]

    summary = _summarize(files_out, classes_out, callables_out, entry)

    full_pre = {
        "project": {
            "entry_script": entry_script,
            "files": sorted(files.keys()),
        },
        "files": files_out,
        "classes": classes_out,
        "callables": callables_out,
        "indexes": indexes,
        "entry": entry,
        "summary": summary,
    }
    return compact_preprocess(full_pre)


# ---------------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json
    import re
    from pathlib import Path

    # 原始数据集根目录
    DATASET_ROOT = Path("PyRepoExec")

    # 新的独立输出目录
    OUTPUT_ROOT = Path("pre_process_outputs")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    def flat_name_to_logical_path(filename: str) -> str:
        """
        单目录平铺文件名 -> 逻辑项目路径
        例如:
            a@@b@@c.py -> a/b/c.py
            pkg@@core@@__init__.py -> pkg/core/__init__.py
        """
        return filename.replace("@@", "/")

    def task_sort_key(task_path: Path):
        m = re.match(r"task(\d+)$", task_path.name)
        if m:
            return int(m.group(1))
        return task_path.name

    def load_slicing_files(slicing_dir: Path) -> dict[str, str]:
        files: dict[str, str] = {}

        if not slicing_dir.exists() or not slicing_dir.is_dir():
            return files

        for path in sorted(slicing_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix != ".py":
                continue

            logical_path = flat_name_to_logical_path(path.name)
            files[logical_path] = path.read_text(encoding="utf-8")

        return files

    def process_task(project_dir: Path, task_dir: Path) -> None:
        slicing_dir = task_dir / "slicing"
        entry_file = task_dir / "entry.py"

        if not slicing_dir.exists():
            print(f"[skip] no slicing dir: {task_dir}")
            return
        if not entry_file.exists():
            print(f"[skip] no entry.py: {task_dir}")
            return

        files = load_slicing_files(slicing_dir)
        if not files:
            print(f"[skip] no slice files: {task_dir}")
            return

        entry_script = entry_file.read_text(encoding="utf-8").strip()

        try:
            pre = analyze_project(files, entry_script)
        except Exception as e:
            print(f"[error] analyze failed: {task_dir} -> {e}")
            return

        project_output_dir = OUTPUT_ROOT / project_dir.name
        project_output_dir.mkdir(parents=True, exist_ok=True)

        json_path = project_output_dir / f"{task_dir.name}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pre, f, indent=2, ensure_ascii=False)

        print(f"[ok] {project_dir.name}/{task_dir.name} -> {json_path}")

    project_dirs = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()])

    for project_dir in project_dirs:
        task_dirs = sorted(
            [p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith("task")],
            key=task_sort_key,
        )

        for task_dir in task_dirs:
            process_task(project_dir, task_dir)
