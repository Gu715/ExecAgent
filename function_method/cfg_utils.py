from __future__ import annotations

import ast
import io
import tokenize
from typing import List, Tuple

from cfg_builder import build_cfg_text


def remove_docstrings(code: str) -> str:
    if not code:
        return ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    remove_lines: set[int] = set()

    def mark_docstring(body: list[ast.stmt]) -> None:
        if not body or not isinstance(body[0], ast.Expr):
            return
        value = getattr(body[0], "value", None)
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return
        start = getattr(body[0], "lineno", None)
        end = getattr(body[0], "end_lineno", start)
        if start is None or end is None:
            return
        remove_lines.update(range(start, end + 1))

    mark_docstring(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark_docstring(node.body)

    return "\n".join(
        line for idx, line in enumerate(code.splitlines(), start=1)
        if idx not in remove_lines
    ).strip("\n")


def strip_python_comments(code: str) -> str:
    if not code:
        return ""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        return tokenize.untokenize(tok for tok in tokens if tok.type != tokenize.COMMENT).strip("\n")
    except Exception:
        return "\n".join(
            line for line in code.splitlines()
            if not line.strip().startswith("#")
        ).strip("\n")


def remove_top_level_imports(code: str) -> str:
    if not code:
        return ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    remove_lines: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is not None and end is not None:
                remove_lines.update(range(start, end + 1))

    return "\n".join(
        line for idx, line in enumerate(code.splitlines(), start=1)
        if idx not in remove_lines
    ).strip("\n")


def clean_code_for_cfg(code: str) -> str:
    """Keep the model-facing code intact, but simplify the code used for CFG construction."""
    code = remove_top_level_imports(code)
    code = remove_docstrings(code)
    code = strip_python_comments(code)
    return code.strip("\n")


def _top_level_functions(code: str) -> List[ast.AST]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _source_for_node(code: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(code, node)
    if segment is not None:
        return segment.strip("\n")

    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return code.strip("\n")
    return "\n".join(code.splitlines()[start - 1:end]).strip("\n")


def select_cfg_source(code: str) -> str:
    """
    Select the function body used to build CFG.

    Many benchmark snippets put helpers first and the target function second.
    The original experiment used that heuristic, so this cleaned version keeps it.
    """
    cleaned = clean_code_for_cfg(code)
    functions = _top_level_functions(cleaned)
    if len(functions) >= 2:
        return _source_for_node(cleaned, functions[1])
    if len(functions) == 1:
        return _source_for_node(cleaned, functions[0])
    return cleaned


def build_method_inputs(code: str) -> Tuple[str, str]:
    model_code = str(code or "")
    cfg_source = select_cfg_source(model_code)
    cfg_text = build_cfg_text(cfg_source)
    return model_code, cfg_text

