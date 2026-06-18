from __future__ import annotations
import ast
import json
import os
import pprint
import re
import threading
import traceback
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from pathlib import Path
from json_repair import repair_json
from openai import OpenAI
from cfg_builder import build_cfg_text
from .preprocess import analyze_project
from .prompts import (
    build_completion_state_prompt,
    build_prompt_continue,
    build_prompt_normal,
    build_route_prompt,
    build_route_search_prompt,
    build_state_handoff_prompt,
    build_state_search_prompt,
    render_retrieved_binding_sources,
)
from .static_tools import (
    build_route_items,
    get_class_context,
    get_entry_code,
    get_frame_class_context_code,
    get_frame_class_contexts,
    get_unit_code,
    resolve_candidates,
    search_binding_sources,
    search_callable_sources,
)
from .dataset import iter_dataset_samples
DEFAULT_MAX_ROUNDS = 150
DEFAULT_OUTPUT_DIR = "repo_inference_result"
DEFAULT_MODEL = "YOUR_MODEL_NAME"
DEFAULT_BASE_URL = "YOUR_OPENAI_COMPATIBLE_BASE_URL"
MAX_RESPONSE_TEXT_FOR_NORMAL = 12000
MAX_STEPS_JSON_FOR_NORMAL = 8000
MAX_REPEAT_CALL_SIGNATURE = 20
MAX_FRAME_DEPTH = 60
ROUTE_SEARCH_TOP_K = 3
STATE_SEARCH_TOP_K = 5
MAX_SEARCH_KEYWORDS = 10
@dataclass
class InferenceConfig:
    max_rounds: int = DEFAULT_MAX_ROUNDS
    output_dir: str = DEFAULT_OUTPUT_DIR
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    base_url: Optional[str] = DEFAULT_BASE_URL
    max_retries: int = 5
@dataclass
class AnalysisFrame:
    unit_labels: List[str]
    unit_qualnames: List[str]
    codes: List[str]
    cfgs: List[str]
    frame_context_text: str
    pause_callees: List[str]
    inline_callees: List[str]
    prior_trace_texts: List[str] = field(default_factory=list)
    pause_at: Dict[str, Any] = field(default_factory=dict)
    call_info: Dict[str, Any] = field(default_factory=dict)
    frame_name: str = ""
    route_records: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_routes: List[Dict[str, Any]] = field(default_factory=list)
    class_contexts: List[Dict[str, Any]] = field(default_factory=list)
    class_context_code: str = ""
    def copy(self) -> "AnalysisFrame":
        return AnalysisFrame(
            unit_labels=list(self.unit_labels),
            unit_qualnames=list(self.unit_qualnames),
            codes=list(self.codes),
            cfgs=list(self.cfgs),
            frame_context_text=self.frame_context_text,
            pause_callees=list(self.pause_callees),
            inline_callees=list(self.inline_callees),
            prior_trace_texts=list(self.prior_trace_texts),
            pause_at=dict(self.pause_at),
            call_info=dict(self.call_info),
            frame_name=self.frame_name,
            route_records=[dict(r) for r in self.route_records],
            unresolved_routes=[dict(r) for r in self.unresolved_routes],
            class_contexts=[dict(c) for c in self.class_contexts],
            class_context_code=self.class_context_code,
        )
@dataclass
class RoundResult:
    round_idx: int
    frame_depth: int
    frame_name: str
    prompt_text: str
    raw_response_text: str
    response_json: Dict[str, Any]
    status: str
    call_result: Optional[Dict[str, Any]] = None
@dataclass
class RouteRoundResult:
    round_idx: int
    frame_depth: int
    frame_name: str
    unit_label: str
    unit_qualname: str
    prompt_text: str
    raw_response_text: str
    response_json: Dict[str, Any]
    route_items: List[Dict[str, Any]] = field(default_factory=list)
    route_records: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
@dataclass
class StateRoundResult:
    round_idx: int
    frame_depth: int
    state_kind: str
    caller_frame_name: str
    callee_name: str
    callee_qualname: str
    prompt_text: str
    raw_response_text: str
    response_json: Dict[str, Any]
    status: str = "ok"
    partial_pause: Dict[str, Any] = field(default_factory=dict)
    static_info: Dict[str, Any] = field(default_factory=dict)
@dataclass
class TaskResult:
    sample_id: str
    rounds: List[RoundResult] = field(default_factory=list)
    state_rounds: List[StateRoundResult] = field(default_factory=list)
    route_rounds: List[RouteRoundResult] = field(default_factory=list)
    final_output: Any = None
    success: bool = False
    error_message: str = ""
def _strip_blank_lines(text: str) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines)

def _trim_code_text(text: str) -> str:
    return str(text or "").strip()
def _write_text(path: str, text: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
def _json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _pretty_text(obj: Any) -> str:
    try:
        return pprint.pformat(obj, width=120, sort_dicts=False, compact=False)
    except Exception:
        return repr(obj)

def _indent_text(text: str, prefix: str = "  ") -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return prefix.rstrip()
    return "\n".join(prefix + line for line in lines)
def _tail_qualname(qualname: str) -> str:
    return (qualname or "").split("::")[-1]
def _owner_class_name(pre_process: dict, qualname: str) -> Optional[str]:
    if not isinstance(qualname, str) or not qualname.strip():
        return None

    callables = pre_process.get("callables", {}) or {}
    record = callables.get(qualname)

    if not isinstance(record, dict):
        return None

    owner = record.get("owner_class")
    if not isinstance(owner, str) or not owner.strip():
        return None

    return owner.split(".")[-1]
def _short_name_from_qualname(qualname: str) -> str:
    return qualname.split("::")[-1].split(".")[-1]
class InferenceEngine:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.client = self._init_openai_client()
        self.route_cache_mem: Dict[str, Dict[str, Any]] = {}
        self.current_sample_file_count: int = 0
    def _init_openai_client(self) -> OpenAI:
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未找到API Key，请设置环境变量 OPENAI_API_KEY 或在配置中指定")
        client_kwargs = {"api_key": api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        return OpenAI(**client_kwargs)
    def _strip_response_text(self, text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def call_llm(
        self,
        prompt: str,
        *,
        system_prompt: str = "You are a strict JSON extractor. Output JSON ONLY.",
    ) -> Tuple[Dict[str, Any], str]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        text = self._strip_response_text(content)
        try:
            parsed = repair_json(text, return_objects=True)
            if not isinstance(parsed, dict):
                parsed = {"_raw": text}
            return parsed, text
        except Exception:
            return {"_raw": text}, text

    def call_text_llm(
        self,
        prompt: str,
        *,
        system_prompt: str = "You are a Python program analyst. Follow the requested section style.",
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return self._strip_response_text(content)

    def _extract_json_block(self, text: str, section_names: Sequence[str]) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {}
        names = [re.escape(str(name).strip()) for name in section_names if str(name).strip()]
        patterns: List[str] = []
        if names:
            joined = "|".join(names)
            patterns.append(rf"(?is)\[(?:{joined})\]\s*```json\s*(\{{.*?\}})\s*```")
            patterns.append(rf"(?is)\[(?:{joined})\]\s*(\{{.*\}})\s*$")
        patterns.append(r"(?is)```json\s*(\{.*?\})\s*```")
        patterns.append(r"(?is)(\{.*\})\s*$")
        for pat in patterns:
            matches = re.findall(pat, text)
            if not matches:
                continue
            candidate = matches[-1].strip()
            try:
                parsed = repair_json(candidate, return_objects=True)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _split_text_before_json_section(self, text: str, section_names: Sequence[str]) -> str:
        if not isinstance(text, str):
            return ""
        names = [re.escape(str(name).strip()) for name in section_names if str(name).strip()]
        if names:
            joined = "|".join(names)
            m = re.search(rf"(?is)^(.*?)(?:\n\s*\[(?:{joined})\]\s*```json.*|\n\s*\[(?:{joined})\]\s*\{{.*)$", text)
            if m:
                return m.group(1).strip()
        m = re.search(r"(?is)^(.*?)(?:\n\s*```json.*|\n\s*\{.*)$", text)
        if m:
            return m.group(1).strip()
        return text.strip()

    def _safe_literal_from_text(self, text: str) -> Any:
        s = str(text or "").strip()
        if not s:
            return s
        try:
            return ast.literal_eval(s)
        except Exception:
            low = s.lower()
            if low == "true":
                return True
            if low == "false":
                return False
            if low == "none":
                return None
            try:
                if re.fullmatch(r"[-+]?\d+", s):
                    return int(s)
                if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", s):
                    return float(s)
            except Exception:
                pass
        return s

    def _literal_from_repr_field(self, value: Any, *, type_name: str = "") -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return ""

        should_parse = False
        if type_name and type_name not in {"str", "string"}:
            should_parse = True
        if text in {"True", "False", "None"}:
            should_parse = True
        if re.fullmatch(r"[-+]?\d+", text) or re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", text):
            should_parse = True
        if text[:1] in {"'", '"', "[", "{", "("}:
            should_parse = True

        if should_parse:
            return self._safe_literal_from_text(text)
        return value

    def _normalize_state_tree(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._normalize_state_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_state_tree(v) for v in value]
        if isinstance(value, str):
            return self._literal_from_repr_field(value)
        return value

    def _normalize_callee_value(self, value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") and text.endswith("}"):
                parsed = self._safe_literal_from_text(text)
                if isinstance(parsed, dict):
                    return self._normalize_callee_value(parsed)
            return text
        if isinstance(value, dict):
            for key in ("callee_qualname", "unit_qualname", "callee", "qualname", "unit", "name"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return str(value or "").strip()

    def _guess_type_name(self, value: Any) -> str:
        if value is None:
            return "NoneType"
        return type(value).__name__
    
    def _normalize_small_return(self, ret: Any) -> Dict[str, Any]:
        if isinstance(ret, dict):
            raw_value = ret.get("value_repr", ret.get("value"))
            type_name = str(ret.get("type", "") or "").strip() or "str"
            value = self._literal_from_repr_field(raw_value, type_name=type_name)
            return {"type": type_name, "value": value}
        return {"type": self._guess_type_name(ret), "value": ret}

    def _parse_sim_text_response(self, raw_text: str, current_frame: AnalysisFrame) -> Dict[str, Any]:
        result_json = self._extract_json_block(raw_text, ["Result"])
        trace_text = self._split_text_before_json_section(raw_text, ["Result"])

        if not isinstance(result_json, dict):
            return {"trace_text": trace_text}

        kind = str(result_json.get("kind", "") or "").strip().lower()

        if kind == "pause":
            pause_at_json = result_json.get("pause_at")
            if not isinstance(pause_at_json, dict):
                pause_at_json = {}
            pause_u = str(pause_at_json.get("unit", "") or "").strip()
            pause_b = str(pause_at_json.get("block", "") or "").strip()
            stmt = str(pause_at_json.get("stmt", "") or "").strip()
            callee = self._normalize_callee_value(result_json.get("callee", ""))
            return {
                "trace_text": trace_text,
                "pause_at": {"u": pause_u, "b": pause_b, "stmt": stmt},
                "callee": callee,
            }

        if kind == "complete" and isinstance(result_json.get("final_output_lines"), list):
            return {
                "trace_text": trace_text,
                "final_output_lines": list(result_json.get("final_output_lines") or []),
            }

        if kind == "complete":
            if "return" in result_json:
                return {"trace_text": trace_text, "return": self._normalize_small_return(result_json.get("return"))}
            return {"trace_text": trace_text}

        return {"trace_text": trace_text}

    def _parse_state_text_response(self, raw_text: str) -> Dict[str, Any]:
        parsed = self._extract_json_block(raw_text, ["Child Input"])
        return parsed if isinstance(parsed, dict) else {}

    def _parse_completion_state_response(self, raw_text: str) -> Dict[str, Any]:
        parsed = self._extract_json_block(raw_text, ["State Patch JSON", "State Patch"])
        return parsed if isinstance(parsed, dict) else {}

    def _extract_named_text_section(self, raw_text: str, section_name: str) -> str:
        if not isinstance(raw_text, str) or not raw_text.strip():
            return ""
        name = re.escape(str(section_name or "").strip())
        if not name:
            return ""
        match = re.search(rf"(?is)\[{name}\]\s*(.*?)(?=\n\s*\[[^\]]+\]\s*|\Z)", raw_text)
        if not match:
            return ""
        text = match.group(1).strip()
        fence = re.fullmatch(r"(?is)```(?:text|python)?\s*(.*?)\s*```", text)
        if fence:
            text = fence.group(1).strip()
        elif text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _parse_route_text_response(self, raw_text: str) -> Dict[str, Any]:
        parsed = self._extract_json_block(raw_text, ["Route JSON"])
        return parsed if isinstance(parsed, dict) else {}
    def _route_cache_key(self, unit_qualname: str, function_source: str) -> str:
        return f"{unit_qualname}\n---\n{function_source.strip()}"
    def _live_round_dir(self, sample_id: str, action: str, round_idx: int) -> str:
        sample_dir = os.path.join(self.config.output_dir, sample_id)
        round_dir = os.path.join(sample_dir, f"round{round_idx}_{action}")
        os.makedirs(round_dir, exist_ok=True)
        return round_dir
    def _save_prompt_for_round(self, *, sample_id: str, action: str, round_idx: int, prompt_text: str) -> None:
        round_dir = self._live_round_dir(sample_id, action, round_idx)
        _write_text(os.path.join(round_dir, "prompt.txt"), prompt_text)
    def _flush_task_result(self, task_result: Optional["TaskResult"], *, write_summary: bool = False) -> None:
        if task_result is None:
            return
        self.save_task_result(task_result, self.config.output_dir, write_summary=write_summary)
    def _next_global_round_idx(self, task_result: "TaskResult") -> int:
        return len(task_result.rounds) + len(task_result.state_rounds) + len(task_result.route_rounds) + 1
        
    def _normalize_route_action(self, action: Any) -> str:
        s = str(action or "").strip().lower()
        return s if s in {"inline", "expand", "unresolved"} else ""
    def _normalize_stmt_key(self, stmt: Any) -> str:
        if not isinstance(stmt, str):
            return ""
        return " ".join(stmt.strip().split())
    def _extract_search_keywords(self, parsed: Any, *, max_keywords: int = MAX_SEARCH_KEYWORDS) -> List[str]:
        if not isinstance(parsed, dict):
            return []
        raw = None
        search_obj = parsed.get("search")
        if isinstance(search_obj, dict):
            raw = search_obj.get("keywords", [])
        if raw is None:
            decision = str(parsed.get("decision", "") or "").strip().lower()
            if decision == "search":
                raw = parsed.get("keywords", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        seen = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            kw = " ".join(item.strip().split())
            if not kw or kw in seen:
                continue
            seen.add(kw)
            out.append(kw)
            if len(out) >= max_keywords:
                break
        return out

    def _extract_static_state_keywords(self, code: str, *, max_keywords: int = MAX_SEARCH_KEYWORDS) -> List[str]:
        if not isinstance(code, str) or not code.strip():
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        defined: set[str] = set()
        loaded: List[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                defined.add(node.name)
                for arg in list(node.args.args) + list(node.args.kwonlyargs):
                    defined.add(arg.arg)
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)
                for stmt in node.body:
                    self.visit(stmt)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.visit_FunctionDef(node)  # type: ignore[arg-type]

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                defined.add(node.name)

            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, (ast.Store, ast.Param)):
                    defined.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    loaded.append(node.id)

        Visitor().visit(tree)
        builtins = {
            "None", "True", "False", "len", "list", "dict", "set", "tuple", "str", "int", "float",
            "range", "enumerate", "zip", "sorted", "sum", "min", "max", "isinstance", "getattr",
            "setattr", "Exception", "KeyError", "ValueError",
        }
        out: List[str] = []
        seen = set()
        for name in loaded:
            if name in defined or name in builtins or name in seen:
                continue
            if name.isupper() or name.startswith("_"):
                seen.add(name)
                out.append(name)
                if len(out) >= max_keywords:
                    break
        return out

    def _extract_state_text_keywords(self, text: str, *, max_keywords: int = MAX_SEARCH_KEYWORDS) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        out: List[str] = []
        seen = set()

        def add(value: str) -> None:
            value = str(value or "").strip().strip("'\"`")
            if not value or value in seen:
                return
            seen.add(value)
            out.append(value)

        for match in re.finditer(r"\b[A-Z][A-Z0-9_]{2,}\b", text):
            add(match.group(0))
            if len(out) >= max_keywords:
                return out
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){2,}(?:\.%s)?\b", text):
            add(match.group(0))
            if len(out) >= max_keywords:
                return out
        return out

    def _route_search_prompt(self, *, function_source: str, route_items: List[Dict[str, Any]]) -> str:
        return build_route_search_prompt(function_source=function_source, route_items=route_items)
    def _attach_search_results_to_route_items(
        self,
        route_items: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        qn_to_code = {
            str(r.get("qualname", "") or ""): str(r.get("code", "") or "").strip()
            for r in search_results
            if isinstance(r, dict) and str(r.get("qualname", "") or "").strip() and str(r.get("code", "") or "").strip()
        }
        if not qn_to_code:
            return [dict(it) for it in route_items if isinstance(it, dict)]
        fallback_qns = [qn for qn in qn_to_code.keys()]
        out: List[Dict[str, Any]] = []
        for item in route_items:
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            existing_sources: Dict[str, str] = {}
            raw_existing = new_item.get("callee_sources")
            if isinstance(raw_existing, dict):
                for qn, code in raw_existing.items():
                    qn = str(qn).strip()
                    code = str(code or "").strip()
                    if qn and code:
                        existing_sources[qn] = code
            raw_single = str(new_item.get("callee_source", "") or "").strip()
            resolved = list(new_item.get("resolved", []) or [])
            if raw_single and len(resolved) == 1:
                existing_sources.setdefault(resolved[0], raw_single)
            candidate = list(new_item.get("candidate_resolved", []) or [])
            pool = []
            for qn in resolved + candidate:
                if isinstance(qn, str) and qn and qn not in pool:
                    pool.append(qn)
            matched = {qn: qn_to_code[qn] for qn in pool if qn in qn_to_code}
            merged = dict(existing_sources)
            merged.update(matched)
            if merged:
                if len(resolved) == 1 and resolved[0] in merged:
                    new_item["callee_source"] = merged[resolved[0]]
                new_item["callee_sources"] = merged
            elif not pool and fallback_qns:
                new_item["candidate_resolved"] = list(fallback_qns)
                new_item["callee_sources"] = {qn: qn_to_code[qn] for qn in fallback_qns}
            out.append(new_item)
        return out
    def _attach_available_sources_to_route_items(
        self,
        *,
        pre_process: dict,
        route_items: List[Dict[str, Any]],
        main_code: str,
        dep_sources: Optional[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in route_items:
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            source_map: Dict[str, str] = {}
            resolved = list(new_item.get("resolved", []) or [])
            candidate = list(new_item.get("candidate_resolved", []) or [])
            pool: List[str] = []
            for qn in resolved + candidate:
                if isinstance(qn, str) and qn and qn not in pool:
                    pool.append(qn)
            for qn in pool:
                try:
                    code = _trim_code_text(get_unit_code(pre_process, qn, main_code, dep_sources))
                except Exception:
                    code = ""
                if code:
                    source_map[qn] = code
            if source_map:
                if len(resolved) == 1 and resolved[0] in source_map:
                    new_item["callee_source"] = source_map[resolved[0]]
                else:
                    new_item["callee_sources"] = source_map
            out.append(new_item)
        return out
    def _render_binding_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        return render_retrieved_binding_sources(search_results)
    def _state_search_prompt(
        self,
        *,
        caller_code: str,
        caller_trace_history: Sequence[str] | None,
        pause_at: Mapping[str, Any],
        callee_code: str,
        shared_state: Mapping[str, Any] | None = None,
        class_context_code: str | None = None,
    ) -> str:
        return build_state_search_prompt(
            caller_code=caller_code,
            caller_trace_history=caller_trace_history,
            pause_at=pause_at,
            callee_code=callee_code,
            shared_state=shared_state,
            class_context_code=class_context_code,
        )
    def _canonicalize_callable_target(self, pre_process: dict, qualname: str) -> str:
        if not isinstance(qualname, str) or not qualname.strip():
            return ""
        qualname = qualname.strip()

        callables = pre_process.get("callables", {}) or {}
        classes = pre_process.get("classes", {}) or {}

        if qualname in callables:
            return qualname

        if qualname in classes:
            init_qn = f"{qualname}.__init__"
            if init_qn in callables:
                return init_qn
            # No explicit __init__: keep the class qualname itself so the class
            # definition can still be materialized into the frame.
            return qualname

        return qualname

    def _has_explicit_init(self, pre_process: dict, class_qualname: str) -> bool:
        if not isinstance(class_qualname, str) or not class_qualname.strip():
            return False
        callables = pre_process.get("callables", {}) or {}
        return f"{class_qualname}.__init__" in callables
    def _resolve_route_record_qualname(
        self,
        *,
        pre_process: dict,
        caller_qualname: str,
        route_item: Optional[Dict[str, Any]],
        callee_name: str,
    ) -> str:
        chosen = ""

        if isinstance(route_item, dict):
            resolved = route_item.get("resolved", [])
            if isinstance(resolved, list) and len(resolved) == 1 and isinstance(resolved[0], str):
                chosen = resolved[0]
            else:
                pool: List[str] = []
                for key in ("resolved", "candidate_resolved"):
                    vals = route_item.get(key, [])
                    if isinstance(vals, list):
                        for qn in vals:
                            if isinstance(qn, str) and qn and qn not in pool:
                                pool.append(qn)

                if len(pool) == 1:
                    chosen = pool[0]
                elif callee_name and pool:
                    short = callee_name.split(".")[-1]
                    exact = [qn for qn in pool if _short_name_from_qualname(qn) == short]
                    if len(exact) == 1:
                        chosen = exact[0]
                    else:
                        owner = _owner_class_name(pre_process, caller_qualname)
                        if owner:
                            owner_hits = [qn for qn in pool if owner in qn or f".{owner}." in qn]
                            if len(owner_hits) == 1:
                                chosen = owner_hits[0]
                        if not chosen and pool:
                            chosen = pool[0]

        if not chosen and callee_name:
            cands = resolve_candidates(pre_process, callee_name.split(".")[-1])
            if cands:
                if "." in callee_name:
                    owner = callee_name.split(".")[-2]
                    owner_hits = [qn for qn in cands if _owner_class_name(pre_process, qn) == owner or owner in qn]
                    if len(owner_hits) == 1:
                        chosen = owner_hits[0]
                if not chosen:
                    owner = _owner_class_name(pre_process, caller_qualname)
                    if owner:
                        owner_hits = [qn for qn in cands if _owner_class_name(pre_process, qn) == owner]
                        if len(owner_hits) == 1:
                            chosen = owner_hits[0]
                if not chosen:
                    chosen = cands[0]

        chosen = self._canonicalize_callable_target(pre_process, chosen)
        return chosen
    def _normalize_route_records(
        self,
        *,
        pre_process: dict,
        caller_label: str,
        caller_qualname: str,
        route_items: List[Dict[str, Any]],
        parsed: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        stmt_map: Dict[str, List[Dict[str, Any]]] = {}
        expr_map: Dict[str, List[Dict[str, Any]]] = {}

        def _precision_notes_for_route(
            *,
            route: Dict[str, Any],
            matched_items: List[Dict[str, Any]],
            callee_qualname: str,
        ) -> List[str]:
            chunks: List[str] = [
                str(route.get("callee", "") or ""),
                str(callee_qualname or ""),
                " ".join(str(x or "") for x in route.get("callsites", []) or []),
            ]
            for item in matched_items:
                if not isinstance(item, dict):
                    continue
                chunks.extend([
                    str(item.get("expr", "") or ""),
                    str(item.get("stmt", "") or ""),
                    str(item.get("callee_source", "") or ""),
                ])
                sources = item.get("callee_sources")
                if isinstance(sources, dict):
                    chunks.extend(str(v or "") for v in sources.values())
            text = "\n".join(chunks).lower()
            notes: List[str] = []
            checks = [
                ("hash/digest", ["hash", "digest", "md5", "sha1", "sha256", "hexdigest", "hmac"]),
                ("base64/auth", ["base64", "b64", "authorization", "auth", "basic ", "digest "]),
                ("url/percent-encoding", ["url", "uri", "quote", "unquote", "urlencode", "percent", "query", "fragment"]),
                ("regex/pattern", ["regex", "pattern", "re.", ".sub(", "replace_named", "named_group", "\\z", "?p<"]),
                ("numeric-array", ["numpy", "np.", "array", "linspace", "logspace", "datetime64", "timedelta64"]),
                ("datetime/units", ["datetime", "timedelta", "calendar", "units", "since ", "strftime"]),
                ("encoding/entity", ["encode", "decode", "charset", "encoding", "entity", "escape", "unescape"]),
                ("symbolic/unit-format", ["unitscontainer", "unit", "meter", "second", "kilogram", "sympy", "simplify"]),
            ]
            for label, markers in checks:
                if any(marker in text for marker in markers):
                    notes.append(label)
            return notes

        def _item_short_names(item: Dict[str, Any]) -> List[str]:
            out: List[str] = []
            expr = str(item.get("expr", "") or "").strip()
            if expr:
                head = expr.split("(", 1)[0].strip()
                if head:
                    short = head.split(".")[-1]
                    if short and short not in out:
                        out.append(short)
            raw_target = str(item.get("target", "") or "").strip()
            if raw_target:
                short = raw_target.split(".")[-1]
                if short and short not in out:
                    out.append(short)
            for key in ("resolved", "candidate_resolved"):
                vals = item.get(key, [])
                if isinstance(vals, list):
                    for qn in vals:
                        if isinstance(qn, str) and qn.strip():
                            short = _short_name_from_qualname(qn)
                            if short and short not in out:
                                out.append(short)
            return out

        def _pick_best_matches(callsite_text: str, callee_name: str) -> List[Dict[str, Any]]:
            callsite_key = self._normalize_stmt_key(callsite_text)
            callee_short = callee_name.split(".")[-1] if callee_name else ""
            candidates: List[Dict[str, Any]] = []
            seen_ids = set()
            for item in expr_map.get(callsite_key, []):
                ident = id(item)
                if ident not in seen_ids:
                    seen_ids.add(ident)
                    candidates.append(item)
            for item in stmt_map.get(callsite_key, []):
                ident = id(item)
                if ident not in seen_ids:
                    seen_ids.add(ident)
                    candidates.append(item)
            if not candidates:
                return []
            if not callee_short:
                return candidates[:1]

            scored: List[Tuple[int, Dict[str, Any]]] = []
            for item in candidates:
                score = 0
                expr_key = self._normalize_stmt_key(item.get("expr", ""))
                stmt_key = self._normalize_stmt_key(item.get("stmt", ""))
                if expr_key and expr_key == callsite_key:
                    score += 100
                if stmt_key and stmt_key == callsite_key:
                    score += 10
                expr = str(item.get("expr", "") or "").strip()
                if expr:
                    expr_head = expr.split("(", 1)[0].strip().split(".")[-1]
                    if expr_head == callee_short:
                        score += 80
                    elif callee_short in expr:
                        score += 20
                shorts = _item_short_names(item)
                if callee_short in shorts:
                    score += 50
                scored.append((score, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][0]
            if best <= 0:
                return candidates[:1]
            return [item for score, item in scored if score == best]

        for item in route_items:
            if not isinstance(item, dict):
                continue
            stmt_key = self._normalize_stmt_key(item.get("stmt", ""))
            expr_key = self._normalize_stmt_key(item.get("expr", ""))
            if stmt_key:
                stmt_map.setdefault(stmt_key, []).append(item)
            if expr_key:
                expr_map.setdefault(expr_key, []).append(item)

        routes = parsed.get("routes", []) if isinstance(parsed, dict) else []
        if not isinstance(routes, list):
            routes = []

        out: List[Dict[str, Any]] = []
        seen = set()
        for route in routes:
            if not isinstance(route, dict):
                continue
            action = self._normalize_route_action(route.get("action"))
            if not action:
                continue
            callee_name = str(route.get("callee", "")).strip()
            callsites = route.get("callsites", [])
            if isinstance(callsites, str):
                callsites = [callsites]
            if not isinstance(callsites, list):
                callsites = []

            normalized_callsites: List[str] = []
            matched_items: List[Dict[str, Any]] = []
            pool: List[str] = []
            for callsite in callsites:
                callsite = str(callsite).strip()
                if not callsite:
                    continue
                if callsite not in normalized_callsites:
                    normalized_callsites.append(callsite)
                for matched in _pick_best_matches(callsite, callee_name):
                    if matched not in matched_items:
                        matched_items.append(matched)
                    for key in ("resolved", "candidate_resolved"):
                        vals = matched.get(key, [])
                        if isinstance(vals, list):
                            for qn in vals:
                                if isinstance(qn, str) and qn and qn not in pool:
                                    pool.append(qn)

            if not normalized_callsites:
                continue

            callee_qualname = ""
            if len(pool) == 1:
                callee_qualname = pool[0]
            elif pool:
                short = callee_name.split(".")[-1] if callee_name else ""
                exact = [qn for qn in pool if _short_name_from_qualname(qn) == short]
                if len(exact) == 1:
                    callee_qualname = exact[0]
                elif exact:
                    owner = _owner_class_name(pre_process, caller_qualname)
                    if owner:
                        owner_hits = [qn for qn in exact if owner in qn or f".{owner}." in qn]
                        if len(owner_hits) == 1:
                            callee_qualname = owner_hits[0]
                    if not callee_qualname:
                        callee_qualname = exact[0]
                else:
                    owner = _owner_class_name(pre_process, caller_qualname)
                    if owner:
                        owner_hits = [qn for qn in pool if owner in qn or f".{owner}." in qn]
                        if len(owner_hits) == 1:
                            callee_qualname = owner_hits[0]
                    if not callee_qualname:
                        callee_qualname = pool[0]

            if not callee_qualname:
                callee_qualname = self._resolve_route_record_qualname(
                    pre_process=pre_process,
                    caller_qualname=caller_qualname,
                    route_item=matched_items[0] if matched_items else None,
                    callee_name=callee_name,
                )

            if callee_qualname == caller_qualname and matched_items:
                receivers = [
                    str(item.get("receiver_expr", "") or "").strip()
                    for item in matched_items
                    if isinstance(item, dict)
                ]
                if any(receiver and receiver not in {"self", "cls"} for receiver in receivers):
                    action = "inline"

            record = {
                "unit_label": caller_label,
                "unit_qualname": caller_qualname,
                "callee": callee_name or (_tail_qualname(callee_qualname) if callee_qualname else ""),
                "callee_qualname": callee_qualname,
                "action": action,
                "callsites": normalized_callsites,
            }
            precision_notes = _precision_notes_for_route(
                route=route,
                matched_items=matched_items,
                callee_qualname=callee_qualname,
            )
            if precision_notes:
                record["precision"] = precision_notes

            key = (
                record.get("unit_qualname", ""),
                record.get("callee", ""),
                record.get("callee_qualname", ""),
                record.get("action", ""),
                tuple(self._normalize_stmt_key(x) for x in record.get("callsites", [])),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(record)
        return out

    def _source_looks_like_complex_transform(self, source: str) -> bool:
        text = str(source or "")
        if not text.strip():
            return False
        markers = [
            "re.sub",
            ".sub(",
            ".replace(",
            ".split(",
            ".join(",
            "normalize(",
            "unidecode",
            "html",
            "entity",
            "max_length",
            "truncate",
            "regex",
            "pattern",
        ]
        marker_hits = sum(1 for marker in markers if marker in text)
        control_hits = len(re.findall(r"(?m)^\s*(if|for|while|try|except)\b", text))
        return marker_hits >= 2 or (marker_hits >= 1 and control_hits >= 2)

    def _source_has_substantial_exact_value_flow(self, source: str) -> bool:
        text = str(source or "")
        if not text.strip():
            return False
        control_hits = len(re.findall(r"(?m)^\s*(if|elif|for|while|try|except|with)\b", text))
        call_hits = len(re.findall(r"\b[A-Za-z_][A-Za-z0-9_\.]*\s*\(", text))
        exact_markers = [
            "re.",
            "regex",
            "pattern",
            "escape",
            "unescape",
            "quote",
            "unquote",
            "urlencode",
            "encode",
            "decode",
            "base64",
            "digest",
            "hexdigest",
            "datetime",
            "timedelta",
            "datetime64",
            "timedelta64",
            "np.",
            "numpy",
            "array",
            "units",
            "calendar",
            "since ",
        ]
        marker_hits = sum(1 for marker in exact_markers if marker in text.lower())
        return control_hits >= 3 or call_hits >= 8 or (marker_hits >= 2 and (control_hits >= 1 or call_hits >= 4))

    def _preserve_expand_in_single_file(self, record: Dict[str, Any], pre_process: dict, main_code: str, dep_sources: Optional[Dict[str, str]]) -> bool:
        if record.get("forced_expand_reason"):
            return True
        precision = record.get("precision", [])
        if isinstance(precision, str):
            precision = [precision]
        if not isinstance(precision, list) or not precision:
            return False
        high_risk = {
            "hash/digest",
            "base64/auth",
            "url/percent-encoding",
            "regex/pattern",
            "encoding/entity",
        }
        if not any(str(x) in high_risk for x in precision):
            return False
        qn = str(record.get("callee_qualname", "") or "").strip()
        if not qn:
            return False
        try:
            source = _trim_code_text(get_unit_code(pre_process, qn, main_code, dep_sources))
        except Exception:
            return False
        substantial = self._source_has_substantial_exact_value_flow(source)
        return substantial and len(source.splitlines()) <= 80

    def _add_static_expand_records_for_complex_routes(
        self,
        *,
        route_records: List[Dict[str, Any]],
        route_items: List[Dict[str, Any]],
        caller_label: str,
        caller_qualname: str,
    ) -> List[Dict[str, Any]]:
        if self.current_sample_file_count <= 1:
            return [dict(r) for r in route_records if isinstance(r, dict)]
        out = [dict(r) for r in route_records if isinstance(r, dict)]
        seen = {
            (
                str(r.get("callee_qualname", "") or ""),
                tuple(self._normalize_stmt_key(x) for x in (r.get("callsites", []) or [])),
            )
            for r in out
            if isinstance(r, dict)
        }
        for item in route_items:
            if not isinstance(item, dict):
                continue
            resolved = [x for x in (item.get("resolved", []) or []) if isinstance(x, str) and x.strip()]
            if len(resolved) != 1:
                continue
            qn = resolved[0]
            source = str(item.get("callee_source", "") or "")
            if not self._source_looks_like_complex_transform(source):
                continue
            stmt = str(item.get("stmt", "") or item.get("expr", "") or "").strip()
            if not stmt:
                continue
            key = (qn, (self._normalize_stmt_key(stmt),))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "unit_label": caller_label,
                "unit_qualname": caller_qualname,
                "callee": _tail_qualname(qn),
                "callee_qualname": qn,
                "action": "expand",
                "callsites": [stmt],
                "forced_expand_reason": "complex_transform_source",
            })
        return out

    def _run_route_for_unit(
        self,
        *,
        pre_process: dict,
        unit_label: str,
        unit_qualname: str,
        function_source: str,
        main_code: str,
        dep_sources: Optional[Dict[str, str]],
        task_result: Optional["TaskResult"],
        frame_depth: int,
        frame_name: str,
    ) -> List[Dict[str, Any]]:
        route_items = build_route_items(pre_process, unit_qualname, function_source, dep_sources)
        route_items = self._attach_available_sources_to_route_items(
            pre_process=pre_process,
            route_items=route_items,
            main_code=main_code,
            dep_sources=dep_sources,
        )
        prompt_text = build_route_prompt(function_source=function_source, route_items=route_items)
        route_idx = self._next_global_round_idx(task_result) if task_result is not None else None
        cache_key = self._route_cache_key(unit_qualname, function_source)
        cached = self.route_cache_mem.get(cache_key)
        if cached is not None:
            raw_text = str(cached.get("raw_response_text", "") or "")
            parsed = cached.get("response_json", {})
            if not isinstance(parsed, dict):
                parsed = {}
            normalized = cached.get("route_records", [])
            if not isinstance(normalized, list):
                normalized = []
            cached_items = cached.get("route_items", route_items)
            if not isinstance(cached_items, list):
                cached_items = route_items
            if task_result is not None:
                assert route_idx is not None
                task_result.route_rounds.append(
                    RouteRoundResult(
                        round_idx=route_idx,
                        frame_depth=frame_depth,
                        frame_name=frame_name,
                        unit_label=unit_label,
                        unit_qualname=unit_qualname,
                        prompt_text=prompt_text,
                        raw_response_text=raw_text,
                        response_json=parsed,
                        route_items=[dict(it) for it in cached_items if isinstance(it, dict)],
                        route_records=[dict(r) for r in normalized if isinstance(r, dict)],
                        status="cached",
                    )
                )
                self._flush_task_result(task_result)
            return [dict(r) for r in normalized if isinstance(r, dict)]

        search_results: List[Dict[str, Any]] = []
        search_keywords: List[str] = []
        final_route_items = route_items

        # Route-time search is temporarily disabled.
        # Keep the surrounding variables for easier future re-enable.
        #
        # if route_items:
        #     search_prompt = self._route_search_prompt(function_source=function_source, route_items=route_items)
        #     search_plan, _ = self.call_llm(search_prompt)
        #     search_keywords = self._extract_search_keywords(search_plan)
        #     if search_keywords:
        #         preferred_qualnames: List[str] = []
        #         for item in route_items:
        #             if not isinstance(item, dict):
        #                 continue
        #             for key in ("resolved", "candidate_resolved"):
        #                 vals = item.get(key, [])
        #                 if isinstance(vals, list):
        #                     for qn in vals:
        #                         if isinstance(qn, str) and qn and qn not in preferred_qualnames:
        #                             preferred_qualnames.append(qn)
        #         search_results = search_callable_sources(
        #             pre_process,
        #             search_keywords,
        #             preferred_qualnames=preferred_qualnames,
        #             top_k=ROUTE_SEARCH_TOP_K,
        #         )
        #         final_route_items = self._attach_search_results_to_route_items(route_items, search_results)

        prompt_text = build_route_prompt(function_source=function_source, route_items=final_route_items)
        if task_result is not None and route_idx is not None:
            self._save_prompt_for_round(
                sample_id=task_result.sample_id,
                action="route",
                round_idx=route_idx,
                prompt_text=prompt_text,
            )

        if final_route_items:
            final_raw_text = self.call_text_llm(prompt_text)
            final_parsed = self._parse_route_text_response(final_raw_text)
        else:
            final_raw_text = "[Route Analysis]\nNo user-defined callees need routing."
            final_parsed = {"routes": []}

        normalized = self._normalize_route_records(
            pre_process=pre_process,
            caller_label=unit_label,
            caller_qualname=unit_qualname,
            route_items=final_route_items,
            parsed=final_parsed if isinstance(final_parsed, dict) else {"_raw": final_raw_text},
        )
        normalized = self._add_static_expand_records_for_complex_routes(
            route_records=normalized,
            route_items=final_route_items,
            caller_label=unit_label,
            caller_qualname=unit_qualname,
        )
        for record in normalized:
            if not isinstance(record, dict):
                continue
            qn = str(record.get("callee_qualname", "") or "").strip()
            if not qn:
                continue
            if (
                self.current_sample_file_count <= 1
                and record.get("action") == "expand"
                and not self._preserve_expand_in_single_file(record, pre_process, main_code, dep_sources)
            ):
                record["action"] = "inline"
                record["forced_inline_reason"] = "single_file_prefers_inline"
                continue
            if qn in (pre_process.get("classes", {}) or {}) and not self._has_explicit_init(pre_process, qn):
                record["action"] = "inline"
                record["forced_inline_reason"] = "class_without_explicit_init"

        stored_response_json: Dict[str, Any] = final_parsed if isinstance(final_parsed, dict) else {"_raw": final_raw_text}
        if search_results:
            stored_response_json = {
                **stored_response_json,
                "search": {
                    "keywords": list(search_keywords),
                    "results": deepcopy(search_results),
                },
            }
        self.route_cache_mem[cache_key] = {
            "raw_response_text": final_raw_text,
            "response_json": stored_response_json,
            "route_items": final_route_items,
            "route_records": normalized,
        }
        if task_result is not None:
            assert route_idx is not None
            task_result.route_rounds.append(
                RouteRoundResult(
                    round_idx=route_idx,
                    frame_depth=frame_depth,
                    frame_name=frame_name,
                    unit_label=unit_label,
                    unit_qualname=unit_qualname,
                    prompt_text=prompt_text,
                    raw_response_text=final_raw_text,
                    response_json=stored_response_json,
                    route_items=[dict(it) for it in final_route_items],
                    route_records=[dict(r) for r in normalized],
                    status="ok" if route_items else "empty",
                )
            )
            self._flush_task_result(task_result)
        return normalized
    def _build_routed_frame_plan(
        self,
        *,
        pre_process: dict,
        unit_labels: List[str],
        unit_qualnames: List[str],
        codes: List[str],
        cfgs: List[str],
        main_code: str,
        dep_sources: Optional[Dict[str, str]],
        frame_name: str,
        frame_depth: int,
        task_result: Optional["TaskResult"],
    ) -> Dict[str, Any]:
        out_labels = list(unit_labels)
        out_qualnames = list(unit_qualnames)
        out_codes = list(codes)
        out_cfgs = list(cfgs)

        route_records: List[Dict[str, Any]] = []
        inline_callees: List[str] = []
        pause_callees: List[str] = []
        unresolved_routes: List[Dict[str, Any]] = []

        # already materialized into the current frame
        seen_qn = {qn for qn in out_qualnames if isinstance(qn, str) and qn}
        # already routed once; do not re-route old units
        routed_qn: set[str] = set()

        while True:
            progress = False

            current_units = [
                (unit_label, unit_qualname, code)
                for unit_label, unit_qualname, code in zip(out_labels, out_qualnames, out_codes)
                if isinstance(unit_qualname, str) and unit_qualname and unit_qualname not in routed_qn
            ]
            if not current_units:
                break

            for unit_label, unit_qualname, code in current_units:
                routed_qn.add(unit_qualname)

                new_records = self._run_route_for_unit(
                    pre_process=pre_process,
                    unit_label=unit_label,
                    unit_qualname=unit_qualname,
                    function_source=code,
                    main_code=main_code,
                    dep_sources=dep_sources,
                    task_result=task_result,
                    frame_depth=frame_depth,
                    frame_name=frame_name,
                )
                route_records.extend(new_records)

                for record in new_records:
                    action = str(record.get("action", "") or "").strip()
                    callee_qualname = str(record.get("callee_qualname", "") or "").strip()
                    short = (
                        _short_name_from_qualname(callee_qualname)
                        if callee_qualname
                        else str(record.get("callee", "") or "").split(".")[-1]
                    )

                    if action == "inline":
                        # already in frame -> do not materialize again, avoid recursion / loops
                        if not callee_qualname or callee_qualname in seen_qn:
                            continue

                        try:
                            callee_code = _trim_code_text(
                                get_unit_code(pre_process, callee_qualname, main_code, dep_sources)
                            )
                            callee_cfg = build_cfg_text(
                                callee_code,
                                short or _tail_qualname(callee_qualname),
                            )
                        except Exception:
                            record["action"] = "unresolved"
                            record["fallback_unresolved"] = True
                            record["fallback_unresolved_reason"] = "inline_materialization_failed"
                            unresolved_routes.append(record)
                            continue

                        seen_qn.add(callee_qualname)

                        if short and short not in inline_callees:
                            inline_callees.append(short)

                        out_labels.append(short or _tail_qualname(callee_qualname))
                        out_qualnames.append(callee_qualname)
                        out_codes.append(callee_code)
                        out_cfgs.append(callee_cfg)
                        record["materialized_inline"] = True

                        # newly inlined unit should itself be routed in the next loop
                        progress = True

                    elif action == "expand":
                        if short and short not in pause_callees:
                            pause_callees.append(short)

                    elif action == "unresolved":
                        unresolved_routes.append(record)

            if not progress:
                break

        class_contexts = get_frame_class_contexts(
            pre_process,
            out_qualnames,
            main_code,
            dep_sources=dep_sources,
            include_target_method=False,
        )
        class_context_code = get_frame_class_context_code(
            pre_process,
            out_qualnames,
            main_code,
            dep_sources=dep_sources,
            include_target_method=False,
        )

        return {
            "unit_labels": out_labels,
            "unit_qualnames": out_qualnames,
            "codes": out_codes,
            "cfgs": out_cfgs,
            "inline_callees": inline_callees,
            "pause_callees": pause_callees,
            "route_records": route_records,
            # optional new key if you want to migrate later
            "unresolved_routes": unresolved_routes,
            "class_contexts": class_contexts,
            "class_context_code": class_context_code,
        }
    def _match_expand_route(
        self,
        current_frame: AnalysisFrame,
        callee_name: str,
        pause_at: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        callee_short = callee_name.split(".")[-1] if isinstance(callee_name, str) else ""
        stmt_key = self._normalize_stmt_key((pause_at or {}).get("stmt", ""))
        candidates: List[Dict[str, Any]] = []
        for rec in current_frame.route_records:
            if rec.get("action") != "expand":
                continue
            rec_callee = str(rec.get("callee", "")).strip()
            rec_short = rec_callee.split(".")[-1] if rec_callee else _short_name_from_qualname(str(rec.get("callee_qualname", "")))
            if callee_short and rec_short and callee_short != rec_short:
                continue
            candidates.append(rec)
            rec_callsites = rec.get("callsites", [])
            if isinstance(rec_callsites, str):
                rec_callsites = [rec_callsites]
            if not isinstance(rec_callsites, list):
                rec_callsites = []
            for cs in rec_callsites:
                cs_key = self._normalize_stmt_key(cs)
                if stmt_key and cs_key and stmt_key == cs_key:
                    return rec
        return candidates[0] if len(candidates) == 1 else None
    def _resolve_paused_callee_qualname(
        self,
        current_frame: AnalysisFrame,
        callee_name: str,
        pause_at: Dict[str, Any],
        pre_process: dict,
    ) -> str:
        matched = self._match_expand_route(current_frame, callee_name, pause_at)
        if isinstance(matched, dict):
            qn = str(matched.get("callee_qualname", "")).strip()
            if qn:
                return qn
        short = (callee_name or "").split(".")[-1]
        cands = resolve_candidates(pre_process, short) if short else []
        return cands[0] if cands else ""
    def _get_callee_code_and_cfg(
        self,
        callee_name: str,
        pre_process: dict,
        main_code: str,
        dep_sources: Optional[Dict[str, str]],
        preferred_qualname: str = "",
    ) -> tuple[str, str, str]:
        short_name = callee_name.split(".")[-1]
        qualname = preferred_qualname.strip() if isinstance(preferred_qualname, str) else ""

        if not qualname:
            cands = resolve_candidates(pre_process, short_name)
            if not cands:
                error_code = f"# Function '{callee_name}' not found in codebase\npass\n"
                return error_code, "Error: function not found\n", ""

            if len(cands) > 1 and "." in callee_name:
                class_name = callee_name.split(".")[-2] if callee_name.count(".") >= 1 else callee_name.split(".")[0]
                for cand in cands:
                    if class_name in cand:
                        qualname = cand
                        break

            if not qualname:
                qualname = cands[0]

        qualname = self._canonicalize_callable_target(pre_process, qualname)
        if not qualname:
            error_code = f"# Callee '{callee_name}' resolved to a class without explicit __init__ or to a non-callable target\npass\n"
            return error_code, "Error: non-callable target\n", ""

        try:
            code = get_unit_code(pre_process, qualname, main_code, dep_sources)
        except Exception as e:
            error_code = f"# Failed to get code for '{callee_name}': {e}\npass\n"
            return error_code, "Error: failed to get code\n", qualname

        code_stripped = _trim_code_text(code)
        cfg = build_cfg_text(code_stripped, short_name or _short_name_from_qualname(qualname))
        return code_stripped, cfg, qualname
    
    def _normalize_state_packet(
        self,
        *,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        packet: Dict[str, Any] = {"self": {}, "args": {}, "shared": {}}
        if not isinstance(parsed, dict):
            return packet
        for key in ("self", "args", "shared"):
            value = parsed.get(key)
            if isinstance(value, dict):
                packet[key] = deepcopy(value)
        return packet
    def _state_packet_is_valid(self, packet: Dict[str, Any]) -> bool:
        if not isinstance(packet, dict):
            return False
        for key in ("self", "args", "shared"):
            value = packet.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                return False
        return True
    def _run_state_handoff_round(
        self,
        *,
        current_frame: AnalysisFrame,
        pause_at: Dict[str, Any],
        partial_call: Dict[str, Any],
        partial_resume_state: Dict[str, Any],
        callee_name: str,
        callee_qualname: str,
        callee_code: str,
        pre_process: dict,
        main_code: str,
        dep_sources: Optional[Dict[str, str]],
        state_table: Dict[str, Any],
        task_result: Optional["TaskResult"],
        frame_depth: int,
    ) -> str:
        callee_class_ctx = get_class_context(
            pre_process,
            callee_qualname,
            main_code,
            dep_sources=dep_sources,
            include_class_statements=True,
            include_init=True,
            include_dunder_methods=True,
            include_target_method=False,
        )
        class_context_code = ""
        if isinstance(callee_class_ctx, dict):
            class_context_code = str(callee_class_ctx.get("shared_context_code", "") or "").strip()
        caller_code_parts: List[str] = []
        for label, code in zip(current_frame.unit_labels, current_frame.codes):
            caller_code_parts.extend([
                f"# Unit: {label}",
                (code or "").strip(),
                "",
            ])
        caller_code = "\n".join(caller_code_parts).strip()
        shared_state = deepcopy(state_table or {})

        search_prompt = self._state_search_prompt(
            caller_code=caller_code,
            caller_trace_history=current_frame.prior_trace_texts or [],
            pause_at=deepcopy(pause_at),
            callee_code=callee_code,
            shared_state=shared_state,
            class_context_code=class_context_code,
        )
        search_plan, _ = self.call_llm(search_prompt)
        search_keywords: List[str] = []
        static_keyword_sources = [
            self._extract_static_state_keywords(callee_code),
            self._extract_state_text_keywords(caller_code),
            self._extract_state_text_keywords(current_frame.frame_context_text),
        ]
        for keywords in static_keyword_sources:
            for keyword in keywords:
                if keyword not in search_keywords:
                    search_keywords.append(keyword)
                if len(search_keywords) >= MAX_SEARCH_KEYWORDS:
                    break
            if len(search_keywords) >= MAX_SEARCH_KEYWORDS:
                break
        for keyword in self._extract_search_keywords(search_plan):
            if keyword not in search_keywords:
                search_keywords.append(keyword)
            if len(search_keywords) >= MAX_SEARCH_KEYWORDS:
                break
        search_results: List[Dict[str, Any]] = []
        if search_keywords:
            search_results = search_binding_sources(pre_process, search_keywords, top_k=STATE_SEARCH_TOP_K)

        prompt_text = build_state_handoff_prompt(
            caller_code=caller_code,
            caller_steps=current_frame.prior_trace_texts or [],
            pause_at=deepcopy(pause_at),
            callee_code=callee_code,
            shared_state=shared_state,
            class_context_code=class_context_code,
            retrieved_binding_sources_text=self._render_binding_search_results(search_results),
        )
        state_idx = self._next_global_round_idx(task_result) if task_result is not None else None
        if task_result is not None and state_idx is not None:
            self._save_prompt_for_round(
                sample_id=task_result.sample_id,
                action="state",
                round_idx=state_idx,
                prompt_text=prompt_text,
            )
        raw_text = self.call_text_llm(prompt_text)
        handoff_text = str(raw_text or "").strip()
        status = "ok" if handoff_text else "empty"
        if task_result is not None:
            assert state_idx is not None
            task_result.state_rounds.append(
                StateRoundResult(
                    round_idx=state_idx,
                    frame_depth=frame_depth,
                    state_kind="handoff",
                    caller_frame_name=current_frame.frame_name,
                    callee_name=callee_name,
                    callee_qualname=callee_qualname,
                    prompt_text=prompt_text,
                    raw_response_text=handoff_text,
                    response_json={},
                    status=status,
                    partial_pause={
                        "pause_at": deepcopy(pause_at),
                        "call": deepcopy(partial_call),
                    },
                    static_info={
                        "shared_state": deepcopy(shared_state),
                        "search": {
                            "keywords": list(search_keywords),
                            "results": deepcopy(search_results),
                        },
                    },
                )
            )
            self._flush_task_result(task_result)
        return handoff_text

    def _normalize_completion_state_patch(self, parsed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Any] = {"writes": {}, "self": {}, "return_binding": {}}
        if not isinstance(parsed, dict):
            return out
        for key in ("writes", "self", "return_binding"):
            value = parsed.get(key)
            if isinstance(value, dict):
                out[key] = self._normalize_state_tree(value)
        return out

    def _merge_completion_patch_into_result(
        self,
        completion_result: Dict[str, Any],
        patch: Dict[str, Dict[str, Any]],
        *,
        caller_frame: Optional[AnalysisFrame] = None,
    ) -> Dict[str, Any]:
        new_result = deepcopy(completion_result if isinstance(completion_result, dict) else {})
        writes = new_result.get("writes")
        if not isinstance(writes, dict):
            writes = {}
        else:
            writes = deepcopy(writes)

        patch_writes = patch.get("writes", {}) if isinstance(patch, dict) else {}
        if isinstance(patch_writes, dict):
            self._merge_state_mapping_missing_only(writes, patch_writes)

        patch_self = patch.get("self", {}) if isinstance(patch, dict) else {}
        if isinstance(patch_self, dict) and patch_self:
            existing_self = writes.get("self", {})
            if isinstance(existing_self, dict):
                merged_self = deepcopy(existing_self)
                self._merge_state_mapping_missing_only(merged_self, patch_self)
                writes["self"] = merged_self
            else:
                writes["self"] = deepcopy(patch_self)

        return_binding = patch.get("return_binding", {}) if isinstance(patch, dict) else {}
        if isinstance(return_binding, dict) and self._caller_pause_has_nontrivial_return_target(caller_frame):
            self._merge_state_mapping_missing_only(writes, return_binding)

        new_result["writes"] = writes
        runtime_state_text = str(patch.get("runtime_state_text", "") or "").strip() if isinstance(patch, dict) else ""
        if runtime_state_text:
            new_result["runtime_state_text"] = runtime_state_text
        return new_result

    def _run_completion_state_round(
        self,
        *,
        current_frame: AnalysisFrame,
        response_json: Dict[str, Any],
        completion_result: Dict[str, Any],
        pre_process: dict,
        state_table: Dict[str, Any],
        caller_frame: Optional[AnalysisFrame],
        task_result: Optional["TaskResult"],
        frame_depth: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        shared_state = deepcopy(state_table or {})
        caller_pause_at = caller_frame.pause_at if caller_frame is not None and isinstance(caller_frame.pause_at, dict) else {}
        caller_callee_name = ""
        if caller_frame is not None and isinstance(caller_frame.call_info, dict):
            caller_callee_name = str(caller_frame.call_info.get("callee", "") or "")
        prompt_text = build_completion_state_prompt(
            unit_labels=current_frame.unit_labels,
            codes=current_frame.codes,
            frame_name=current_frame.frame_name,
            frame_trace_text=str(response_json.get("trace_text", "") or ""),
            completion_result=completion_result if isinstance(completion_result, dict) else {},
            shared_state=shared_state,
            caller_pause_at=caller_pause_at,
            caller_callee_name=caller_callee_name,
            class_context_code=current_frame.class_context_code,
        )
        state_idx = (self._next_global_round_idx(task_result) + 1) if task_result is not None else None
        if task_result is not None and state_idx is not None:
            self._save_prompt_for_round(
                sample_id=task_result.sample_id,
                action="state",
                round_idx=state_idx,
                prompt_text=prompt_text,
            )
        raw_text = self.call_text_llm(prompt_text)
        parsed = self._parse_completion_state_response(raw_text)
        patch = self._normalize_completion_state_patch(parsed)
        runtime_state_text = self._extract_named_text_section(raw_text, "Updated Shared Runtime State")
        if runtime_state_text:
            patch["runtime_state_text"] = runtime_state_text
        patched_result = self._merge_completion_patch_into_result(completion_result, patch, caller_frame=caller_frame)
        if task_result is not None:
            assert state_idx is not None
            task_result.state_rounds.append(
                StateRoundResult(
                    round_idx=state_idx,
                    frame_depth=frame_depth,
                    state_kind="completion",
                    caller_frame_name=caller_frame.frame_name if caller_frame is not None else "",
                    callee_name=current_frame.frame_name,
                    callee_qualname=current_frame.unit_qualnames[0] if current_frame.unit_qualnames else "",
                    prompt_text=prompt_text,
                    raw_response_text=str(raw_text or "").strip(),
                    response_json={"patch": patch, "completion_result": patched_result},
                    status="ok" if parsed else "empty",
                    partial_pause={
                        "caller_pause_at": deepcopy(caller_pause_at),
                        "completed_frame": current_frame.frame_name,
                    },
                    static_info={"shared_state": shared_state},
                )
            )
            self._flush_task_result(task_result)
        return patched_result, patch

    def _caller_pause_has_nontrivial_return_target(self, caller_frame: Optional[AnalysisFrame]) -> bool:
        if caller_frame is None or not isinstance(caller_frame.pause_at, dict):
            return False
        stmt = str(caller_frame.pause_at.get("stmt", "") or "").strip()
        if not stmt:
            return False
        try:
            tree = ast.parse(stmt)
        except SyntaxError:
            return False
        if not tree.body:
            return False
        node = tree.body[0]
        targets: List[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, (ast.Attribute, ast.Subscript, ast.Tuple, ast.List)):
                return True
        return False

    def _caller_return_target_needs_state_agent(self, caller_frame: Optional[AnalysisFrame], callee_name: str = "") -> bool:
        if caller_frame is None or not isinstance(caller_frame.pause_at, dict):
            return False
        stmt = str(caller_frame.pause_at.get("stmt", "") or "").strip()
        if not stmt:
            return False
        try:
            tree = ast.parse(stmt)
        except SyntaxError:
            return self._caller_pause_has_nontrivial_return_target(caller_frame)
        if not tree.body:
            return False
        node = tree.body[0]
        targets: List[ast.AST] = []
        value_node: Optional[ast.AST] = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            return self._caller_pause_has_nontrivial_return_target(caller_frame)
        if not targets or not isinstance(value_node, ast.Call) or not self._call_matches_callee(value_node, callee_name):
            return self._caller_pause_has_nontrivial_return_target(caller_frame)
        return any(not self._target_is_engine_bindable(t) for t in targets)

    def _target_is_engine_bindable(self, target: ast.AST) -> bool:
        if isinstance(target, (ast.Name, ast.Attribute, ast.Subscript)):
            return bool(self._state_path_for_target(target))
        if isinstance(target, (ast.Tuple, ast.List)):
            return all(self._target_is_engine_bindable(t) for t in target.elts)
        return False

    def _trace_mentions_state_mutation(self, trace_text: str) -> bool:
        text = str(trace_text or "")
        patterns = [
            ".append(",
            ".extend(",
            ".update(",
            ".add(",
            ".remove(",
            ".pop(",
            ".clear(",
            "__setitem__",
        ]
        if any(p in text for p in patterns):
            return True
        assignment_patterns = [
            r"\b(?:self|cls)\.[A-Za-z_][A-Za-z0-9_\.]*\s*(?:=|\+=|-=|\*=|/=|//=|%=|\*\*=)",
            r"\b[A-Za-z_][A-Za-z0-9_\.]*\s*\[[^\]]+\]\s*(?:=|\+=|-=|\*=|/=|//=|%=|\*\*=)",
            r"\b[A-Za-z_][A-Za-z0-9_\.]*\.[A-Za-z_][A-Za-z0-9_\.]*\s*(?:=|\+=|-=|\*=|/=|//=|%=|\*\*=)",
        ]
        if any(re.search(pattern, text) is not None for pattern in assignment_patterns):
            return True
        durable_words = (
            "cache",
            "caches",
            "cached",
            "store",
            "stores",
            "stored",
            "assign",
            "assigns",
            "assigned",
            "mutate",
            "mutates",
            "updated",
            "writes",
        )
        lower = text.lower()
        if any(word in lower for word in durable_words) and ("[" in text or "." in text or "_" in text):
            return True
        return False

    def _should_run_completion_state_agent(
        self,
        *,
        current_frame: AnalysisFrame,
        response_json: Dict[str, Any],
        completion_result: Dict[str, Any],
        pre_process: dict,
        state_table: Dict[str, Any],
        caller_frame: Optional[AnalysisFrame],
        task_result: Optional["TaskResult"],
    ) -> bool:
        if task_result is not None and self._next_global_round_idx(task_result) + 1 > self.config.max_rounds:
            return False
        callee_name = current_frame.frame_name
        if self._caller_return_target_needs_state_agent(caller_frame, callee_name=callee_name):
            return True
        if self._trace_mentions_state_mutation(str(response_json.get("trace_text", "") or "")):
            return True
        if isinstance(current_frame.call_info, dict) and current_frame.call_info.get("full_self_state"):
            trace = str(response_json.get("trace_text", "") or "")
            return self._trace_mentions_state_mutation(trace)
        return False

    def _empty_completion_state_patch(self) -> Dict[str, Dict[str, Any]]:
        return {"writes": {}, "self": {}, "return_binding": {}}

    def _response_kind(self, response_json: Dict[str, Any], current_frame: Optional[AnalysisFrame] = None) -> str:
        if not isinstance(response_json, dict):
            return "invalid"
        if self._solved_output_mode_for_frame(current_frame) == "final_output_lines":
            if isinstance(response_json.get("final_output_lines"), list):
                return "entry_solved"
        if isinstance(response_json.get("return"), dict):
            return "solved"
        pause_at = response_json.get("pause_at")
        callee = response_json.get("callee")
        callee_text = self._normalize_callee_value(callee)
        if isinstance(pause_at, dict) and pause_at.get("stmt") and callee_text:
            return "paused"
        return "invalid"
    def _sanitize_response(self, response_json: Any, current_frame: Optional[AnalysisFrame] = None) -> Dict[str, Any]:
        if not isinstance(response_json, dict):
            return {}
        out = dict(response_json)
        if "callee" in out:
            out["callee"] = self._normalize_callee_value(out.get("callee"))
        return out
    def _safe_json_loads(self, text: str) -> Any:
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    def _merge_dict_like(self, dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in src.items():
            if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                self._merge_dict_like(dst[k], v)
            else:
                dst[k] = deepcopy(v)
        return dst

    def _state_path_parts_from_key(self, key: Any) -> List[str]:
        if not isinstance(key, str):
            return []
        text = key.strip()
        if not text or "." not in text:
            return []
        parts = [p.strip() for p in text.split(".") if p.strip()]
        if len(parts) < 2:
            return []
        if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts):
            return []
        return parts

    def _state_structural_path_from_key(self, key: Any) -> List[str]:
        if not isinstance(key, str):
            return []
        text = key.strip()
        if not text:
            return []
        subscript_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(.+)\]", text, flags=re.S)
        if subscript_match:
            base = subscript_match.group(1)
            index_text = subscript_match.group(2).strip()
            index_value = self._safe_literal_from_text(index_text)
            if isinstance(index_value, str):
                reparsed = self._safe_literal_from_text(index_value)
                if not isinstance(reparsed, str):
                    index_value = reparsed
            if isinstance(index_value, (tuple, int, float, bool)) or index_value is None:
                index_key = repr(index_value)
            elif isinstance(index_value, str):
                index_key = index_value
            else:
                index_key = str(index_value)
            if index_key:
                return [base, index_key]
        return self._state_path_parts_from_key(key)

    def _merge_state_mapping(self, dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in src.items():
            parts = self._state_structural_path_from_key(k)
            if parts:
                cur = dst
                for part in parts[:-1]:
                    nxt = cur.get(part)
                    if not isinstance(nxt, dict):
                        nxt = {}
                        cur[part] = nxt
                    cur = nxt
                leaf = parts[-1]
                if leaf in cur and isinstance(cur[leaf], dict) and isinstance(v, dict):
                    self._merge_state_mapping(cur[leaf], v)
                else:
                    cur[leaf] = deepcopy(v)
                continue
            if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                self._merge_state_mapping(dst[k], v)
            else:
                dst[k] = deepcopy(v)
        return dst

    def _merge_state_mapping_missing_only(self, dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in src.items():
            parts = self._state_structural_path_from_key(k)
            if parts:
                cur: Optional[Dict[str, Any]] = dst
                for part in parts[:-1]:
                    if cur is None:
                        break
                    nxt = cur.get(part)
                    if nxt is None:
                        nxt = {}
                        cur[part] = nxt
                    if not isinstance(nxt, dict):
                        cur = None
                        break
                    cur = nxt
                if cur is None:
                    continue
                leaf = parts[-1]
                if leaf not in cur:
                    cur[leaf] = deepcopy(v)
                elif isinstance(cur.get(leaf), dict) and isinstance(v, dict):
                    self._merge_state_mapping_missing_only(cur[leaf], v)
                continue
            if k not in dst:
                dst[k] = deepcopy(v)
                continue
            if isinstance(dst.get(k), dict) and isinstance(v, dict):
                self._merge_state_mapping_missing_only(dst[k], v)
        return dst

    def _collect_full_self_state(
        self,
        *,
        current_frame: AnalysisFrame,
        call_info: Dict[str, Any],
        state_table: Dict[str, Any],
        callee_qualname: str,
        pre_process: dict,
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}

        owner = (
            _owner_class_name(pre_process, callee_qualname)
            or _owner_class_name(
                pre_process,
                current_frame.unit_qualnames[0] if current_frame.unit_qualnames else "",
            )
        )

        candidates: List[Any] = []
        if isinstance(call_info, dict):
            candidates.append(call_info.get("full_self_state"))
        if isinstance(current_frame.call_info, dict):
            candidates.append(current_frame.call_info.get("full_self_state"))
        if isinstance(state_table, dict):
            candidates.append(state_table.get("__last_full_self_state"))
            by_class = state_table.get("__full_self_by_class", {})
            if isinstance(by_class, dict) and owner:
                candidates.append(by_class.get(owner))
            candidates.append(state_table.get("self"))
            objects = state_table.get("objects", {})
            if isinstance(objects, dict) and owner:
                obj_row = objects.get(owner)
                if isinstance(obj_row, dict):
                    candidates.append(obj_row)

        for cand in candidates:
            if isinstance(cand, dict) and cand:
                self._merge_dict_like(merged, cand)
        return merged

    def _render_runtime_context_text(
        self,
        *,
        current_frame: AnalysisFrame,
        call_info: Dict[str, Any],
        state_table: Dict[str, Any],
        callee_name: str,
        callee_qualname: str,
        pre_process: dict,
    ) -> str:
        parts: List[str] = []

        inherited_context = str(current_frame.frame_context_text or "").strip()
        if inherited_context:
            parts.extend([
                "[Inherited Parent Frame Context]",
                inherited_context,
                "",
            ])

        full_self_state = self._collect_full_self_state(
            current_frame=current_frame,
            call_info=call_info if isinstance(call_info, dict) else {},
            state_table=state_table if isinstance(state_table, dict) else {},
            callee_qualname=callee_qualname,
            pre_process=pre_process,
        )
        if full_self_state:
            parts.extend([
                "[Full Self State]",
                "Preserve the entire receiver state across this call, including fields not explicitly mentioned later.",
                _pretty_text(full_self_state),
                "",
            ])

        pause_at = (call_info or {}).get("pause_at", {}) if isinstance(call_info, dict) else {}
        if isinstance(pause_at, dict) and pause_at:
            stmt = str(pause_at.get("stmt", "") or "").strip()
            unit = str(pause_at.get("u", "") or "").strip()
            block = str(pause_at.get("b", "") or "").strip()
            pause_lines: List[str] = []
            if unit:
                pause_lines.append(f"unit: {unit}")
            if block:
                pause_lines.append(f"block: {block}")
            if stmt:
                pause_lines.append(f"stmt: {stmt}")
            if pause_lines:
                parts.extend([
                    "[Paused Callsite]",
                    "\n".join(pause_lines),
                    "",
                ])

        handoff_text = str((call_info or {}).get("state_handoff_text", "") or "").strip() if isinstance(call_info, dict) else ""
        if handoff_text:
            parts.extend([
                "[State Agent Handoff]",
                handoff_text,
                "",
            ])

        if isinstance(state_table, dict) and state_table:
            runtime_state_text = str(state_table.get("__runtime_state_text", "") or "").strip()
            if runtime_state_text:
                parts.extend([
                    "[Shared Runtime State Ledger]",
                    runtime_state_text,
                    "",
                ])
            visible_state = {k: deepcopy(v) for k, v in state_table.items() if not str(k).startswith("__")}
            if visible_state:
                parts.extend([
                    "[Shared Runtime Snapshot]",
                    _pretty_text(visible_state),
                    "",
                ])

        parts.extend([
            "[Callee Startup Rule]",
            f"Start `{callee_name}` from the recovered runtime context above.",
            "Do not drop receiver fields just because the state-agent handoff only mentions a subset of them.",
            "Prefer the inherited full self state over any partial self summary.",
        ])

        return "\n".join(parts).strip()

    def _prepare_callee_frame_context(
        self,
        current_frame: AnalysisFrame,
        callee_name: str,
        call_info: Dict[str, Any],
        *,
        pre_process: dict,
        callee_qualname: str,
        state_table: Dict[str, Any],
    ) -> str:
        return self._render_runtime_context_text(
            current_frame=current_frame,
            call_info=call_info if isinstance(call_info, dict) else {},
            state_table=state_table if isinstance(state_table, dict) else {},
            callee_name=callee_name,
            callee_qualname=callee_qualname,
            pre_process=pre_process,
        )

    def _prepare_resume_frame_context(
        self,
        caller_frame: AnalysisFrame,
        state_table: Dict[str, Any],
        completion_result: Dict[str, Any],
        state_agent_patch: Dict[str, Dict[str, Any]],
    ) -> str:
        parts: List[str] = []
        inherited_context = str(caller_frame.frame_context_text or "").strip()
        for marker in ("[Latest Callee Return]", "[Latest State Agent Patch]", "[Updated Shared Runtime State]", "[Updated Shared Runtime Snapshot]"):
            idx = inherited_context.find(marker)
            if idx >= 0:
                inherited_context = inherited_context[:idx].strip()
                break
        if inherited_context:
            parts.extend([inherited_context, ""])

        ret = completion_result.get("return") if isinstance(completion_result, dict) else {}
        if isinstance(ret, dict) and ret:
            parts.extend([
                "[Latest Callee Return]",
                _pretty_text(ret),
                "",
            ])

        if isinstance(state_agent_patch, dict) and any(
            isinstance(state_agent_patch.get(k), dict) and state_agent_patch.get(k)
            for k in ("writes", "self", "return_binding")
        ):
            parts.extend([
                "[Latest State Agent Patch]",
                _pretty_text(state_agent_patch),
                "",
            ])

        runtime_state_text = ""
        if isinstance(completion_result, dict):
            runtime_state_text = str(completion_result.get("runtime_state_text", "") or "").strip()
        if not runtime_state_text and isinstance(state_table, dict):
            runtime_state_text = str(state_table.get("__runtime_state_text", "") or "").strip()
        if runtime_state_text:
            parts.extend([
                "[Updated Shared Runtime State]",
                runtime_state_text,
                "",
            ])

        if isinstance(state_table, dict) and state_table:
            visible_state = {k: deepcopy(v) for k, v in state_table.items() if not str(k).startswith("__")}
            if visible_state:
                parts.extend([
                    "[Updated Shared Runtime Snapshot]",
                    _pretty_text(visible_state),
                    "",
                ])

        return "\n".join(parts).strip()

    def _frame_contains_entry(self, current_frame: Optional[AnalysisFrame]) -> bool:
        if current_frame is None:
            return False
        for qn in current_frame.unit_qualnames or []:
            if isinstance(qn, str) and qn.strip() == "<entry>":
                return True
        return False
    def _solved_output_mode_for_frame(self, current_frame: Optional[AnalysisFrame]) -> str:
        return "final_output_lines" if self._frame_contains_entry(current_frame) else "return_only"
    def _init_light_state_table(
        self,
        pre_process: dict,
        unit_qualnames: List[str],
        unit_labels: List[str],
    ) -> Dict[str, Any]:
        return {
            "__full_self_by_class": {},
            "__last_full_self_state": {},
            "__last_state_handoff_text": "",
        }
    def _get_frame_class_name(self, current_frame: AnalysisFrame, pre_process: dict) -> str:
        qualname = current_frame.unit_qualnames[0] if current_frame.unit_qualnames else ""
        return _owner_class_name(pre_process, qualname) or ""
    def _collect_state_type_conflicts(
        self,
        dst: Any,
        patch: Any,
        *,
        prefix: str = "",
    ) -> List[str]:
        conflicts: List[str] = []
        if not isinstance(dst, dict) or not isinstance(patch, dict):
            return conflicts
        for k, v in patch.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if k not in dst:
                continue
            old = dst.get(k)
            old_is_dict = isinstance(old, dict)
            new_is_dict = isinstance(v, dict)
            if old_is_dict and not new_is_dict:
                conflicts.append(f"{path}: dict -> {type(v).__name__}")
                continue
            if not old_is_dict and new_is_dict:
                conflicts.append(f"{path}: {type(old).__name__} -> dict")
                continue
            if old_is_dict and new_is_dict:
                conflicts.extend(self._collect_state_type_conflicts(old, v, prefix=path))
        return conflicts
    def _validate_completion_writes_against_state_table(
        self,
        state_table: Dict[str, Any],
        completion_result: Dict[str, Any],
    ) -> List[str]:
        writes = completion_result.get("writes") if isinstance(completion_result, dict) else None
        if not isinstance(writes, dict) or not writes:
            return []
        visible_state = {k: v for k, v in state_table.items() if not str(k).startswith("__")}
        return self._collect_state_type_conflicts(visible_state, writes)
    def _record_pause_state(
        self,
        state_table: Dict[str, Any],
        current_frame: AnalysisFrame,
        *,
        pause_at: Dict[str, Any],
        call_info: Dict[str, Any],
        pre_process: dict,
        callee_name: str,
        callee_qualname: str,
    ) -> None:
        if not isinstance(state_table, dict):
            return None
        full_self_state = self._collect_full_self_state(
            current_frame=current_frame,
            call_info=call_info if isinstance(call_info, dict) else {},
            state_table=state_table,
            callee_qualname=callee_qualname,
            pre_process=pre_process,
        )
        if full_self_state:
            state_table["__last_full_self_state"] = deepcopy(full_self_state)
            owner = _owner_class_name(pre_process, callee_qualname) or _owner_class_name(pre_process, current_frame.unit_qualnames[0] if current_frame.unit_qualnames else "")
            if owner:
                by_class = state_table.setdefault("__full_self_by_class", {})
                if isinstance(by_class, dict):
                    by_class[owner] = deepcopy(full_self_state)
        if isinstance(call_info, dict):
            handoff_text = str(call_info.get("state_handoff_text", "") or "").strip()
            if handoff_text:
                state_table["__last_state_handoff_text"] = handoff_text
        return None
    def _record_completion_state(
        self,
        state_table: Dict[str, Any],
        current_frame: AnalysisFrame,
        completion_result: Dict[str, Any],
        *,
        pre_process: dict,
    ) -> None:
        def _merge(dst: Dict[str, Any], patch: Dict[str, Any], prefix: str = "") -> None:
            for k, v in patch.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                if k not in dst:
                    dst[k] = deepcopy(v)
                    continue
                old = dst.get(k)
                old_is_dict = isinstance(old, dict)
                new_is_dict = isinstance(v, dict)
                if old_is_dict and new_is_dict:
                    _merge(dst[k], v, prefix=path)
                    continue
                if old_is_dict != new_is_dict:
                    raise ValueError(
                        f"state_table type conflict at {path}: cannot replace {type(old).__name__} with {type(v).__name__}"
                    )
                dst[k] = deepcopy(v)
        writes = completion_result.get("writes") if isinstance(completion_result, dict) else None
        if isinstance(writes, dict) and writes:
            normalized_writes: Dict[str, Any] = {}
            self._merge_state_mapping(normalized_writes, writes)
            _merge(state_table, normalized_writes)
        runtime_state_text = str(completion_result.get("runtime_state_text", "") or "").strip() if isinstance(completion_result, dict) else ""
        if runtime_state_text:
            state_table["__runtime_state_text"] = runtime_state_text
        self._promote_completed_self_state(
            state_table,
            current_frame,
            completion_result if isinstance(completion_result, dict) else {},
            pre_process=pre_process,
        )

    def _promote_completed_self_state(
        self,
        state_table: Dict[str, Any],
        current_frame: AnalysisFrame,
        completion_result: Dict[str, Any],
        *,
        pre_process: dict,
    ) -> None:
        if not isinstance(state_table, dict):
            return None

        writes = completion_result.get("writes") if isinstance(completion_result, dict) else {}
        candidates: List[Any] = []
        if isinstance(writes, dict):
            for key in ("self", "__self__", "receiver"):
                val = writes.get(key)
                if isinstance(val, dict) and val:
                    candidates.append(val)

        full_self_state = self._collect_full_self_state(
            current_frame=current_frame,
            call_info=current_frame.call_info if isinstance(current_frame.call_info, dict) else {},
            state_table=state_table,
            callee_qualname=current_frame.unit_qualnames[0] if current_frame.unit_qualnames else "",
            pre_process=pre_process,
        )
        candidates.append(full_self_state)

        merged: Dict[str, Any] = {}
        for cand in candidates:
            if isinstance(cand, dict) and cand:
                self._merge_dict_like(merged, cand)

        if not merged:
            return None

        state_table["__last_full_self_state"] = deepcopy(merged)
        owner = self._get_frame_class_name(current_frame, pre_process)
        if owner:
            by_class = state_table.setdefault("__full_self_by_class", {})
            if isinstance(by_class, dict):
                existing = by_class.get(owner, {})
                if isinstance(existing, dict):
                    merged_owner = deepcopy(existing)
                    self._merge_dict_like(merged_owner, merged)
                    by_class[owner] = merged_owner
                else:
                    by_class[owner] = deepcopy(merged)
        return None

    def _state_path_for_target(self, target: ast.AST) -> List[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Attribute):
            base = self._state_path_for_target(target.value)
            if not base:
                return []
            return base + [target.attr]
        if isinstance(target, ast.Subscript):
            base = self._state_path_for_target(target.value)
            if not base:
                return []
            key = self._state_key_from_subscript(target.slice)
            if key is None:
                return []
            return base + [key]
        return []

    def _state_key_from_subscript(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Index):
            return self._state_key_from_subscript(node.value)
        return None

    def _set_state_path(self, state_table: Dict[str, Any], path: List[str], value: Any) -> None:
        if not path:
            return None
        cur: Dict[str, Any] = state_table
        for part in path[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[path[-1]] = deepcopy(value)
        return None

    def _bind_return_to_target(self, state_table: Dict[str, Any], target: ast.AST, value: Any) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (list, tuple)):
            for sub_target, sub_value in zip(target.elts, value):
                self._bind_return_to_target(state_table, sub_target, sub_value)
            return None
        path = self._state_path_for_target(target)
        if path:
            self._set_state_path(state_table, path, value)
        return None

    def _apply_return_to_caller_state(
        self,
        state_table: Dict[str, Any],
        caller_frame: AnalysisFrame,
        completion_result: Dict[str, Any],
        callee_name: str = "",
    ) -> None:
        if not isinstance(state_table, dict):
            return None
        ret, has_ret = self._extract_completion_return_value(completion_result if isinstance(completion_result, dict) else {})
        if not has_ret:
            return None

        pause_at = caller_frame.pause_at if isinstance(caller_frame.pause_at, dict) else {}
        stmt = str(pause_at.get("stmt", "") or "").strip()
        if not stmt:
            return None

        try:
            tree = ast.parse(stmt)
        except SyntaxError:
            return None
        if not tree.body:
            return None

        node = tree.body[0]
        targets: List[ast.AST] = []
        value_node: Optional[ast.AST] = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        elif isinstance(node, ast.AugAssign):
            return None

        if targets and isinstance(value_node, ast.Call) and self._call_matches_callee(value_node, callee_name):
            for target in targets:
                self._bind_return_to_target(state_table, target, ret)
        return None

    def _call_matches_callee(self, node: ast.Call, callee_name: str) -> bool:
        if not isinstance(node, ast.Call):
            return False
        callee_short = str(callee_name or "").split(".")[-1]
        if not callee_short:
            return True
        func = node.func
        if isinstance(func, ast.Name):
            return func.id == callee_short or func.id == callee_name
        if isinstance(func, ast.Attribute):
            return func.attr == callee_short
        return False

    def _finalize_resolved_call(
        self,
        state_table: Dict[str, Any],
        caller_frame: AnalysisFrame,
        callee_name: str,
        completion_result: Dict[str, Any],
        *,
        pre_process: dict,
    ) -> None:
        if not isinstance(state_table, dict):
            return None
        ret, has_ret = self._extract_completion_return_value(completion_result if isinstance(completion_result, dict) else {})
        if has_ret:
            state_table["__last_return_value"] = deepcopy(ret)
            state_table["__last_return_callee"] = str(callee_name or "")
            self._apply_return_to_caller_state(state_table, caller_frame, completion_result, callee_name=callee_name)
        full_self_state = self._collect_full_self_state(
            current_frame=caller_frame,
            call_info=caller_frame.call_info if isinstance(caller_frame.call_info, dict) else {},
            state_table=state_table,
            callee_qualname=caller_frame.unit_qualnames[0] if caller_frame.unit_qualnames else "",
            pre_process=pre_process,
        )
        if full_self_state:
            state_table["__last_full_self_state"] = deepcopy(full_self_state)
            owner = self._get_frame_class_name(caller_frame, pre_process)
            if owner:
                by_class = state_table.setdefault("__full_self_by_class", {})
                if isinstance(by_class, dict):
                    by_class[owner] = deepcopy(full_self_state)
        return None
    def _compress_previous_response_for_retry(self, response_json: Any, raw_text: str = "") -> Dict[str, Any]:
        if isinstance(response_json, dict):
            return deepcopy(response_json)
        if isinstance(raw_text, str) and raw_text.strip():
            return {"_raw": raw_text}
        return {}
    def _extract_completion_return_value(self, completion_result: Dict[str, Any]) -> Tuple[Any, bool]:
        if not isinstance(completion_result, dict) or "return" not in completion_result:
            return None, False
        ret = completion_result.get("return")
        if isinstance(ret, dict):
            return ret.get("value"), ("value" in ret)
        return ret, True
    def _known_callsite_texts(self, current_frame: AnalysisFrame) -> set[str]:
        out: set[str] = set()
        if isinstance(current_frame.pause_at, dict):
            stmt = str(current_frame.pause_at.get("stmt", "") or "").strip()
            if stmt:
                out.add(" ".join(stmt.split()))
        for rec in current_frame.route_records or []:
            callsites = rec.get("callsites", [])
            if isinstance(callsites, str):
                callsites = [callsites]
            if not isinstance(callsites, list):
                continue
            for cs in callsites:
                text = str(cs or "").strip()
                if text:
                    out.add(" ".join(text.split()))
        return out
    def _looks_like_call_expression_text(self, text: str) -> bool:
        t = " ".join(str(text or "").strip().split())
        if not t:
            return False
        if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\s*\(.*\)$", t):
            return True
        if "=" in t:
            rhs = t.split("=", 1)[1].strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\s*\(.*\)$", rhs):
                return True
        for prefix in ("return ", "if ", "elif ", "while ", "assert "):
            if t.startswith(prefix):
                body = t[len(prefix):].strip()
                if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\s*\(.*\)$", body):
                    return True
        return False
    def _value_looks_like_nonfinal_output(self, value: Any, *, current_frame: AnalysisFrame) -> bool:
        if not isinstance(value, str):
            return False
        t = " ".join(value.strip().split())
        if not t:
            return False
        if t in self._known_callsite_texts(current_frame):
            return True
        if self._looks_like_call_expression_text(t):
            return True
        if t.startswith("<function") or t.startswith("<bound method") or t.startswith("<module ") or t.startswith("<class "):
            return True
        if " object at 0x" in t:
            return True
        return False
    def _pause_stmt_is_plausible(self, pause_at: Dict[str, Any], callee: str) -> bool:
        return True
    def _entry_print_call_count(self, pre_process: dict) -> int:
        source = str(((pre_process.get("project", {}) or {}).get("entry_script", "") or "")).strip()
        if not source:
            source = str(((pre_process.get("entry", {}) or {}).get("code", "") or "")).strip()
        if not source:
            return 0
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0
        return self._count_runtime_print_calls(tree.body)

    def _count_runtime_print_calls(self, statements: Sequence[ast.stmt]) -> int:
        class RuntimePrintCounter(ast.NodeVisitor):
            def __init__(self) -> None:
                self.count = 0

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return None

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return None

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return None

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return None

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    self.count += 1
                self.generic_visit(node)

        counter = RuntimePrintCounter()
        for stmt in statements:
            counter.visit(stmt)
        return counter.count

    def _entry_linear_print_call_count(self, pre_process: dict) -> int:
        source = str(((pre_process.get("project", {}) or {}).get("entry_script", "") or "")).strip()
        if not source:
            source = str(((pre_process.get("entry", {}) or {}).get("code", "") or "")).strip()
        if not source:
            return 0
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0
        count = 0
        for stmt in getattr(tree, "body", []) or []:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            func = stmt.value.func
            if isinstance(func, ast.Name) and func.id == "print":
                count += 1
        return count

    def _entry_linear_prints_have_newline_literal(self, pre_process: dict) -> bool:
        source = str(((pre_process.get("project", {}) or {}).get("entry_script", "") or "")).strip()
        if not source:
            source = str(((pre_process.get("entry", {}) or {}).get("code", "") or "")).strip()
        if not source:
            return False
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for stmt in getattr(tree, "body", []) or []:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            func = call.func
            if not (isinstance(func, ast.Name) and func.id == "print"):
                continue
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and ("\n" in arg.value or "\r" in arg.value):
                    return True
        return False

    def _review_entry_solved_output(
        self,
        response_json: Dict[str, Any],
        *,
        current_frame: AnalysisFrame,
        pre_process: dict,
    ) -> Tuple[bool, str]:
        if isinstance(response_json.get("effect"), dict):
            return False, "entry_should_not_return_effect"
        if "pause_at" in response_json or "callee" in response_json:
            return False, "entry_and_paused_together"
        lines = response_json.get("final_output_lines")
        if not isinstance(lines, list):
            return False, "entry_missing_final_output_lines"
        if any(not isinstance(x, str) for x in lines):
            return False, "entry_bad_final_output_lines"
        print_count = self._entry_print_call_count(pre_process)
        linear_print_count = self._entry_linear_print_call_count(pre_process)
        has_newline_literal = self._entry_linear_prints_have_newline_literal(pre_process)
        if print_count == 1 and len(lines) > 1:
            response_json["final_output_lines"] = ["\n".join(lines)]
            lines = response_json["final_output_lines"]
        if (
            print_count == 1
            and len(lines) == 1
            and isinstance(lines[0], str)
            and (
                lines[0].count("\\n") + lines[0].count("\\r") >= 2
                or "\\\n" in lines[0]
                or "\\\r" in lines[0]
            )
        ):
            restored_line = (
                lines[0]
                .replace("\\\r\n", "")
                .replace("\\\n", "")
                .replace("\\r\\n", "\n")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
            )
            response_json["final_output_lines"] = [restored_line]
            lines = response_json["final_output_lines"]
        if print_count and linear_print_count == print_count and not has_newline_literal and len(lines) != print_count:
            return False, f"entry_wrong_output_line_count:{len(lines)}!={print_count}"
        if print_count and len(lines) > print_count:
            return False, f"entry_too_many_output_lines:{len(lines)}>{print_count}"
        return True, "ok"
    def _review_solved_output(
        self,
        response_json: Dict[str, Any],
        *,
        current_frame: AnalysisFrame,
        pre_process: dict,
    ) -> Tuple[bool, str]:
        if isinstance(response_json.get("effect"), dict):
            return False, "solved_should_not_return_effect"
        if "pause_at" in response_json or "callee" in response_json:
            return False, "solved_and_paused_together"
        ret = response_json.get("return")
        if not isinstance(ret, dict):
            return False, "solved_missing_return"
        if "value" not in ret:
            return False, "solved_missing_return"
        return True, "ok"
    def _validate_response(
        self,
        response_json: Dict[str, Any],
        raw_text: str,
        *,
        pre_process: dict,
        current_frame: AnalysisFrame,
    ) -> Tuple[bool, str]:
        if not isinstance(response_json, dict) or not response_json:
            return False, "json_invalid"
        kind = self._response_kind(response_json, current_frame=current_frame)
        if self._solved_output_mode_for_frame(current_frame) == "final_output_lines" and isinstance(response_json.get("effect"), dict):
            return False, "entry_should_not_return_effect"
        if kind == "entry_solved":
            return self._review_entry_solved_output(
                response_json,
                current_frame=current_frame,
                pre_process=pre_process,
            )
        if kind == "solved":
            return self._review_solved_output(
                response_json,
                current_frame=current_frame,
                pre_process=pre_process,
            )
        if kind == "paused":
            return True, "ok"
        return False, "missing_return_or_pause"
    def _looks_truncated_or_too_long(self, raw_text: str, response_json: Dict[str, Any]) -> bool:
        return False
    
    def _build_routed_payload_text(self, current_frame: AnalysisFrame) -> str:
        return json.dumps(current_frame.route_records, ensure_ascii=False, indent=2)
    def _build_retry_note(
        self,
        issue: str,
        *,
        attempt: int,
        round_idx: int,
        too_long: bool,
        previous_response: Optional[Dict[str, Any]] = None,
        repeated_callee: str = "",
    ) -> str:
        parts: List[str] = []
        normalized_issue = issue or ""
        if issue and issue != "ok":
            parts.append(f"Your previous reply was rejected: {issue}.")
        if previous_response:
            try:
                prev = json.dumps(previous_response, ensure_ascii=False, indent=2)
                parts.append("Previous rejected reply:")
                parts.append("```json")
                parts.append(prev)
                parts.append("```")
            except Exception:
                pass
        if normalized_issue == "entry_missing_final_output_lines":
            parts.append("You returned a completed entry frame, but final_output_lines was missing.")
            parts.append("For the entry script, return final_output_lines as a list with one exact output string per executed print call.")
        elif normalized_issue == "entry_bad_final_output_lines":
            parts.append("You returned final_output_lines, but it was not a list of strings.")
            parts.append("Return final_output_lines as a JSON list where each element is exactly one executed print call's output before the print-added trailing newline.")
        elif normalized_issue == "entry_output_contains_embedded_newline":
            parts.append("One final_output_lines element contained an embedded newline.")
            parts.append("This is allowed only when one executed print call prints a string containing that newline. Otherwise, preserve one element per executed print call.")
        elif normalized_issue.startswith("entry_wrong_output_line_count:"):
            detail = normalized_issue.split(":", 1)[1] if ":" in normalized_issue else ""
            parts.append(f"The number of final_output_lines was wrong ({detail}).")
            parts.append("The entry script has a straight-line sequence of print calls; preserve exactly one JSON string per executed print call, in order.")
        elif normalized_issue.startswith("entry_too_many_output_lines:"):
            detail = normalized_issue.split(":", 1)[1] if ":" in normalized_issue else ""
            parts.append(f"You returned too many printed output lines ({detail}).")
            parts.append("Each executed print call contributes one line. If print receives a list, tuple, dict, or object, keep the whole Python printed representation as one string.")
            parts.append("Do not split list/tuple/dict elements into separate output lines.")
        elif normalized_issue == "entry_should_not_return_effect":
            parts.append("For a completed entry frame, return final_output_lines only.")
        elif normalized_issue == "entry_and_paused_together":
            parts.append("Do not mix a completed entry output with pause fields.")
            parts.append("Choose exactly one: completed entry frame with final_output_lines, or paused frame with pause_at and callee.")
        elif normalized_issue == "solved_should_not_return_effect":
            parts.append('For a completed non-entry frame, return only {"kind":"complete","return":{"type":...,"value_repr":...}}.')
        elif normalized_issue == "solved_missing_return":
            parts.append("You returned a completed frame, but return was missing or invalid.")
            parts.append('If the current frame has really finished, include only {"kind":"complete","return":{"type":...,"value_repr":...}}.')
        elif normalized_issue == "solved_and_paused_together":
            parts.append("Do not mix a completed-frame return with pause fields in the same reply.")
            parts.append("Choose exactly one: completed frame with return, or paused frame with pause_at and callee.")
        elif normalized_issue in {"suspicious_final_output", "returned_pause_stmt_as_output"}:
            parts.append("Your previous output looked like an intermediate value or a debug/object summary, not the true final return value of the current frame.")
            parts.append("Do NOT return helper call text, object/callable/module tags, estimator/config shells, sparse-matrix string shells, or other summaries as final output.")
            parts.append("Locate the step where the caller should continue after that intermediate value, then continue CFG execution until the current frame really returns.")
        elif normalized_issue == "paused_missing_pause_at":
            parts.append("You returned a paused frame, but pause_at was missing.")
            parts.append("If you truly need to pause, include pause_at completely.")
        elif normalized_issue == "paused_missing_callee":
            parts.append("You returned a paused frame, but callee was missing.")
            parts.append("If you pause, include callee as a single function name or qualname.")
        elif normalized_issue == "paused_bad_callsite":
            parts.append("Your paused reply did not identify a plausible callsite.")
            parts.append("pause_at must contain the real current unit and the call statement where execution truly pauses. Include the block too when it is recoverable.")
        elif normalized_issue == "paused_and_solved_together":
            parts.append("Do not mix paused-frame fields with a completed-frame output in the same reply.")
            parts.append("Choose exactly one: paused frame with pause_at and callee, or completed frame with return/final_output_lines.")
        elif normalized_issue == "missing_return_or_pause":
            parts.append("Your JSON had neither a valid completed-frame return nor a valid paused callsite packet.")
            parts.append("Choose exactly one: completed frame with return, or paused frame with pause_at and callee.")
        elif normalized_issue.startswith("state_write_type_conflict:"):
            bad = normalized_issue.split(":", 1)[1]
            parts.append("Your previous solved reply proposed writes that would corrupt the accumulated state structure.")
            parts.append("Do NOT replace an existing dict subtree with a scalar/list/string, and do NOT replace an existing scalar with a dict.")
            parts.append(f"Conflicting paths: {bad}")
            parts.append("Return writes as a partial structural update that preserves the existing object shape.")
        elif normalized_issue.startswith("unresolved_callee:"):
            bad = normalized_issue.split(":", 1)[1]
            parts.append(f"We do not have a resolvable implementation of callee '{bad}' in the provided code context.")
            parts.append(f"Do NOT pause at callee '{bad}' again unless its implementation is actually available in the current units or inline library.")
            parts.append("If the truly important dependency is nested inside a larger expression, pause at that inner user-defined callee instead.")
        elif normalized_issue.startswith("repeat_pause_same_callee"):
            bad = repeated_callee or (normalized_issue.split(":", 1)[1] if ":" in normalized_issue else "")
            if bad:
                parts.append(f"You already paused repeatedly at callee '{bad}'. Do NOT pause at it again in this reply.")
            else:
                parts.append("You already paused repeatedly at the same callee. Do NOT repeat the same pause again.")
            parts.append("Either pause at the truly value-producing inner callee, or continue current-frame reasoning from the existing context.")
        elif normalized_issue.startswith("paused_to_blocked_callee:"):
            bad = normalized_issue.split(":", 1)[1]
            parts.append(f"Do NOT pause at blocked callee '{bad}'.")
            parts.append("This target is a builtin/library/container-style operation that should be reasoned inline.")
        elif normalized_issue.startswith("paused_to_inline_callee:"):
            bad = normalized_issue.split(":", 1)[1]
            parts.append(f"Do NOT pause at inline callee '{bad}'.")
            parts.append("That callee has already been materialized into the current frame units and should be executed locally.")
        elif normalized_issue.startswith("paused_not_routed_expand:"):
            bad = normalized_issue.split(":", 1)[1]
            parts.append(f"Do NOT pause at '{bad}' unless it is one of the routed expand callsites for this frame.")
            parts.append("Re-check the routed callsite list and only pause when execution reaches one of those mandatory expand calls on the taken path.")
        if too_long:
            parts.append("Your previous reply was too long or likely truncated.")
            parts.append("You may greatly shorten the trace text.")
            parts.append("If a variable is too long (arrays, long strings, dense numeric text, complex objects), omit it from the trace.")
            parts.append("Keep only the key control-flow and value updates.")
        if attempt >= 1 and not too_long:
            parts.append("Keep the trace short and only include the minimum key state needed for control flow and the final value.")
        if attempt >= 2:
            parts.append("Focus on correctness over verbosity. Do not repeat the same rejected structure.")
        return "\n".join(parts)
    def run_task(
        self,
        sample: dict,
        pre_process: dict,
        entry_info: dict,
        main_code: str,
        dep_sources: Optional[Dict[str, str]],
    ) -> TaskResult:
        sample_id = sample["id"]
        self.current_sample_file_count = len(sample.get("files", {}) or {})
        result = TaskResult(sample_id=sample_id)
        mode = entry_info.get("mode")
        if mode not in ("single", "entry_main"):
            result.success = False
            result.error_message = f"不支持的entry模式: {mode}"
            return result
        entry = entry_info.get("entry", {})
        entry_code_text = _trim_code_text(entry.get("code", "") or "")
        entry_cfg = build_cfg_text(entry_code_text, entry.get("name"))
        unit_labels: List[str] = [entry.get("name", "entry")]
        unit_qualnames: List[str] = [entry.get("qualname", "")]
        codes: List[str] = [entry_code_text]
        cfgs: List[str] = [entry_cfg]
        main_code_text = ""
        if mode == "entry_main":
            main = entry_info.get("main", {})
            main_code_text = _trim_code_text(main.get("code", "") or "")
            main_cfg = build_cfg_text(main_code_text, "main")
            unit_labels.append("main")
            unit_qualnames.append(main.get("qualname", ""))
            codes.append(main_code_text)
            cfgs.append(main_cfg)
        root_plan = self._build_routed_frame_plan(
            pre_process=pre_process,
            unit_labels=unit_labels,
            unit_qualnames=unit_qualnames,
            codes=codes,
            cfgs=cfgs,
            main_code=main_code,
            dep_sources=dep_sources,
            frame_name=entry.get("name", "entry"),
            frame_depth=0,
            task_result=result,
        )
        current_frame = AnalysisFrame(
            unit_labels=root_plan["unit_labels"],
            unit_qualnames=root_plan["unit_qualnames"],
            codes=root_plan["codes"],
            cfgs=root_plan["cfgs"],
            frame_context_text="",
            pause_callees=root_plan["pause_callees"],
            inline_callees=root_plan["inline_callees"],
            frame_name=entry.get("name", "entry"),
            route_records=root_plan["route_records"],
            unresolved_routes=root_plan["unresolved_routes"],
            class_contexts=root_plan["class_contexts"],
            class_context_code=root_plan["class_context_code"],
                    )
        state_table = self._init_light_state_table(pre_process, current_frame.unit_qualnames, current_frame.unit_labels)
        frame_stack: List[AnalysisFrame] = []
        prompt_text = build_prompt_normal(
            unit_labels=current_frame.unit_labels,
            codes=current_frame.codes,
            cfgs=current_frame.cfgs,
            frame_context_text=current_frame.frame_context_text,
            class_context_code=current_frame.class_context_code,
            routed_calls_text=self._build_routed_payload_text(current_frame),
            solved_output_mode=self._solved_output_mode_for_frame(current_frame),
        )
        repeat_call_counts: Dict[str, int] = {}
        while self._next_global_round_idx(result) <= self.config.max_rounds:
            global_round_idx = self._next_global_round_idx(result)
            response_json: Dict[str, Any] = {}
            raw_text = ""
            prompt_used = prompt_text
            valid = False
            issue = "missing_return_or_call"
            paused_callee_bundle: Optional[Tuple[str, str, str]] = None
            for attempt in range(self.config.max_retries + 1):
                extra_note = ""
                if attempt > 0:
                    too_long_prev = self._looks_truncated_or_too_long(raw_text, response_json)
                    previous_response = self._compress_previous_response_for_retry(
                        response_json=response_json,
                        raw_text=raw_text,
                    )
                    extra_note = self._build_retry_note(
                        issue,
                        attempt=attempt,
                        round_idx=global_round_idx,
                        too_long=too_long_prev,
                        previous_response=previous_response,
                    )
                prompt_used = prompt_text if not extra_note else f"{prompt_text}\n\n[IMPORTANT NOTE]\n{extra_note}\n"
                self._save_prompt_for_round(
                    sample_id=result.sample_id,
                    action="sim",
                    round_idx=global_round_idx,
                    prompt_text=prompt_used,
                )
                raw_text = self.call_text_llm(prompt_used)
                response_json = self._sanitize_response(
                    self._parse_sim_text_response(raw_text, current_frame),
                    current_frame=current_frame,
                )
                valid, issue = self._validate_response(
                    response_json,
                    raw_text,
                    pre_process=pre_process,
                    current_frame=current_frame,
                )
                response_kind = self._response_kind(response_json, current_frame=current_frame)
                if valid and response_kind == "paused":
                    pause_at = response_json.get("pause_at", {})
                    callee_name = str(response_json.get("callee", "") or "")
                    signature = json.dumps(
                        {
                            "frame": current_frame.frame_name,
                            "callee": callee_name,
                            "stmt": pause_at.get("stmt"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if repeat_call_counts.get(signature, 0) >= MAX_REPEAT_CALL_SIGNATURE:
                        valid = False
                        issue = f"repeat_pause_same_callee:{callee_name}"
                    else:
                        preferred_callee_qualname = self._resolve_paused_callee_qualname(
                            current_frame,
                            callee_name,
                            pause_at,
                            pre_process,
                        )
                        tmp_callee_code, tmp_callee_cfg, tmp_callee_qualname = self._get_callee_code_and_cfg(
                            callee_name,
                            pre_process,
                            main_code,
                            dep_sources,
                            preferred_qualname=preferred_callee_qualname,
                        )
                        if not isinstance(tmp_callee_qualname, str) or not tmp_callee_qualname.strip():
                            valid = False
                            issue = f"unresolved_callee:{callee_name}"
                            paused_callee_bundle = None
                        else:
                            paused_callee_bundle = (tmp_callee_code, tmp_callee_cfg, tmp_callee_qualname)
                if valid:
                    break
            if not valid:
                result.rounds.append(
                    RoundResult(
                        round_idx=global_round_idx,
                        frame_depth=len(frame_stack),
                        frame_name=current_frame.frame_name,
                        prompt_text=prompt_used,
                        raw_response_text=raw_text,
                        response_json=response_json,
                        status="error",
                    )
                )
                self._flush_task_result(result)
                result.success = False
                result.error_message = f"LLM响应无效 (round {global_round_idx}): {issue}"
                break
            status = self._response_kind(response_json, current_frame=current_frame)
            if status == "entry_solved":
                final_output_lines = list(response_json.get("final_output_lines", []) or [])
                result.rounds.append(
                    RoundResult(
                        round_idx=global_round_idx,
                        frame_depth=len(frame_stack),
                        frame_name=current_frame.frame_name,
                        prompt_text=prompt_used,
                        raw_response_text=raw_text,
                        response_json=response_json,
                        status="solved",
                    )
                )
                self._flush_task_result(result)
                if frame_stack:
                    result.success = False
                    result.error_message = "internal error: entry frame solved with nonempty frame stack"
                    break
                result.final_output = final_output_lines
                result.success = True
                break
            if status == "solved":
                completion_result: Dict[str, Any] = {
                    "return": deepcopy(response_json.get("return", {})),
                    "writes": {},
                }
                caller_for_state = frame_stack[-1] if frame_stack else None
                if self._should_run_completion_state_agent(
                    current_frame=current_frame,
                    response_json=response_json,
                    completion_result=completion_result,
                    pre_process=pre_process,
                    state_table=state_table,
                    caller_frame=caller_for_state,
                    task_result=result,
                ):
                    completion_result, state_agent_patch = self._run_completion_state_round(
                        current_frame=current_frame,
                        response_json=response_json,
                        completion_result=completion_result,
                        pre_process=pre_process,
                        state_table=state_table,
                        caller_frame=caller_for_state,
                        task_result=result,
                        frame_depth=len(frame_stack),
                    )
                else:
                    state_agent_patch = self._empty_completion_state_patch()
                patch_conflicts = self._validate_completion_writes_against_state_table(state_table, completion_result)
                if patch_conflicts:
                    response_json["state_agent_patch_discarded"] = {
                        "patch": state_agent_patch,
                        "reason": f"state_write_type_conflict:{'; '.join(patch_conflicts[:4])}",
                    }
                    completion_result = {"return": deepcopy(response_json.get("return", {})), "writes": {}}
                else:
                    response_json["state_agent_patch"] = state_agent_patch
                completion_return = completion_result.get("return")
                output = completion_return.get("value") if isinstance(completion_return, dict) and "value" in completion_return else completion_return
                self._record_completion_state(
                    state_table,
                    current_frame,
                    completion_result,
                    pre_process=pre_process,
                )
                result.rounds.append(
                    RoundResult(
                        round_idx=global_round_idx,
                        frame_depth=len(frame_stack),
                        frame_name=current_frame.frame_name,
                        prompt_text=prompt_used,
                        raw_response_text=raw_text,
                        response_json=response_json,
                        status="solved",
                    )
                )
                self._flush_task_result(result)
                if not frame_stack:
                    result.final_output = output
                    result.success = True
                    break
                caller_frame = frame_stack.pop()
                self._finalize_resolved_call(
                    state_table,
                    caller_frame,
                    current_frame.frame_name,
                    completion_result,
                    pre_process=pre_process,
                )
                call_result_payload = {
                    "return": deepcopy(completion_result.get("return", {})),
                }
                caller_frame.frame_context_text = self._prepare_resume_frame_context(
                    caller_frame,
                    state_table,
                    completion_result,
                    state_agent_patch,
                )
                continue_prompt = build_prompt_continue(
                    unit_labels=caller_frame.unit_labels,
                    codes=caller_frame.codes,
                    cfgs=caller_frame.cfgs,
                    frame_context_text=caller_frame.frame_context_text,
                    prior_trace_history=caller_frame.prior_trace_texts,
                    previous_pause_at=deepcopy(caller_frame.pause_at),
                    previous_callee=str(caller_frame.call_info.get("callee", "") or current_frame.frame_name),
                    call_result=call_result_payload,
                    class_context_code=caller_frame.class_context_code,
                    routed_calls_text=self._build_routed_payload_text(caller_frame),
                    solved_output_mode=self._solved_output_mode_for_frame(caller_frame),
                )
                current_frame = caller_frame
                prompt_text = continue_prompt
                continue
            if status == "paused":
                pause_at = response_json.get("pause_at", {})
                trace_text = str(response_json.get("trace_text", "") or "").strip()
                callee_name = str(response_json.get("callee", "") or "")
                call_info = {
                    "callee": callee_name,
                    "state_handoff_text": "",
                    "pause_at": deepcopy(pause_at),
                    "full_self_state": self._collect_full_self_state(
                        current_frame=current_frame,
                        call_info=current_frame.call_info if isinstance(current_frame.call_info, dict) else {},
                        state_table=state_table,
                        callee_qualname="",
                        pre_process=pre_process,
                    ),
                    "extensions": {
                        "search": {
                            "enabled": False,
                            "queries": [],
                            "notes": "reserved for future state-handoff search",
                        }
                    },
                }
                signature = json.dumps(
                    {
                        "frame": current_frame.frame_name,
                        "callee": callee_name,
                        "stmt": pause_at.get("stmt"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                repeat_call_counts[signature] = repeat_call_counts.get(signature, 0) + 1
                if len(frame_stack) >= MAX_FRAME_DEPTH:
                    result.rounds.append(
                        RoundResult(
                            round_idx=global_round_idx,
                            frame_depth=len(frame_stack),
                            frame_name=current_frame.frame_name,
                            prompt_text=prompt_used,
                            raw_response_text=raw_text,
                            response_json=response_json,
                            status="error",
                        )
                    )
                    self._flush_task_result(result)
                    result.success = False
                    result.error_message = "frame depth exceeded"
                    break
                if trace_text:
                    current_frame.prior_trace_texts = list(current_frame.prior_trace_texts) + [trace_text]
                current_frame.pause_at = pause_at
                current_frame.call_info = call_info
                result.rounds.append(
                    RoundResult(
                        round_idx=global_round_idx,
                        frame_depth=len(frame_stack),
                        frame_name=current_frame.frame_name,
                        prompt_text=prompt_used,
                        raw_response_text=raw_text,
                        response_json=response_json,
                        status="paused",
                    )
                )
                self._flush_task_result(result)
                if paused_callee_bundle is None:
                    result.rounds.append(
                        RoundResult(
                            round_idx=global_round_idx,
                            frame_depth=len(frame_stack),
                            frame_name=current_frame.frame_name,
                            prompt_text=prompt_used,
                            raw_response_text=raw_text,
                            response_json=response_json,
                            status="error",
                        )
                    )
                    self._flush_task_result(result)
                    result.success = False
                    result.error_message = f"internal error: missing resolved callee bundle for paused callee {callee_name}"
                    break
                callee_code, callee_cfg, callee_qualname = paused_callee_bundle
                if (
                    isinstance(callee_qualname, str)
                    and callee_qualname in (pre_process.get("classes", {}) or {})
                    and not self._has_explicit_init(pre_process, callee_qualname)
                ):
                    # bare class 没有显式 __init__，不要真的下钻成 child frame
                    # 直接把 class code 物化进当前 frame，然后让 caller 继续
                    if callee_qualname not in current_frame.unit_qualnames:
                        current_frame.unit_labels.append(_short_name_from_qualname(callee_qualname))
                        current_frame.unit_qualnames.append(callee_qualname)
                        current_frame.codes.append(callee_code)
                        current_frame.cfgs.append(callee_cfg)

                        current_frame.class_contexts = get_frame_class_contexts(
                            pre_process,
                            current_frame.unit_qualnames,
                            main_code,
                            dep_sources=dep_sources,
                            include_target_method=False,
                        )
                        current_frame.class_context_code = get_frame_class_context_code(
                            pre_process,
                            current_frame.unit_qualnames,
                            main_code,
                            dep_sources=dep_sources,
                            include_target_method=False,
                        )

                    # 不创建 child frame，直接让当前 frame 重新推理
                    prompt_text = build_prompt_normal(
                        unit_labels=current_frame.unit_labels,
                        codes=current_frame.codes,
                        cfgs=current_frame.cfgs,
                        frame_context_text=current_frame.frame_context_text,
                        class_context_code=current_frame.class_context_code,
                        routed_calls_text=self._build_routed_payload_text(current_frame),
                        solved_output_mode=self._solved_output_mode_for_frame(current_frame),
                    )
                    continue
                state_handoff_text = self._run_state_handoff_round(
                    current_frame=current_frame,
                    pause_at=pause_at,
                    partial_call=call_info,
                    partial_resume_state={},
                    callee_name=callee_name,
                    callee_qualname=callee_qualname,
                    callee_code=callee_code,
                    pre_process=pre_process,
                    main_code=main_code,
                    dep_sources=dep_sources,
                    state_table=state_table,
                    task_result=result,
                    frame_depth=len(frame_stack),
                )
                call_info = {
                    "callee": callee_name,
                    "state_handoff_text": state_handoff_text,
                    "pause_at": deepcopy(pause_at),
                    "full_self_state": self._collect_full_self_state(
                        current_frame=current_frame,
                        call_info=current_frame.call_info if isinstance(current_frame.call_info, dict) else {},
                        state_table=state_table,
                        callee_qualname=callee_qualname,
                        pre_process=pre_process,
                    ),
                    "extensions": {
                        "search": {
                            "enabled": False,
                            "queries": [],
                            "notes": "reserved for future state-handoff search",
                        }
                    },
                }
                current_frame.pause_at = pause_at
                current_frame.call_info = call_info
                frame_stack.append(current_frame.copy())
                self._record_pause_state(
                    state_table,
                    current_frame,
                    pause_at=pause_at,
                    call_info=call_info,
                    pre_process=pre_process,
                    callee_name=callee_name,
                    callee_qualname=callee_qualname,
                )
                callee_frame_context = self._prepare_callee_frame_context(
                    current_frame,
                    callee_name,
                    call_info,
                    pre_process=pre_process,
                    callee_qualname=callee_qualname,
                    state_table=state_table,
                )
                callee_plan = self._build_routed_frame_plan(
                    pre_process=pre_process,
                    unit_labels=[callee_name],
                    unit_qualnames=[callee_qualname],
                    codes=[callee_code],
                    cfgs=[callee_cfg],
                    main_code=main_code,
                    dep_sources=dep_sources,
                    frame_name=callee_name,
                    frame_depth=len(frame_stack),
                    task_result=result,
                )
                callee_frame = AnalysisFrame(
                    unit_labels=callee_plan["unit_labels"],
                    unit_qualnames=callee_plan["unit_qualnames"],
                    codes=callee_plan["codes"],
                    cfgs=callee_plan["cfgs"],
                    frame_context_text=callee_frame_context,
                    pause_callees=callee_plan["pause_callees"],
                    inline_callees=callee_plan["inline_callees"],
                    frame_name=callee_name,
                    route_records=callee_plan["route_records"],
                    unresolved_routes=callee_plan["unresolved_routes"],
                    class_contexts=callee_plan["class_contexts"],
                    class_context_code=callee_plan["class_context_code"],
                )
                callee_frame.call_info = {
                    "callee": callee_name,
                    "state_handoff_text": state_handoff_text,
                    "pause_at": deepcopy(pause_at),
                    "full_self_state": deepcopy(call_info.get("full_self_state", {})) if isinstance(call_info.get("full_self_state"), dict) else {},
                }
                callee_prompt = build_prompt_normal(
                    unit_labels=callee_frame.unit_labels,
                    codes=callee_frame.codes,
                    cfgs=callee_frame.cfgs,
                    frame_context_text=callee_frame.frame_context_text,
                    class_context_code=callee_frame.class_context_code,
                    routed_calls_text=self._build_routed_payload_text(callee_frame),
                    solved_output_mode=self._solved_output_mode_for_frame(callee_frame),
                )
                current_frame = callee_frame
                prompt_text = callee_prompt
                continue
            result.rounds.append(
                RoundResult(
                    round_idx=global_round_idx,
                    frame_depth=len(frame_stack),
                    frame_name=current_frame.frame_name,
                    prompt_text=prompt_used,
                    raw_response_text=raw_text,
                    response_json=response_json,
                    status="error",
                )
            )
            self._flush_task_result(result)
            result.success = False
            result.error_message = f"未知status: {status}"
            break
        if (
            self._next_global_round_idx(result) > self.config.max_rounds
            and not result.success
            and not result.error_message
        ):
            result.error_message = f"达到最大轮数限制 ({self.config.max_rounds})"
        self._flush_task_result(result)
        return result
    def save_task_result(self, result: TaskResult, output_dir: str, *, write_summary: bool = False) -> None:
        sample_dir = os.path.join(output_dir, result.sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        for state_round in result.state_rounds:
            round_dir = os.path.join(sample_dir, f"round{state_round.round_idx}_state")
            os.makedirs(round_dir, exist_ok=True)
            _write_text(os.path.join(round_dir, "prompt.txt"), state_round.prompt_text)
            _write_text(os.path.join(round_dir, "response_raw.txt"), state_round.raw_response_text)
            state_meta = {
                "status": state_round.status,
                "state_kind": state_round.state_kind,
                "frame_depth": state_round.frame_depth,
                "caller_frame_name": state_round.caller_frame_name,
                "callee_name": state_round.callee_name,
                "callee_qualname": state_round.callee_qualname,
                "partial_pause": state_round.partial_pause,
                "static_info": state_round.static_info,
                "response_struct": state_round.response_json,
            }
            _write_text(os.path.join(round_dir, "meta.txt"), _pretty_text(state_meta))
        for route_round in result.route_rounds:
            round_dir = os.path.join(sample_dir, f"round{route_round.round_idx}_route")
            os.makedirs(round_dir, exist_ok=True)
            _write_text(os.path.join(round_dir, "prompt.txt"), route_round.prompt_text)
            _write_text(os.path.join(round_dir, "response_raw.txt"), route_round.raw_response_text)
            route_meta = {
                "status": route_round.status,
                "frame_depth": route_round.frame_depth,
                "frame_name": route_round.frame_name,
                "unit_label": route_round.unit_label,
                "unit_qualname": route_round.unit_qualname,
                "route_items": route_round.route_items,
                "route_records": route_round.route_records,
            }
            _write_text(os.path.join(round_dir, "meta.txt"), _pretty_text(route_meta))
        for round_result in result.rounds:
            round_dir = os.path.join(sample_dir, f"round{round_result.round_idx}_sim")
            os.makedirs(round_dir, exist_ok=True)
            _write_text(os.path.join(round_dir, "prompt.txt"), round_result.prompt_text)
            _write_text(os.path.join(round_dir, "response_raw.txt"), round_result.raw_response_text)
            status_info = {
                "status": round_result.status,
                "frame_name": round_result.frame_name,
                "frame_depth": round_result.frame_depth,
                "call_result": round_result.call_result,
                "response_struct": round_result.response_json,
            }
            _write_text(os.path.join(round_dir, "status.txt"), _pretty_text(status_info))
        if write_summary:
            summary = {
                "sample_id": result.sample_id,
                "success": result.success,
                "final_output": result.final_output,
                "error_message": result.error_message,
                "total_sim_rounds": len(result.rounds),
                "total_state_rounds": len(result.state_rounds),
                "total_state_agent_rounds": len(result.state_rounds),
                "total_route_rounds": len(result.route_rounds),
            }
            _write_text(
                os.path.join(sample_dir, "summary.json"),
                json.dumps(summary, ensure_ascii=False, indent=2),
            )
DATASET_ROOT = Path("PyRepoExec")
def flat_name_to_logical_path(filename: str) -> str:
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
        if not path.is_file() or path.suffix != ".py":
            continue
        logical_path = flat_name_to_logical_path(path.name)
        files[logical_path] = path.read_text(encoding="utf-8")
    return files
def build_sample_from_task(project_dir: Path, task_dir: Path) -> Optional[dict]:
    slicing_dir = task_dir / "slicing"
    entry_file = task_dir / "entry.py"
    if not slicing_dir.exists() or not slicing_dir.is_dir():
        print(f"[skip] no slicing dir: {task_dir}")
        return None
    if not entry_file.exists() or not entry_file.is_file():
        print(f"[skip] no entry.py: {task_dir}")
        return None
    files = load_slicing_files(slicing_dir)
    if not files:
        print(f"[skip] no slice files: {task_dir}")
        return None
    entry_script = entry_file.read_text(encoding="utf-8").strip()
    return {
        "id": f"{project_dir.name}/{task_dir.name}",
        "files": files,
        "entry_script": entry_script,
    }
def run_single_sample_by_data(sample: dict, config: InferenceConfig) -> TaskResult:
    sample_id = sample["id"]
    engine = InferenceEngine(config)
    try:
        project_files = sample.get("files")
        entry_script = sample.get("entry_script")
        if not isinstance(project_files, dict) or not isinstance(entry_script, str):
            raise ValueError(
                "dataset sample must contain 'files' (dict[path, code]) and 'entry_script' (str)."
            )
        dep_sources = None
        pre_process = analyze_project(project_files, entry_script)
        entry_info = get_entry_code(pre_process, "", "", dep_sources=dep_sources)
        result = engine.run_task(
            sample=sample,
            pre_process=pre_process,
            entry_info=entry_info,
            main_code="",
            dep_sources=dep_sources,
        )
        engine.save_task_result(result, config.output_dir, write_summary=True)
        return result
    except Exception as e:
        result = TaskResult(
            sample_id=sample_id,
            success=False,
            error_message=f"异常: {e}\n{traceback.format_exc()}",
        )
        try:
            engine.save_task_result(result, config.output_dir, write_summary=True)
        except Exception:
            pass
        return result
def _run_samples_parallel(samples: list, config: InferenceConfig, max_workers: int):
    print(f"\n并行执行 {len(samples)} 个样本，最大并发数: {max_workers}")
    print_lock = threading.Lock()
    completed_count = [0]
    def run_with_progress(sample, idx):
        sample_id = sample["id"]
        result = run_single_sample_by_data(sample, config)
        completed_count[0] += 1
        with print_lock:
            status = "成功" if result.success else "失败"
            print(
                f"[{completed_count[0]}/{len(samples)}] {sample_id}: {status}, "
                f"sim轮数: {len(result.rounds)}, state agent轮数: {len(result.state_rounds)}, "
                f"route轮数: {len(result.route_rounds)}"
            )
            if result.error_message:
                print(f"  错误: {result.error_message}")
        return result
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_with_progress, sample, idx): sample
            for idx, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            future.result()
    print(f"\n完成! 结果保存在: {config.output_dir}")
def main(
    target_sample_id: str = None,
    sample_prefix: str = None,
    parallel: bool = True,
    max_workers: int = 20,
    dataset_root: str = "PyRepoExec",
):
    config = InferenceConfig(
        max_rounds=150,
        output_dir=DEFAULT_OUTPUT_DIR,
        model=DEFAULT_MODEL,
        api_key=os.environ.get("YOUR_API_KEY_ENV"),
        base_url=DEFAULT_BASE_URL,
    )

    all_samples = iter_dataset_samples(dataset_root)
    print(f"读取到 {len(all_samples)} 个样本")

    if target_sample_id:
        samples_to_run = [s for s in all_samples if s["id"] == target_sample_id]
        if not samples_to_run:
            print(f"未找到样本: {target_sample_id}")
            return
    elif sample_prefix:
        samples_to_run = [s for s in all_samples if s["id"].startswith(sample_prefix)]
        print(f"匹配前缀 '{sample_prefix}' 的样本数: {len(samples_to_run)}")
    else:
        samples_to_run = all_samples

    if not samples_to_run:
        print("没有要运行的样本")
        return

    if parallel:
        _run_samples_parallel(samples_to_run, config, max_workers)
    else:
        for idx, sample in enumerate(samples_to_run, 1):
            result = run_single_sample_by_data(sample, config)
            status = "成功" if result.success else "失败"
            print(
                f"[{idx}/{len(samples_to_run)}] {sample['id']}: {status}, "
                f"sim轮数: {len(result.rounds)}, state agent轮数: {len(result.state_rounds)}, "
                f"route轮数: {len(result.route_rounds)}"
            )
            if result.error_message:
                print(f"  错误: {result.error_message}")

    print(f"\n完成! 结果保存在: {config.output_dir}")
if __name__ == "__main__":
    main(dataset_root="PyRepoExec")
