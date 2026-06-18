from __future__ import annotations

import json
import pprint
from typing import Any, Dict, List, Mapping, Sequence


def _j(obj: Any) -> str:
    def _default(value: Any) -> str:
        if isinstance(value, (bytes, bytearray)):
            return repr(bytes(value))
        if isinstance(value, set):
            try:
                return sorted(value)
            except Exception:
                return sorted([repr(x) for x in value])
        return repr(value)

    return json.dumps(obj, ensure_ascii=False, indent=2, default=_default)


def _plain(obj: Any) -> str:
    try:
        return pprint.pformat(obj, width=120, sort_dicts=False, compact=False)
    except Exception:
        return repr(obj)


def _plain_mapping_lines(mapping: Mapping[str, Any] | None) -> str:
    if not mapping:
        return "none"
    lines: List[str] = []
    for key, value in dict(mapping).items():
        lines.append(f"{key}: {_plain(value)}")
    return "\n".join(lines)


def _code_block(text: str, lang: str) -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def _append_note(parts: List[str], note_text: str) -> None:
    note_text = (note_text or "").strip()
    if not note_text:
        return
    parts.extend([
        "[Note]",
        note_text,
        "",
    ])


def _render_code_sections(unit_labels: Sequence[str], codes: Sequence[str]) -> List[str]:
    parts: List[str] = ["[Code]"]
    for label, code in zip(unit_labels, codes):
        parts.extend([
            f"Unit: {label}",
            "```python",
            (code or "").strip(),
            "```",
            "",
        ])
    return parts


def _render_cfg_sections(unit_labels: Sequence[str], cfgs: Sequence[str]) -> List[str]:
    parts: List[str] = ["[CFG]"]
    for label, cfg in zip(unit_labels, cfgs):
        parts.extend([
            f"Unit: {label}",
            "```",
            (cfg or "").strip(),
            "```",
            "",
        ])
    return parts


def _core_header_text(solved_output_mode: str) -> str:
    if solved_output_mode == "final_output_lines":
        goal = "determine whether the current unit completes or pauses, and if it completes, derive its final printed output"
    else:
        goal = "determine whether the current unit completes or pauses, and if it completes, derive its final return value"

    return "\n".join([
        "YOU ARE A PYTHON PROGRAM ANALYST.",
        f"Your job is to simulate execution using the given source code and CFG, and {goal}.",
        "Write your reasoning as a compact block-by-block execution trace, then give the result in a small JSON object.",
    ])

OUTPUT_RULES = """
Return exactly two top-level sections in this order:
[Trace]
<Your simulated execution trace>
[Result]
<exactly one compact valid JSON object>

[Trace] should be a compact semi-formal execution trace in natural language.
Write it as a sequence of local execution blocks.
For each trace block, include as much of the following as is relevant:
- the current function, method, or execution context
- the current block
- the statement, operation, or event being executed
- the relevant state after this part of execution
- the next block or control-flow destination
Prefer one paragraph per trace block.
A trace block may contain multiple short sentences, but it should stay unified rather than split into separate mini-sections.
Keep each trace block local, concrete, and easy to follow.

Exact-value rules:
- Preserve the exact runtime value and the exact Python printed/repr form; do not replace it with an equivalent-looking normalized form.
- For strings, distinguish the runtime string from its printed repr. Backslashes may be escaped in repr, but length/counting operations use the runtime characters.
- Preserve numeric-looking strings exactly as strings. Do not convert `'10'` to `'10.0'`, `10`, or `10.0` unless the executed code explicitly performs that conversion.
- Preserve date/time unit strings exactly unless the executed code explicitly changes them; do not append default time fields or change calendar names by convention.
- Preserve regex patterns, URL components, base64/auth strings, and angle-bracket text exactly. Do not treat `<name>` as a placeholder and do not remove or rewrite angle brackets unless the code explicitly does so.
- When code calls an escaping helper such as `re.escape`, apply it to literal punctuation exactly; do not omit escapes for characters that the helper escapes in the produced regex string.
- When encoding or decoding base64-like data, track every source character exactly; punctuation such as `:` is significant and must not be dropped or substituted.
- For ordered list merging or media-like asset merging, if the code removes duplicates, keep the first occurrence and skip later duplicates without reordering.
- For symbolic/math expressions, do not rewrite to an algebraically equivalent alternative form unless the executed code actually performs that rewrite.
- For loops that append, concatenate, or accumulate output fragments, update the accumulator after each iteration and preserve separators produced by slices before and after every match.
- When reading an object attribute whose class/source is shown, use the exact class attribute value from source; do not replace it with a semantically similar value.
- For functions wrapped with `functools.wraps(original_func)`, preserve the wrapper metadata copied from the original function. In particular, `wrapper.__name__` should be the original function's `__name__`, not the local wrapper's source name.

Example style for [Trace]:
At main:b1, evaluating `x = A(1)` initiates a call to `A.__init__`, so control transfers out of `main` and into the callee.
At A.__init__:b1, the initializer assigns `self.v = 1` and then returns. Back in the caller, the relevant state is now `x.v = 1`, and execution proceeds to `main:b2`.
At main:b2, `i = 0` and `acc = 0` are established, after which control reaches the loop header at `main:b3`.
At main:b3, the condition `i < 2` is true under the current state, so execution takes the loop body at `main:b4`.
At main:b4, `acc += i` runs and keeps `acc = 0`; the update block `main:b5` then advances `i` to `1` and sends control back to `main:b3`.
At main:b3, the condition still holds, so the body runs once more: `main:b4` updates `acc` to `1`, `main:b5` advances `i` to `2`, and control returns to `main:b3`.
At main:b3, the condition is now false, so execution exits the loop to `main:b6`, which returns `acc = 1`.
""".strip()


def build_result_rules(mode: str) -> str:
    if mode == "return_only":
        return """
[Result] must be exactly one compact valid JSON object.

If execution pauses before the current unit completes, output:
{"kind":"pause","pause_at":{"unit":...,"block":...,"stmt":...},"callee":...}
Here, "pause_at.unit" is the current unit, "pause_at.block" is the paused block, "pause_at.stmt" is the paused statement, and "callee" identifies the callee unit to enter next, using the task's callee representation.

If the current unit completes, output:
{"kind":"complete","return":{"type":...,"value_repr":...}}
Here, "return.type" must be the exact Python type, and "return.value_repr" must be the exact Python return value rendered in valid Python syntax.
For string return values, preserve every character unless the executed code explicitly removes or replaces it. Track digits, punctuation, entity/escape decoding, separators, case conversion, substitutions, and max-length trimming step by step; do not paraphrase or normalize strings from memory.
Preserve numeric-looking strings exactly as strings. Do not convert `'10'` to `'10.0'`, `10`, or `10.0` unless the executed code explicitly performs that conversion.
Preserve regex patterns, URL components, base64/auth strings, and angle-bracket text exactly. Do not treat `<name>` as a placeholder and do not remove or rewrite angle brackets unless the code explicitly does so.
When code calls an escaping helper such as `re.escape`, apply it to literal punctuation exactly; do not omit escapes for characters that the helper escapes in the produced regex string.
When encoding or decoding base64-like data, track every source character exactly; punctuation such as `:` is significant and must not be dropped or substituted.
When applying character filters, preserve characters that the filter allows. In particular, do not drop digits merely because punctuation around them is removed.
When applying word-boundary truncation, split on the actual separator and test the exact candidate length before appending each word. Do not include a later shorter word after a longer rejected word unless the code's save-order logic permits it.
For symbolic/math return values, preserve the exact expression form produced by the executed code; do not substitute an algebraically equivalent alternative.
For loops that append, concatenate, or accumulate return-value fragments, update the accumulator after each iteration and preserve separators produced by slices before and after every match.
When returning object attributes from instances whose class/source is shown, use the exact class attribute value from source; do not replace it with a semantically similar value.
For generator functions that execute `yield`: when this expanded frame is being consumed by a caller `for` loop, `list(...)`, `tuple(...)`, or another immediate iteration context, report the caller-visible yielded sequence as the return value, not the generator function's implicit `None`.
For yielded imported modules or module-like objects, preserve the module identity and any retrieved source bindings needed by the caller, for example a module-like dict/object containing `__module__` plus relevant attributes.
For functions wrapped with `functools.wraps(original_func)`, preserve copied metadata such as `__name__`; a wrapper defined as `compile_func` may still have `__name__ == original_func.__name__`.
""".strip()
    elif mode == "final_output_lines":
        return """
[Result] must be exactly one compact valid JSON object.

If execution pauses before the current unit completes, output:
{"kind":"pause","pause_at":{"unit":...,"block":...,"stmt":...},"callee":...}
Here, "pause_at.unit" is the current unit, "pause_at.block" is the paused block, "pause_at.stmt" is the paused statement, and "callee" identifies the callee unit to enter next, using the task's callee representation.

If the current unit completes, output:
{"kind":"complete","final_output_lines":[...]}
Here, "final_output_lines" must be a list of strings, where each string is the exact output produced by one executed print call before the print-added trailing newline.
Each executed print call contributes exactly one output element. If print receives a list, tuple, dict, set, object representation, or other non-string value, put Python's printed representation of that whole value in one string; do not split its elements into multiple output lines and do not summarize it.
If the entry has multiple print calls, preserve one final_output_lines element per executed print call in the same order. Do not merge several print calls into one string, even if they execute in the same block.
If one executed print call prints a string containing "\n" or "\r", keep that exact embedded newline inside that one element, because it is still the output of one print call.
For string-processing code, preserve every character unless the executed code explicitly removes or replaces it. Track digits, punctuation, HTML/entity decoding, separators, case conversion, regex substitutions, and max-length trimming step by step; do not paraphrase or normalize strings from memory.
Preserve numeric-looking strings exactly as strings. Do not convert `'10'` to `'10.0'`, `10`, or `10.0` unless the executed code explicitly performs that conversion.
Preserve regex patterns, URL components, base64/auth strings, and angle-bracket text exactly. Do not treat `<name>` as a placeholder and do not remove or rewrite angle brackets unless the code explicitly does so.
When code calls an escaping helper such as `re.escape`, apply it to literal punctuation exactly; do not omit escapes for characters that the helper escapes in the produced regex string.
When encoding or decoding base64-like data, track every source character exactly; punctuation such as `:` is significant and must not be dropped or substituted.
When applying word-boundary truncation, split on the actual separator and test the exact candidate length before appending each word. Do not include a later shorter word after a longer rejected word unless the code's save-order logic permits it.
For containers, preserve Python repr exactly: list/tuple/dict/set delimiters, quotes, spaces after commas/colons, and insertion order.
For loops that append, concatenate, or accumulate printed-output fragments, update the accumulator after each iteration and preserve separators produced by slices before and after every match.
When printing object attributes from instances whose class/source is shown, use the exact class attribute value from source; do not replace it with a semantically similar value.
If shared state represents a Python dict key as a tuple-repr string such as `('a', 'b')`, treat that as the runtime tuple key when printing the dict, not as a normal string key.
For functions wrapped with `functools.wraps(original_func)`, preserve copied metadata such as `__name__` when printing or comparing attributes.
""".strip()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_prompt_normal(
    *,
    unit_labels: List[str],
    codes: List[str],
    cfgs: List[str],
    frame_context_text: str = "",
    class_context_code: str = "",
    routed_calls_text: str = "",
    note_text: str = "",
    solved_output_mode: str = "return_only",
) -> str:
    assert len(unit_labels) == len(codes) == len(cfgs), "unit_labels/codes/cfgs length mismatch"

    if solved_output_mode == "return_only":
        result_mode = "return_only"
    elif solved_output_mode == "final_output_lines":
        result_mode = "final_output_lines"
    else:
        raise ValueError(f"Unknown solved_output_mode: {solved_output_mode}")

    parts: List[str] = []

    # 1) Core job
    parts.extend(_core_header_text(result_mode).splitlines())
    parts.append("")

    # 2) Context
    parts.extend(_render_code_sections(unit_labels, codes))
    parts.extend(_render_cfg_sections(unit_labels, cfgs))

    if class_context_code.strip():
        parts.extend([
            "[Class Context]",
            _code_block(class_context_code.strip(), "python"),
            "",
        ])

    if (routed_calls_text or "").strip():
        parts.extend([
            "[Routed Calls]",
            "The following routed decisions are already fixed for this unit.",
            "If a callee is marked inline, reason through it locally in [Trace].",
            "If a callee is marked expand, return a pause result when execution reaches that callsite on the taken path.",
            "If a route item has a `precision` field, treat it as a warning that exact bytes/characters/order/numeric representation matter; do not fill in values from memory or convention.",
            "```",
            routed_calls_text.strip(),
            "```",
            "",
        ])

    if (frame_context_text or "").strip():
        parts.extend([
            "[Frame Context]",
            "```",
            frame_context_text.strip(),
            "```",
            "",
        ])

    _append_note(parts, note_text)

    # 3) Output requirements
    parts.extend([
        "[Output requirements]",
        OUTPUT_RULES,
        "",
        build_result_rules(result_mode),
    ])

    return "\n".join(parts)



def build_prompt_continue(
    *,
    unit_labels: List[str],
    codes: List[str],
    cfgs: List[str],
    frame_context_text: str = "",
    prior_trace_history: Sequence[str],
    previous_pause_at: Dict[str, Any],
    previous_callee: str,
    call_result: Dict[str, Any],
    class_context_code: str = "",
    routed_calls_text: str = "",
    note_text: str = "",
    solved_output_mode: str = "return_only",
) -> str:
    assert len(unit_labels) == len(codes) == len(cfgs), "unit_labels/codes/cfgs length mismatch"

    if solved_output_mode == "return_only":
        result_mode = "return_only"
    elif solved_output_mode == "final_output_lines":
        result_mode = "final_output_lines"
    else:
        raise ValueError(f"Unknown solved_output_mode: {solved_output_mode}")

    parts: List[str] = []

    # 1) Core job
    parts.extend(_core_header_text(result_mode).splitlines())
    parts.extend([
        "This unit is resuming after a separately expanded callee has returned.",
        "Use the returned value at the paused callsite and continue execution from that point.",
        "Do not repeat the already-finished earlier blocks.",
        "",
    ])

    # 2) Context
    parts.extend(_render_code_sections(unit_labels, codes))
    parts.extend(_render_cfg_sections(unit_labels, cfgs))

    if class_context_code.strip():
        parts.extend([
            "[Class Context]",
            _code_block(class_context_code.strip(), "python"),
            "",
        ])

    if (routed_calls_text or "").strip():
        parts.extend([
            "[Routed Calls]",
            "The following routed decisions are already fixed for this unit.",
            "If a callee is marked inline, reason through it locally in [Trace].",
            "If a callee is marked expand, return a pause result when execution reaches that callsite on the taken path.",
            "If a route item has a `precision` field, treat it as a warning that exact bytes/characters/order/numeric representation matter; do not fill in values from memory or convention.",
            "```json",
            routed_calls_text.strip(),
            "```",
            "",
        ])

    if (frame_context_text or "").strip():
        parts.extend([
            "[Frame Context]",
            "```text",
            frame_context_text.strip(),
            "```",
            "",
        ])

    parts.extend([
        "[Resume Inputs]",
        "[Previous Pause]",
        "```text",
        _plain_mapping_lines(previous_pause_at or {}),
        "```",
        "",
        "[Previous Callee]",
        (previous_callee or "").strip(),
        "",
    ])

    if prior_trace_history:
        history_lines = [str(x) for x in prior_trace_history if str(x).strip()]
        if history_lines:
            parts.extend([
                "[Recent Caller Trace]",
                "\n".join(history_lines),
                "",
            ])

    parts.extend([
        "[Call Result]",
        "```text",
        _plain_mapping_lines(call_result or {}),
        "```",
        "",
    ])

    _append_note(parts, note_text)

    # 3) Output requirements
    parts.extend([
        "[Output requirements]",
        OUTPUT_RULES,
        "",
        "[Resume-specific requirements]",
        "- Start [Trace] from the paused point onward.",
        "- Incorporate the returned call result before continuing execution.",
        "- The paused call has already completed. First bind its returned value to the paused expression/assignment target exactly as Python would, then continue with the next caller statement.",
        "- Treat [Call Result] as authoritative at the paused expression. Do not recompute, reinterpret, normalize, or replace it with a value guessed from the callee source or from general library knowledge.",
        "- If the returned value is a container, iterator materialization, string, bytes-like data, URL, digest, regex, numeric array, unit string, datetime value, or module-like object, preserve its exact representation when the caller consumes it.",
        "- Do not re-trace blocks that were already completed before the pause.",
        "",
        build_result_rules(result_mode),
    ])

    return "\n".join(parts)



def build_state_handoff_prompt(
    *,
    caller_code: str,
    caller_steps: Sequence[Mapping[str, Any]] | None,
    pause_at: Mapping[str, Any],
    callee_code: str,
    shared_state: Mapping[str, Any] | None = None,
    class_context_code: str | None = None,
    retrieved_binding_sources_text: str = "",
) -> str:
    steps_text = "\n\n".join(str(x) for x in (caller_steps or []) if str(x).strip()) or "none"
    pause_text = _plain_mapping_lines(dict(pause_at or {}))
    shared_mapping = dict(shared_state or {})
    shared_text = _plain_mapping_lines(shared_mapping)
    runtime_state_text = str(shared_mapping.get("__runtime_state_text", "") or "").strip()

    parts: List[str] = [
        "YOU ARE A PYTHON PROGRAM ANALYST.",
        "You are the state agent for this repository-level execution.",
        "Infer the best possible child-frame runtime state for the callee of a paused call.",
        "Use the caller code, recent caller steps, the paused callsite, the callee code, and any recovered shared bindings together.",
        "Your goal is to produce a faithful handoff note for the next callee frame.",
        "",
        "[Caller Code]",
        _code_block((caller_code or "").strip(), "python"),
        "",
        "[Caller Recent Steps]",
        "```text",
        steps_text,
        "```",
        "",
        "[Caller Paused Point]",
        "```text",
        pause_text,
        "```",
        "",
        "[Shared State After Previous Call]",
        "```text",
        shared_text,
        "```",
        "",
    ]

    if runtime_state_text:
        parts.extend([
            "[Shared Runtime State Ledger]",
            "```text",
            runtime_state_text,
            "```",
            "",
        ])

    parts.extend([
        "[Callee Code]",
        _code_block((callee_code or "").strip(), "python"),
    ])

    if class_context_code and class_context_code.strip():
        parts.extend([
            "",
            "[Class Context]",
            _code_block(class_context_code.strip(), "python"),
        ])

    if (retrieved_binding_sources_text or "").strip():
        parts.extend([
            "",
            "[Retrieved Shared Binding Sources]",
            retrieved_binding_sources_text.strip(),
        ])

    parts.extend([
        "",
        "Output requirements:",
        "Return these top-level sections in this order:",
        "[State Analysis]",
        "[Recovered Child State]",
        "",
        "[State Analysis] should be short, concrete, and evidence-based.",
        "Use [Caller Recent Steps] as the primary evidence.",
        "Use [Shared State After Previous Call] only as a fallback snapshot.",
        "If recent caller steps and shared state disagree, prefer [Caller Recent Steps].",
        "Recover the actual runtime child-frame input at the paused callsite, not a default, initial, or stale state.",
        "Do not use unevaluated source expressions when a runtime value is already known.",
        "Do not replace a recoverable concrete value with null, unknown, or an empty placeholder.",
        "For string values, preserve the exact runtime string. Keep leading/trailing spaces, blank lines, repeated newlines, escape sequences, and indentation by using Python repr or JSON string notation.",
        "Never rewrite a multiline string as a cleaned paragraph or omit blank lines unless the caller code has already performed that transformation.",
        "",
        "[Recovered Child State] must be natural-language runtime handoff text, not JSON.",
        "Write it as a compact structured note for the next callee frame.",
        "It should explicitly cover, when relevant:",
        "- the callee being entered",
        "- the paused callsite and how its arguments were evaluated at runtime",
        "- the bound receiver/self state",
        "- the explicit arguments passed to the callee",
        "- any shared readable state the callee may use",
        "- any relevant external/object state already prepared by the caller",
        "",
        "Important rules for self/state recovery:",
        "- If this is a method call, preserve the full known receiver/self state, not only the fields directly mentioned at this callsite.",
        "- Do not shrink self to only one attribute just because the current callee appears to read only that attribute.",
        "- If the caller's object already has other known attributes, keep them in the handoff because deeper callees may depend on them.",
        "- Prefer a full, conservative self snapshot over a partial one.",
        "- Preserve object-like state, tuple/set-like values, and external resources as descriptive runtime state rather than forcing them into JSON.",
        "",
        "Formatting guidance for [Recovered Child State]:",
        "- Keep it concise but explicit.",
        "- Use short labeled lines or short bullet-style lines.",
        "- Represent values in valid Python-style notation when helpful.",
        "- For multiline strings, prefer a repr-like value with explicit `\\n` escapes so the callee can recover the exact string.",
        "- It is fine to describe values textually when they are not naturally JSON-like.",
        "",
        "Example:",
        "[State Analysis]",
        "The paused call is `helper(str(x), obj)`.",
        "From [Caller Recent Steps], `x` has already been resolved to `123456`, and `obj` is the current receiver with known attributes `name='demo'` and `cache={'a': 1}`.",
        "So the child frame should receive the runtime string `'123456'` and the full known receiver/object state rather than default placeholders.",
        "",
        "[Recovered Child State]",
        "callee: helper",
        "paused callsite: helper(str(x), obj)",
        "runtime argument recovery: first argument is already resolved to '123456'; second argument is the existing object `obj`.",
        "self/receiver: not a bound-method receiver for helper itself, but the passed object has known state {name='demo', cache={'a': 1}}.",
        "explicit args: s='123456', obj carries the known state above.",
        "shared readable state: none beyond the current caller-visible values.",
    ])

    return "\n".join(parts) + "\n"


def build_completion_state_prompt(
    *,
    unit_labels: Sequence[str],
    codes: Sequence[str],
    frame_name: str,
    frame_trace_text: str,
    completion_result: Mapping[str, Any],
    shared_state: Mapping[str, Any] | None = None,
    caller_pause_at: Mapping[str, Any] | None = None,
    caller_callee_name: str = "",
    class_context_code: str = "",
) -> str:
    code_parts: List[str] = []
    for label, code in zip(unit_labels, codes):
        code_parts.extend([
            f"# Unit: {label}",
            (code or "").strip(),
            "",
        ])

    shared_mapping = dict(shared_state or {})
    runtime_state_text = str(shared_mapping.get("__runtime_state_text", "") or "").strip()

    parts: List[str] = [
        "YOU ARE A PYTHON PROGRAM ANALYST.",
        "You are the state agent for this repository-level execution.",
        "A function/frame has just completed. Infer the faithful shared runtime state before the caller continues.",
        "Prefer a Python-readable text ledger over JSON when Python values are not naturally JSON-like.",
        "",
        "[Completed Frame]",
        str(frame_name or ""),
        "",
        "[Frame Code]",
        _code_block("\n".join(code_parts).strip(), "python"),
        "",
        "[Frame Trace]",
        "```",
        str(frame_trace_text or "").strip(),
        "```",
        "",
        "[Simulated Completion Result]",
        "```text",
        _plain_mapping_lines(dict(completion_result or {})),
        "```",
        "",
        "[Shared Runtime State Before Patch]",
        "```text",
        _plain_mapping_lines(shared_mapping),
        "```",
    ]

    if runtime_state_text:
        parts.extend([
            "",
            "[Current Shared Runtime State Ledger]",
            "```text",
            runtime_state_text,
            "```",
        ])

    if caller_pause_at:
        parts.extend([
            "",
            "[Caller Paused Callsite]",
            "```text",
            _plain_mapping_lines(dict(caller_pause_at or {})),
            "```",
        ])
    if caller_callee_name:
        parts.extend(["", "[Caller Callee Name]", caller_callee_name])
    if class_context_code and class_context_code.strip():
        parts.extend(["", "[Class Context]", _code_block(class_context_code.strip(), "python")])

    parts.extend([
        "",
        "Output requirements:",
        "Return exactly two top-level sections in this order:",
        "[State Analysis]",
        "[Updated Shared Runtime State]",
        "You may add a third section [State Patch JSON] only for simple machine-applicable patches.",
        "",
        "[State Analysis] should be short and evidence-based. Mention only state that survives after the function returns.",
        "",
        "[Updated Shared Runtime State] must be a compact Python-readable text ledger, not JSON.",
        "Use one binding per line or small indented blocks, for example:",
        "_format_cache: {('DATE_INPUT_FORMATS', 'en'): ['%Y-%m-%d']}",
        "obj: {field='value', cache={'a': 1}}",
        "Preserve Python repr for tuple keys, set/tuple/list/dict values, module-like objects, exceptions, and other non-JSON runtime values.",
        "For function objects, preserve runtime metadata that later code may read, especially `__name__`.",
        "If a wrapper is defined under `@functools.wraps(func)`, represent it with the copied metadata, for example `<function compile_func wrapping greet, __name__='greet'>`; do not reduce it to `<function compile_func>`.",
        "For object instances whose class source is available, preserve caller-visible class attributes that later code may read, for example `<SlugConverter object regex='[-a-zA-Z0-9_]+'>`; do not shorten, normalize, or rewrite attribute values to equivalent forms such as `[-\\w]+`.",
        "If existing state remains valid, carry it forward in the ledger instead of dropping it.",
        "",
        "[State Patch JSON], if present, must be exactly one compact valid JSON object with this shape:",
        '{"writes": {"name_or_path": value}, "self": {"field": value}, "return_binding": {"name_or_path": value}}',
        "",
        "Rules:",
        "- Treat the simulated completion result's return value as authoritative.",
        "- Only add durable state that survives after the completed frame returns; do not include local-only temporaries.",
        "- Include only durable state: receiver/self mutations, object/container mutations visible to the caller, module/class/global binding changes, and caller-visible return bindings.",
        "- If the trace says an object, dict, list, set, cache, target, receiver, or module binding was mutated, [Updated Shared Runtime State] must contain the full post-call value that the caller can observe, not only the returned value.",
        "- For mutating helpers that return a status/value while also modifying an argument, preserve both facts: the exact return and the mutated argument/object state.",
        "- Do not invent state just because a value was computed locally.",
        "- Do not put ordinary local assignments like `x = foo(...)` under return_binding; the engine applies those returns itself.",
        "- Use return_binding only when the caller target is nontrivial and caller-visible, such as `obj.attr = foo(...)`, `mapping[key] = foo(...)`, or tuple/list unpacking.",
        "- If a Python value is not naturally JSON-like, keep it accurately in [Updated Shared Runtime State] and omit it from [State Patch JSON].",
        "- If there is no durable state change, repeat the prior durable ledger or write `no durable shared state changes`.",
    ])

    return "\n".join(parts) + "\n"


def build_route_prompt(function_source: str, route_items: Sequence[Mapping[str, Any]]) -> str:
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2:
                return "\n".join(lines[1:-1]).strip()
        return text

    def _normalize_statement(item: Mapping[str, Any]) -> str:
        stmt = str(item.get("stmt", "")).strip()
        if stmt:
            return stmt
        return str(item.get("expr", "")).strip()

    def _normalize_name_list(value: Any) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return []

    def _collect_targets(item: Mapping[str, Any]) -> tuple[str, list[str]]:
        resolved = _normalize_name_list(item.get("resolved"))
        candidate_resolved = _normalize_name_list(item.get("candidate_resolved"))
        if len(resolved) == 1:
            return "resolved", resolved
        if len(resolved) >= 2:
            return "candidates", resolved
        if candidate_resolved:
            return "candidates", candidate_resolved
        return "unresolved", []

    def _collect_source_map(item: Mapping[str, Any], target_names: list[str]) -> dict[str, str]:
        source_map: dict[str, str] = {}
        raw_sources = item.get("callee_sources", None)
        raw_single_source = item.get("callee_source", None)

        if isinstance(raw_sources, dict):
            for name, src in raw_sources.items():
                key = str(name).strip()
                val = str(src or "").strip()
                if key:
                    source_map[key] = val
        elif isinstance(raw_sources, (list, tuple)):
            for name, src in zip(target_names, raw_sources):
                key = str(name).strip()
                val = str(src or "").strip()
                if key:
                    source_map[key] = val
        elif isinstance(raw_single_source, str) and target_names:
            source_map[target_names[0]] = raw_single_source.strip()

        return source_map

    def _group_route_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        group_index: dict[tuple[Any, ...], int] = {}

        for item in items:
            stmt = _normalize_statement(item)
            if not stmt:
                continue

            target_status, target_names = _collect_targets(item)
            source_map = _collect_source_map(item, target_names)

            if target_status == "resolved":
                key = ("resolved", target_names[0])
            elif target_status == "candidates":
                key = ("candidates", tuple(target_names))
            else:
                key = ("unresolved", stmt)

            if key not in group_index:
                group_index[key] = len(groups)
                groups.append(
                    {
                        "status": target_status,
                        "callee": target_names[0] if target_status == "resolved" and target_names else "",
                        "callee_candidates": list(target_names) if target_status == "candidates" else [],
                        "callsites": [],
                    }
                )

            g = groups[group_index[key]]

            source_code: Any = ""
            if target_status == "resolved" and target_names:
                source_code = source_map.get(target_names[0], "")
            elif target_status == "candidates":
                source_code = [{"callee": name, "code": source_map.get(name, "")} for name in target_names]

            if stmt not in {c["stmt"] for c in g["callsites"]}:
                g["callsites"].append({"stmt": stmt, "source": source_code})

        return groups

    def _render_grouped_callsites(groups: Sequence[Mapping[str, Any]]) -> str:
        if not groups:
            return "No route items are provided."

        rendered: list[str] = []
        for idx, g in enumerate(groups, start=1):
            lines: list[str] = [f"[Callee Group {idx}]"]

            status = str(g.get("status", "")).strip()
            if status == "resolved":
                callee = str(g.get("callee", "")).strip() or "<unknown>"
                lines.append(f"Status: resolved")
                lines.append(f"Callee: {callee}")
            elif status == "candidates":
                candidates = [str(x).strip() for x in (g.get("callee_candidates", []) or []) if str(x).strip()]
                lines.append("Status: candidates")
                lines.append("Callee candidates: " + (", ".join(candidates) if candidates else "<none>"))
            else:
                lines.append("Status: unresolved")
                lines.append("Callee: <unresolved>")

            for j, callsite in enumerate(g.get("callsites", []) or [], start=1):
                stmt = str(callsite.get("stmt", "") or "").strip()
                if not stmt:
                    continue

                lines.append("")
                lines.append(f"Callsite {j}:")
                lines.append(stmt)

                source = callsite.get("source")
                if isinstance(source, str):
                    code = source.strip()
                    if code:
                        lines.append("Callee code:")
                        lines.append("```python")
                        lines.append(_strip_code_fence(code))
                        lines.append("```")
                elif isinstance(source, list) and source:
                    for row in source:
                        if not isinstance(row, dict):
                            continue
                        callee = str(row.get("callee", "") or "").strip()
                        code = str(row.get("code", "") or "").strip()
                        lines.append(f"Candidate callee: {callee or '<unknown>'}")
                        if code:
                            lines.append("```python")
                            lines.append(_strip_code_fence(code))
                            lines.append("```")

            rendered.append("\n".join(lines))

        return "\n\n".join(rendered)

    grouped_items = _group_route_items(route_items)
    calls_text = _render_grouped_callsites(grouped_items)

    parts = [
        "YOU ARE A PYTHON PROGRAM ANALYST.",
        "You are the route agent for this repository-level execution.",
        "Decide, for each provided callee group, whether the call should be handled as inline, expand, or unresolved.",
        "Write a short [Route Analysis], then output [Route JSON] as a small JSON object.",
        "",
        "[Current Function]",
        "```python",
        function_source.strip(),
        "```",
        "",
        "[Grouped Callsites]",
        calls_text,
        "",
        "Output requirements:",
        "Return exactly two top-level sections in this order:",
        "[Route Analysis]",
        "[Route JSON]",
        "",
        "[Route Analysis] should be short, concrete, and evidence-based.",
        "For each callee group, explain the decision in 1 or 2 sentences.",
        "",
        "Available actions:",
        "- inline: the callee is short, simple, and can mostly be handled inside the current frame; it should have limited control flow and should not, or only rarely, call other user-defined functions",
        "- expand: the callee is long, complex, stateful, or repeatedly depends on other user-defined functions, so it deserves separate analysis",
        "- unresolved: use this only as a fallback when you truly cannot determine what function is being called or which concrete callee should be analyzed",
        "",
        "Rules:",
        "- If callee code is provided, use it as the main evidence for deciding inline or expand.",
        "- Prefer inline when the callee is a short accessor, setter, constructor-like initializer, tiny container update helper, or a very short branch-free transformation whose source is fully shown.",
        "- Prefer inline for short methods even when they mutate receiver state, as long as the mutation is local and understandable from the provided code.",
        "- Prefer expand when exact printed/returned value depends on a callee with substantial branching/looping, recursion, generator behavior, complex protocol dispatch, or multiple user-defined helper calls.",
        "- Prefer expand when a string/URL/parsing/formatting callee has loops, regex substitutions, replacements, normalization, truncation, entity decoding, or multiple transformation stages.",
        "- Prefer expand when an exact string, digest, URL, regex, numeric array, unit string, datetime value, or encoded/decoded value depends on helper/library semantics that are not fully shown in the current frame.",
        "- Use unresolved only when the callee identity is genuinely unclear even after considering the provided callsite and candidate information.",
        "- If multiple callee candidates are listed, choose the most plausible one and return it in the callee field.",
        "- Some callsites may correspond to builtins, python library calls, or container/object protocol operations. Do not return routing decisions for such callsites.",
        "- Only include callsites in routes when they correspond to user-defined callees that should actually be handled by this routing step.",
        "",
        '[Route JSON] must be exactly one compact valid JSON object of the form:',
        '{"routes":[{"callee":"A._normalize","action":"inline","callsites":["normalized = self._normalize(raw)"]}]}',
        "",
        'Each route item should include:',
        '- "callee": the chosen callee name',
        '- "action": one of "inline", "expand", or "unresolved"',
        '- "callsites": the list of callsite statements covered by that decision',
        '- optional "precision": a short list of exactness risks such as "hash/digest", "url/percent-encoding", "base64/auth", "regex/pattern", "numeric-array", "datetime/units", "encoding/entity", or "symbolic/unit-format"',
        "",
        "Return JSON only inside [Route JSON].",
    ]

    return "\n".join(parts) + "\n"


# -----------------------------
# Search-planning prompts for v2
# -----------------------------


def build_route_search_prompt(
    *,
    function_source: str,
    route_items: Sequence[Mapping[str, Any]],
) -> str:
    slim_items: List[Dict[str, Any]] = []
    for item in route_items:
        if not isinstance(item, Mapping):
            continue
        slim_items.append({
            "stmt": item.get("stmt", ""),
            "expr": item.get("expr", ""),
            "resolved": item.get("resolved", []),
            "candidate_resolved": item.get("candidate_resolved", []),
        })

    parts = [
        "YOU ARE A PYTHON PROGRAM ANALYST.",
        "Decide whether the current callsites can be routed immediately, or whether you need to search for raw callee implementation source first.",
        "If you already have enough evidence, return JSON: {\"decision\": \"direct\", \"routes\": [...]} using the same route schema as the final routing step.",
        "If you need more evidence, return JSON: {\"decision\": \"search\", \"keywords\": [\"kw1\", \"kw2\"]}.",
        "Keywords must be short identifiers or short code phrases only, at most 3 items, no sentences.",
        "Search retrieves only raw function/method implementation source.",
        "Output JSON ONLY.",
        "",
        "[Current Function]",
        "```python",
        function_source.strip(),
        "```",
        "",
        "[Route Items]",
        "```json",
        _j(slim_items),
        "```",
    ]
    return "\n".join(parts)



def build_state_search_prompt(
    *,
    caller_code: str,
    caller_trace_history: Sequence[str] | None,
    pause_at: Mapping[str, Any],
    callee_code: str,
    shared_state: Mapping[str, Any] | None = None,
    class_context_code: str | None = None,
) -> str:
    parts: List[str] = [
        "YOU ARE A PYTHON PROGRAM ANALYST.",
        "Decide whether you can already infer the child-frame packet, or whether you need to search for shared binding assignment source first.",
        "If you already have enough evidence, return JSON: {\"decision\": \"direct\", \"self\": {}, \"args\": {}, \"shared\": {}}.",
        "If you need more evidence, return JSON: {\"decision\": \"search\", \"keywords\": [\"kw1\", \"kw2\"]}.",
        "Keywords must be short identifiers or short code phrases only, at most 3 items, no sentences.",
        "Search retrieves only raw shared-binding assignment source snippets (entry/module/class bindings).",
        "Output JSON ONLY.",
        "",
        "[Caller Code]",
        "```python",
        caller_code.strip(),
        "```",
        "",
        "[Caller Recent Trace]",
        "\n\n".join(str(x) for x in (caller_trace_history or []) if str(x).strip()) or "none",
        "",
        "[Caller Paused Point]",
        _plain_mapping_lines(dict(pause_at or {})),
        "",
        "[Shared State Snapshot]",
        _plain_mapping_lines(dict(shared_state or {})),
        "",
        "[Callee Code]",
        "```python",
        callee_code.strip(),
        "```",
    ]
    if class_context_code and class_context_code.strip():
        parts.extend(["", "[Class Context]", "```python", class_context_code.strip(), "```"])
    return "\n".join(parts) + "\n"



def render_retrieved_binding_sources(search_results: Sequence[Mapping[str, Any]]) -> str:
    if not search_results:
        return "No retrieved binding source snippets.\n"

    parts: List[str] = []
    for row in search_results:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name", "") or "").strip()
        scope = str(row.get("scope", "") or "").strip()
        owner = str(row.get("owner", "") or "").strip()
        file_path = str(row.get("file_path", "") or "").strip()
        code = str(row.get("code", "") or "").rstrip()
        if not code:
            continue
        parts.extend([
            f"- name: {name}",
            f"  scope: {scope}",
            f"  owner: {owner}",
            f"  file: {file_path}",
            "  source:",
            "```python",
            code,
            "```",
            "",
        ])
    return "\n".join(parts).rstrip() + "\n"


__all__ = [
    "build_prompt_normal",
    "build_prompt_continue",
    "build_state_handoff_prompt",
    "build_completion_state_prompt",
    "build_route_prompt",
    "build_route_search_prompt",
    "build_state_search_prompt",
    "render_retrieved_binding_sources",
]
