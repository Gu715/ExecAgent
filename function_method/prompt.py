from __future__ import annotations


TRACE_RULES = """
Return exactly two top-level sections in this order:
[TRACE]
<Your simulated execution trace>
[/TRACE]

[RESULT]<the exact final result>[/RESULT]

Your trace should be a compact block-by-block execution trace in natural language.
For each block, include the current CFG block, the statement or event being executed,
the relevant state after execution, and the next control-flow destination when useful.
""".strip()


def _code_block(text: str, lang: str = "") -> str:
    fence = f"```{lang}" if lang else "```"
    return f"{fence}\n{text.rstrip()}\n```"


def build_cfg_guided_prompt(*, code: str, cfg_text: str, input_text: str) -> str:
    return "\n".join([
        "YOU ARE A PYTHON PROGRAM ANALYST.",
        "Your job is to simulate execution using the given source code and CFG.",
        "Derive the exact final answer for the concrete input.",
        "",
        "[Code]",
        _code_block((code or "").strip(), "python"),
        "",
        "[CFG]",
        _code_block((cfg_text or "CFG not provided.").strip()),
        "",
        "[Input]",
        _code_block((input_text or "").strip()),
        "",
        "[Output requirements]",
        TRACE_RULES,
        "",
        "The result must be the exact Python value representation of the final result.",
    ])

