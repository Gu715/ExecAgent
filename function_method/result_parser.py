from __future__ import annotations

import ast
from typing import Any, Optional, Tuple


def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    text = "" if text is None else str(text)
    start = text.find(start_tag)
    if start < 0:
        return None
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end < 0:
        return None
    return text[start:end]


def extract_result_text(raw_text: str) -> Optional[str]:
    result = extract_between(raw_text, "[RESULT]", "[/RESULT]")
    if result is None:
        return None
    result = result.replace("\r\n", "\n").replace("\r", "\n")
    if result.startswith("\n"):
        result = result[1:]
    if result.endswith("\n"):
        result = result[:-1]
    return result


def _try_literal(value: Any) -> Tuple[bool, Any]:
    if isinstance(value, (int, float, bool, list, tuple, dict, set)) or value is None:
        return True, value
    if not isinstance(value, str) or not value.strip():
        return False, value
    try:
        return True, ast.literal_eval(value.strip())
    except Exception:
        return False, value


def outputs_equal(predicted: Any, expected: Any) -> bool:
    pred_ok, pred_val = _try_literal(predicted)
    gold_ok, gold_val = _try_literal(expected)
    if pred_ok and gold_ok:
        return pred_val == gold_val

    pred_text = "" if predicted is None else str(predicted).replace("\r\n", "\n").replace("\r", "\n")
    gold_text = "" if expected is None else str(expected).replace("\r\n", "\n").replace("\r", "\n")
    if pred_text == gold_text:
        return True
    if pred_ok and isinstance(pred_val, str) and pred_val == gold_text:
        return True
    if gold_ok and isinstance(gold_val, str) and pred_text == gold_val:
        return True
    return False

