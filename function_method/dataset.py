from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class FunctionSample:
    sample_id: str
    task_id: str
    case_id: int
    code: str
    input_text: str
    expected_output: Any = None


def _normalize_record(obj: dict, fallback_idx: int) -> Optional[FunctionSample]:
    code = str(obj.get("code", "") or "")
    if not code.strip():
        return None

    raw_id = obj.get("id")
    idx = obj.get("idx", obj.get("index", fallback_idx))
    sample_id = str(raw_id) if raw_id not in (None, "") else f"sample_{idx}"

    return FunctionSample(
        sample_id=sample_id,
        task_id=str(obj.get("task_id") or sample_id),
        case_id=int(obj.get("case_id", 0) or 0),
        code=code,
        input_text="" if obj.get("input") is None else str(obj.get("input")),
        expected_output=obj.get("output"),
    )


def load_jsonl_dataset(path: str | Path) -> list[FunctionSample]:
    path = Path(path)
    samples: list[FunctionSample] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                continue

            sample = _normalize_record(obj, fallback_idx=line_no - 1)
            if sample is not None:
                samples.append(sample)

    return samples


def select_samples(
    samples: Iterable[FunctionSample],
    target: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[FunctionSample]:
    selected = [
        sample for sample in samples
        if not target or sample.sample_id == target or sample.task_id == target
    ]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected

