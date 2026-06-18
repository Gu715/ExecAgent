from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from .config import MethodConfig
from .dataset import FunctionSample, load_jsonl_dataset, select_samples
from .pipeline import summary_output_path
from .result_parser import outputs_equal


def _load_answer(config: MethodConfig, sample: FunctionSample) -> tuple[Any, str]:
    path = summary_output_path(config, sample)
    if not path.exists():
        return None, "missing_summary"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "bad_summary"
    return payload.get("answer"), "ok"


def evaluate_dataset(
    config: MethodConfig,
    *,
    target: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    samples = select_samples(load_jsonl_dataset(config.dataset_path), target=target, limit=limit)
    rows: list[dict] = []
    correct = 0

    for sample in samples:
        pred, status = _load_answer(config, sample)
        ok = status == "ok" and outputs_equal(pred, sample.expected_output)
        correct += int(ok)
        rows.append({
            "sample_id": sample.sample_id,
            "task_id": sample.task_id,
            "case_id": sample.case_id,
            "correct": int(ok),
            "status": "ok" if ok else status if status != "ok" else "wrong",
            "pred": json.dumps(pred, ensure_ascii=False),
            "gold": json.dumps(sample.expected_output, ensure_ascii=False),
        })

    total = len(samples)
    summary = {
        "model": config.model,
        "dataset_path": str(config.dataset_path),
        "total_cases": total,
        "correct_cases": correct,
        "case_accuracy": correct / total if total else 0.0,
    }

    out_dir = config.output_dir / "__accuracy__"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "case_accuracy.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "task_id", "case_id", "correct", "status", "pred", "gold"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

