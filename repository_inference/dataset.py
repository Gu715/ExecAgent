from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DATASET_ROOT


DATASET_NAME = "PyRepoExec"


def _flat_name_to_logical_path(filename: str) -> str:
    return filename.replace("@@", "/")


def _task_sort_key(task_path: Path):
    match = re.match(r"task(\d+)$", task_path.name)
    if match:
        return int(match.group(1))
    return task_path.name


def _load_slice_files(files_dir: Path) -> Dict[str, str]:
    files: Dict[str, str] = {}
    if not files_dir.exists() or not files_dir.is_dir():
        return files

    for path in sorted(files_dir.iterdir()):
        if path.is_file() and path.suffix == ".py":
            files[_flat_name_to_logical_path(path.name)] = path.read_text(encoding="utf-8")
    return files


def _load_cases(task_dir: Path) -> List[Dict[str, Any]]:
    task_json = task_dir / "task.json"
    if task_json.exists() and task_json.is_file():
        try:
            obj = json.loads(task_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        cases = obj.get("cases", [])
        if not isinstance(cases, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for case in cases:
            if not isinstance(case, dict):
                continue
            entry_script = str(case.get("entry", "") or "").strip()
            if entry_script:
                normalized.append({
                    "entry": entry_script,
                    "expected_output": case.get("expected_output"),
                })
        return normalized

    entry_file = task_dir / "entry.py"
    if not entry_file.exists() or not entry_file.is_file():
        return []

    entry_script = entry_file.read_text(encoding="utf-8").strip()
    if not entry_script:
        return []

    expected_output_file = task_dir / "expected_output.txt"
    expected_output: Any = None
    if expected_output_file.exists() and expected_output_file.is_file():
        expected_output = expected_output_file.read_text(encoding="utf-8").strip()

    return [{"entry": entry_script, "expected_output": expected_output}]


def _build_samples(project_dir: Path, task_dir: Path) -> List[dict]:
    files = _load_slice_files(task_dir / "slicing")
    if not files:
        return []

    cases = _load_cases(task_dir)
    if not cases:
        return []

    task_id = f"{project_dir.name}/{task_dir.name}"
    samples: List[dict] = []

    for idx, case in enumerate(cases):
        sample_id = task_id if len(cases) == 1 else f"{task_id}/case{idx}"
        samples.append({
            "id": sample_id,
            "dataset": DATASET_NAME,
            "task_id": task_id,
            "case_id": idx,
            "files": files,
            "entry_script": case["entry"],
            "expected_output": case.get("expected_output"),
        })

    return samples


def iter_dataset_samples(dataset_root: str | Path = DATASET_ROOT) -> List[dict]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"Dataset directory does not exist: {dataset_root}")
        return []

    samples: List[dict] = []
    project_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    for project_dir in project_dirs:
        task_dirs = sorted(
            [p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith("task")],
            key=_task_sort_key,
        )
        for task_dir in task_dirs:
            samples.extend(_build_samples(project_dir, task_dir))

    return samples


def read_sample(sample_id: str, dataset_root: str | Path = DATASET_ROOT) -> Optional[dict]:
    for sample in iter_dataset_samples(dataset_root):
        if sample["id"] == sample_id or sample.get("task_id") == sample_id:
            return sample
    return None

