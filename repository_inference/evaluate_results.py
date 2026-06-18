from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import with_overrides
from .dataset import iter_dataset_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate repository-level inference summaries.")
    parser.add_argument("--dataset-root", help="Path to the PyRepoExec dataset directory.")
    parser.add_argument("--output-dir", required=True, help="Directory containing inference summary.json files.")
    parser.add_argument("--project", help="Evaluate one project, for example: django.")
    parser.add_argument("--task", help="Evaluate one task. Use project/task, or combine with --project and pass task name.")
    parser.add_argument("--show-wrong", action="store_true", help="Print wrong or missing sample ids.")
    return parser.parse_args()


def normalize_output(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return [str(value).strip()]


def summary_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir.joinpath(*sample_id.split("/"), "summary.json")


def load_summary(output_dir: Path, sample_id: str) -> dict[str, Any] | None:
    path = summary_path(output_dir, sample_id)
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def sample_matches_filter(sample: dict[str, Any], project: str, task: str) -> bool:
    sample_id = str(sample["id"])
    parts = sample_id.split("/")
    if project and (not parts or parts[0] != project):
        return False
    if task:
        task_id = task if "/" in task or not project else f"{project}/{task}"
        if sample.get("task_id") != task_id and sample_id != task_id and not sample_id.startswith(f"{task_id}/"):
            return False
    return True


def evaluate_sample(sample: dict[str, Any], output_dir: Path) -> tuple[bool, str]:
    summary = load_summary(output_dir, str(sample["id"]))
    if summary is None:
        return False, "missing_summary"
    if summary.get("success") is not True:
        return False, str(summary.get("error_message") or "failed")
    pred = normalize_output(summary.get("final_output", []))
    gold = normalize_output(sample.get("expected_output", []))
    if pred != gold:
        return False, "wrong_output"
    return True, "ok"


def main() -> None:
    args = parse_args()
    config = with_overrides(dataset_root=args.dataset_root, output_dir=args.output_dir)
    output_dir = Path(args.output_dir)
    samples = [
        sample
        for sample in iter_dataset_samples(config.dataset_root)
        if sample_matches_filter(sample, args.project or "", args.task or "")
    ]

    total = len(samples)
    correct = 0
    wrong_rows: list[tuple[str, str]] = []
    by_project: dict[str, list[int]] = {}

    for sample in samples:
        sample_id = str(sample["id"])
        ok, reason = evaluate_sample(sample, output_dir)
        project = sample_id.split("/", 1)[0]
        stat = by_project.setdefault(project, [0, 0])
        stat[0] += 1
        if ok:
            correct += 1
            stat[1] += 1
        else:
            wrong_rows.append((sample_id, reason))

    accuracy = (correct / total) if total else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Wrong/Missing: {total - correct}")
    print(f"Accuracy: {accuracy:.2%}")

    if len(by_project) > 1:
        print("")
        print("By project:")
        for project in sorted(by_project):
            project_total, project_correct = by_project[project]
            project_acc = (project_correct / project_total) if project_total else 0.0
            print(f"  {project}: {project_correct}/{project_total} ({project_acc:.2%})")

    if args.show_wrong and wrong_rows:
        print("")
        print("Wrong or missing samples:")
        for sample_id, reason in wrong_rows:
            print(f"  {sample_id}: {reason}")


if __name__ == "__main__":
    main()
