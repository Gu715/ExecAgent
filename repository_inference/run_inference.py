from __future__ import annotations

import argparse
import os

from .config import MODEL_CONFIG, RepoInferenceConfig, load_project_env, with_overrides
from .dataset import iter_dataset_samples
from .engine import InferenceConfig, _run_samples_parallel, run_single_sample_by_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository-level inference on PyRepoExec.")
    parser.add_argument("--dataset-root", help="Path to the PyRepoExec dataset directory.")
    parser.add_argument("--output-dir", help="Directory for inference traces and summaries.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument("--project", help="Run all samples from one project, for example: django.")
    parser.add_argument("--task", help="Run one task. Use project/task, or combine with --project and pass task name.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--sequential", action="store_true")
    return parser.parse_args()


def build_engine_config(config: RepoInferenceConfig) -> InferenceConfig:
    load_project_env()
    api_key = os.environ.get(MODEL_CONFIG["api_key_env"])
    return InferenceConfig(
        max_rounds=config.max_rounds,
        output_dir=str(config.output_dir),
        model=config.model,
        api_key=api_key,
        base_url=MODEL_CONFIG["base_url"],
        max_retries=config.max_retries,
    )


def main() -> None:
    args = parse_args()
    config = with_overrides(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model=args.model,
        max_workers=args.max_workers,
    )

    samples = iter_dataset_samples(config.dataset_root)
    if args.project:
        samples = [s for s in samples if s["id"].split("/", 1)[0] == args.project]
    if args.task:
        task_id = args.task if "/" in args.task or not args.project else f"{args.project}/{args.task}"
        samples = [s for s in samples if s.get("task_id") == task_id or s["id"] == task_id or s["id"].startswith(f"{task_id}/")]

    if not samples:
        print("No samples to run.")
        return

    engine_config = build_engine_config(config)

    if args.sequential:
        for idx, sample in enumerate(samples, start=1):
            result = run_single_sample_by_data(sample, engine_config)
            status = "success" if result.success else "failed"
            print(f"[{idx}/{len(samples)}] {sample['id']}: {status}")
    else:
        _run_samples_parallel(samples, engine_config, config.max_workers)


if __name__ == "__main__":
    main()
