from __future__ import annotations

import argparse
import asyncio

from .config import with_overrides
from .evaluate import evaluate_dataset
from .pipeline import run_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean CFG-guided function-level method.")
    parser.add_argument("--dataset-name", help="Dataset key configured in function_method/config.py.")
    parser.add_argument("--dataset", help="Path to a jsonl dataset. Overrides --dataset-name.")
    parser.add_argument("--output-dir", help="Directory for responses and summaries.")
    parser.add_argument("--prompt-dir", help="Directory for generated prompts.")
    parser.add_argument("--model", help="Model name configured in function_method/config.py.")
    parser.add_argument("--target", help="Run one sample_id or task_id.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected samples.")
    parser.add_argument("--max-concurrency", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Regenerate existing responses.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate saved summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_overrides(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        prompt_dir=args.prompt_dir,
        model=args.model,
    )

    if not args.eval_only:
        asyncio.run(
            run_dataset(
                config,
                target=args.target,
                limit=args.limit,
                max_concurrency=args.max_concurrency,
                force=args.force,
            )
        )

    summary = evaluate_dataset(config, target=args.target, limit=args.limit)
    print(summary)


if __name__ == "__main__":
    main()
