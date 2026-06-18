from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .cfg_utils import build_method_inputs
from .config import MethodConfig
from .dataset import FunctionSample, load_jsonl_dataset, select_samples
from .llm import call_llm
from .prompt import build_cfg_guided_prompt
from .result_parser import extract_result_text


def _safe_sample_path(sample_id: str) -> Path:
    return Path(*[part for part in str(sample_id).replace("\\", "/").split("/") if part])


def sample_output_dir(config: MethodConfig, sample: FunctionSample) -> Path:
    return config.output_dir / _safe_sample_path(sample.sample_id)


def prompt_output_path(config: MethodConfig, sample: FunctionSample) -> Path:
    return config.prompt_dir / _safe_sample_path(sample.sample_id) / "prompt.txt"


def response_output_path(config: MethodConfig, sample: FunctionSample) -> Path:
    return sample_output_dir(config, sample) / "response.txt"


def summary_output_path(config: MethodConfig, sample: FunctionSample) -> Path:
    return sample_output_dir(config, sample) / "summary.json"


def has_completed(config: MethodConfig, sample: FunctionSample) -> bool:
    path = response_output_path(config, sample)
    return path.exists() and path.is_file() and path.read_text(encoding="utf-8") != ""


async def run_sample(sample: FunctionSample, config: MethodConfig, *, force: bool = False) -> dict:
    if not force and has_completed(config, sample):
        raw_text = response_output_path(config, sample).read_text(encoding="utf-8")
    else:
        code, cfg_text = build_method_inputs(sample.code)
        prompt = build_cfg_guided_prompt(
            code=code,
            cfg_text=cfg_text,
            input_text=sample.input_text,
        )

        prompt_path = prompt_output_path(config, sample)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        raw_text = await call_llm(prompt, config)
        response_path = response_output_path(config, sample)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(raw_text, encoding="utf-8")

    answer = extract_result_text(raw_text)
    summary = {
        "sample": asdict(sample),
        "model": config.model,
        "answer": answer,
        "has_answer": answer is not None and answer != "",
    }

    summary_path = summary_output_path(config, sample)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def run_dataset(
    config: MethodConfig,
    *,
    target: Optional[str] = None,
    limit: Optional[int] = None,
    max_concurrency: int = 20,
    force: bool = False,
) -> list[dict]:
    samples = select_samples(load_jsonl_dataset(config.dataset_path), target=target, limit=limit)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(sample: FunctionSample) -> dict:
        async with semaphore:
            return await run_sample(sample, config, force=force)

    return await asyncio.gather(*(run_one(sample) for sample in samples))

