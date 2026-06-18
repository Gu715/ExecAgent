from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

DATASET_FILES: Dict[str, Path] = {
    "CRUXEval": DATASET_DIR / "cruxeval_o.jsonl",
    "HumanEval": DATASET_DIR / "humaneval.jsonl",
    "LCB-Exec": DATASET_DIR / "livecodebench_execution.jsonl",
    "CodeSense-O": DATASET_DIR / "input_output_dataset_python.jsonl",
}


MODEL_CONFIG: Dict[str, str] = {
    "api_key_env": "YOUR_API_KEY_ENV",
    "base_url": "YOUR_OPENAI_COMPATIBLE_BASE_URL",
}


@dataclass(frozen=True)
class MethodConfig:
    dataset_name: str = "CodeSense-O"
    dataset_path: Path = DATASET_FILES["CodeSense-O"]
    output_dir: Path = Path("clean_result/function_method")
    prompt_dir: Path = Path("clean_prompt/function_method")
    model: str = "YOUR_MODEL_NAME"
    temperature: float = 0.0
    request_timeout_seconds: float = 60.0
    max_retries: int = 5


def get_model_config(model: str) -> Dict[str, str]:
    _ = model
    return MODEL_CONFIG


def with_overrides(
    *,
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    prompt_dir: Optional[str] = None,
    model: Optional[str] = None,
) -> MethodConfig:
    base = MethodConfig()
    resolved_dataset_name = dataset_name or base.dataset_name
    resolved_dataset_path = DATASET_FILES.get(resolved_dataset_name, base.dataset_path)
    if dataset_path:
        resolved_dataset_path = Path(dataset_path)
    return MethodConfig(
        dataset_name=resolved_dataset_name,
        dataset_path=resolved_dataset_path,
        output_dir=Path(output_dir) if output_dir else base.output_dir,
        prompt_dir=Path(prompt_dir) if prompt_dir else base.prompt_dir,
        model=model or base.model,
        temperature=base.temperature,
        request_timeout_seconds=base.request_timeout_seconds,
        max_retries=base.max_retries,
    )
