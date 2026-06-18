from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "PyRepoExec"


MODEL_CONFIG: Dict[str, str] = {
    "api_key_env": "DEEPSEEK_API_KEY",
    "base_url": "https://api.deepseek.com",
}


@dataclass(frozen=True)
class RepoInferenceConfig:
    dataset_name: str = "PyRepoExec"
    dataset_root: Path = DATASET_ROOT
    output_dir: Path = Path("repo_inference_result")
    model: str = "deepseek-v4-flash"
    max_rounds: int = 150
    max_retries: int = 5
    max_workers: int = 20


def load_project_env(env_path: Path = PROJECT_ROOT / ".env") -> None:
    if not env_path.exists() or not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def with_overrides(
    *,
    dataset_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> RepoInferenceConfig:
    base = RepoInferenceConfig()
    return RepoInferenceConfig(
        dataset_name=base.dataset_name,
        dataset_root=Path(dataset_root) if dataset_root else base.dataset_root,
        output_dir=Path(output_dir) if output_dir else base.output_dir,
        model=model or base.model,
        max_rounds=base.max_rounds,
        max_retries=base.max_retries,
        max_workers=max_workers if max_workers is not None else base.max_workers,
    )
