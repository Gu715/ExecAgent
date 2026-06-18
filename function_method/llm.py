from __future__ import annotations

import asyncio
import os
from typing import Optional

from openai import APITimeoutError, AsyncOpenAI

from .config import MethodConfig, get_model_config


def create_client(config: MethodConfig) -> AsyncOpenAI:
    model_config = get_model_config(config.model)
    api_key_env = model_config["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key. Please set environment variable: {api_key_env}")
    return AsyncOpenAI(api_key=api_key, base_url=model_config["base_url"])


async def call_llm(prompt: str, config: MethodConfig) -> str:
    retry_idx = 0
    client: Optional[AsyncOpenAI] = None

    try:
        while True:
            if client is None:
                client = create_client(config)
            try:
                response = await client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                    timeout=config.request_timeout_seconds,
                )
                return response.choices[0].message.content or ""
            except (asyncio.TimeoutError, TimeoutError, APITimeoutError):
                return ""
            except Exception:
                retry_idx += 1
                if retry_idx > config.max_retries:
                    raise
                if client is not None:
                    await client.close()
                    client = None
                await asyncio.sleep(min(2.0 * retry_idx, 10.0))
    finally:
        if client is not None:
            await client.close()

