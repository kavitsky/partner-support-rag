"""OpenRouter chat client: strict JSON-schema output, backoff on transient errors, on-disk response cache."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "llm"

load_dotenv(ROOT / ".env")


@dataclass
class Completion:
    content: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, model: str, prices: tuple[float, float], openrouter_cfg: dict, temperature: float = 0.0, use_cache: bool = True):
        self.model = model
        self.prices = prices
        self.temperature = temperature
        self.use_cache = use_cache
        self.max_retries = openrouter_cfg["max_retries"]
        key = os.getenv(openrouter_cfg["api_key_env"])
        if not key:
            raise LLMError(f"{openrouter_cfg['api_key_env']} is not set: copy .env.example to .env and add your key")
        self._client = openai.OpenAI(base_url=openrouter_cfg["base_url"], api_key=key, timeout=openrouter_cfg["timeout_s"], max_retries=0)

    def cost(self, c: Completion) -> float:
        return (c.input_tokens * self.prices[0] + c.output_tokens * self.prices[1]) / 1_000_000

    def complete(self, messages: list[dict], schema: dict, schema_name: str) -> Completion:
        key = hashlib.sha256(
            json.dumps([self.model, self.temperature, messages, schema], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        if self.use_cache and path.exists():
            return Completion(**{**json.loads(path.read_text()), "cached": True})

        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
                    extra_body={"provider": {"require_parameters": True}},
                )
                if not r.choices:
                    raise LLMError("empty choices")
                break
            except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError, LLMError):
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 20) + random.random())
        usage = r.usage
        comp = Completion(
            r.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            round(time.perf_counter() - start, 3),
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: v for k, v in asdict(comp).items() if k != "cached"}, ensure_ascii=False))
        return comp
