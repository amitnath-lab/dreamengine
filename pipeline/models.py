"""
Unified model client – sync + async – talks to AWS Bedrock and Anthropic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import httpx

from .config import ModelSpec, ModelTier, PipelineConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trace logging — writes every call to a JSONL file for inspection
# ---------------------------------------------------------------------------

_trace_path: str | None = None


def set_trace_path(path: str) -> None:
    global _trace_path
    _trace_path = path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Write a header entry so the file is easy to identify
    _write_trace({"_event": "session_start", "trace_file": path})


def _write_trace(entry: dict) -> None:
    if not _trace_path:
        return
    entry.setdefault("ts", time.time())
    try:
        with open(_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never let tracing break the pipeline


def _trace_call(
    *,
    provider: str,
    model: str,
    tier: str,
    system: str,
    user: str,
    response: str,
    duration_s: float,
    context: str = "",
) -> None:
    _write_trace({
        "provider": provider,
        "model": model,
        "tier": tier,
        "context": context,
        "system_chars": len(system),
        "user_chars": len(user),
        "response_chars": len(response),
        "duration_s": round(duration_s, 3),
        "system": system,
        "user": user,
        "response": response,
    })


# ---------------------------------------------------------------------------
# Bedrock client helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_bedrock_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def _bedrock_chat(
    model: str, system: str, user_prompt: str, *,
    region: str, temperature: float = 0.4, max_tokens: int = 4096,
) -> str:
    client = _get_bedrock_client(region)
    resp = client.converse(
        modelId=model,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
    )
    output = resp.get("output", {})
    message = output.get("message", {})
    parts = message.get("content", [])
    return "".join(p.get("text", "") for p in parts)


async def _bedrock_chat_async(
    model: str, system: str, user_prompt: str, *,
    region: str, temperature: float = 0.4, max_tokens: int = 4096,
) -> str:
    return await asyncio.to_thread(
        _bedrock_chat, model, system, user_prompt,
        region=region, temperature=temperature, max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Anthropic clients (sync + async)
# ---------------------------------------------------------------------------

def _anthropic_chat(
    model: str, system: str, user_prompt: str, *,
    api_key: str, temperature: float = 0.3, max_tokens: int = 4096,
) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "system": system, "messages": [{"role": "user", "content": user_prompt}],
    }
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=payload, timeout=180.0,
    )
    resp.raise_for_status()
    blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


async def _anthropic_chat_async(
    model: str, system: str, user_prompt: str, *,
    api_key: str, temperature: float = 0.3, max_tokens: int = 4096,
) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "system": system, "messages": [{"role": "user", "content": user_prompt}],
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload,
        )
    resp.raise_for_status()
    blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


# ---------------------------------------------------------------------------
# Unified dispatch (sync + async)
# ---------------------------------------------------------------------------

def call_model(
    spec: ModelSpec, system_prompt: str, user_prompt: str,
    config: PipelineConfig, *, temperature: float | None = None,
    max_tokens: int = 4096, context: str = "",
) -> str:
    temp = temperature if temperature is not None else (
        0.3 if spec.tier == ModelTier.PAID_SUPERVISOR else 0.4
    )
    log.info(f"  [sync] {spec.provider}:{spec.name} ({spec.tier.value})")
    t0 = time.monotonic()
    if spec.provider == "bedrock":
        result = _bedrock_chat(spec.name, system_prompt, user_prompt,
                               region=config.aws_bedrock_region, temperature=temp, max_tokens=max_tokens)
    elif spec.provider == "anthropic":
        result = _anthropic_chat(spec.name, system_prompt, user_prompt,
                                 api_key=config.anthropic_api_key, temperature=temp, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider: {spec.provider}")
    _trace_call(provider=spec.provider, model=spec.name, tier=spec.tier.value,
                system=system_prompt, user=user_prompt, response=result,
                duration_s=time.monotonic() - t0, context=context)
    return result


async def call_model_async(
    spec: ModelSpec, system_prompt: str, user_prompt: str,
    config: PipelineConfig, *, temperature: float | None = None,
    max_tokens: int = 4096, context: str = "",
) -> str:
    temp = temperature if temperature is not None else (
        0.3 if spec.tier == ModelTier.PAID_SUPERVISOR else 0.4
    )
    log.info(f"  [async] {spec.provider}:{spec.name} ({spec.tier.value})")
    t0 = time.monotonic()
    if spec.provider == "bedrock":
        result = await _bedrock_chat_async(spec.name, system_prompt, user_prompt,
                                           region=config.aws_bedrock_region, temperature=temp, max_tokens=max_tokens)
    elif spec.provider == "anthropic":
        result = await _anthropic_chat_async(spec.name, system_prompt, user_prompt,
                                             api_key=config.anthropic_api_key, temperature=temp, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider: {spec.provider}")
    _trace_call(provider=spec.provider, model=spec.name, tier=spec.tier.value,
                system=system_prompt, user=user_prompt, response=result,
                duration_s=time.monotonic() - t0, context=context)
    return result


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def pick_model(
    task_type: str, strength_map: dict[str, list[str]],
    models: list[ModelSpec], max_tier: ModelTier = ModelTier.FREE_HEAVY,
) -> ModelSpec:
    wanted = set(strength_map.get(task_type, ["general"]))
    tier_order = list(ModelTier)
    candidates = [m for m in models if tier_order.index(m.tier) <= tier_order.index(max_tier)]
    candidates.sort(key=lambda m: tier_order.index(m.tier))
    best: ModelSpec | None = None
    best_score = -1
    for m in candidates:
        score = len(wanted & set(m.strengths))
        if score > best_score:
            best_score = score
            best = m
    return best or candidates[0]


def get_supervisor_model(models: list[ModelSpec]) -> ModelSpec:
    for m in models:
        if m.tier == ModelTier.PAID_SUPERVISOR:
            return m
    raise RuntimeError("No supervisor model configured")
