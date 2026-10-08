"""Observe final SDK create calls with request-owned clients and hash-only data."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from hashlib import sha256

from .budget import ESTIMATE_VERSION, profile_key, text_tokens

logger = logging.getLogger(__name__)


def hashed(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def request_metadata(kwargs, config, image_reserve):
    body = {k: v for k, v in kwargs.items() if type(v).__name__ != "NotGiven"}
    extra = body.pop("extra_body", None)
    if isinstance(extra, dict):
        body.update(extra)
    messages = body.get("messages", [])
    images = []

    def trim(value):
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                images.append(hashed(value))
                return {"type": "image_url"}
            return {k: trim(v) for k, v in value.items()}
        if isinstance(value, list):
            return [trim(v) for v in value]
        return value

    raw_text = text_tokens(trim(messages)) + text_tokens(body.get("tools", [])) + 120 * len(messages)
    system = [m for m in messages if m.get("role") in {"system", "developer"}]
    message_hashes = [hashed(m) for m in messages]
    # Only known generation parameters; headers, query options and credentials
    # are never included in a stored diagnostic, including their hashes.
    parameters = {
        k: body[k]
        for k in (
            "model",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
            "tool_choice",
        )
        if k in body
    }
    return {
        "profile": profile_key(config, body.get("model")),
        "provider": config.get("id"),
        "model": body.get("model"),
        "estimate_version": ESTIMATE_VERSION,
        "system": hashed(system),
        "tools": hashed(body.get("tools", [])),
        "parameters": hashed(parameters),
        "messages": message_hashes,
        "stable": message_hashes[:-1],
        "request": hashed([message_hashes, hashed(body.get("tools", [])), hashed(parameters)]),
        "images": len(images),
        "image_hashes": images,
        "raw_text_tokens": raw_text,
        "raw_image_tokens": len(images) * image_reserve,
    }


def _usage(response):
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    output = getattr(usage, "completion_tokens", None)
    if not isinstance(prompt, int) or not isinstance(cached, int) or not 0 <= cached <= prompt:
        return {"input_tokens": None, "cached_tokens": None, "output_tokens": None}
    return {"input_tokens": prompt, "cached_tokens": cached, "output_tokens": output}


class WireObserver:
    def __init__(self, record, image_reserve=1600):
        self.record = record
        self.image_reserve = image_reserve
        self.attempts = 0
        self.completed = 0

    async def _record(self, payload):
        try:
            await self.record(payload)
        except Exception as exc:
            logger.warning("[NativeMM] SDK 请求诊断失败 type=%s", type(exc).__name__)

    def wrap(self, create, config):
        async def observed(**kwargs):
            self.attempts += 1
            payload = request_metadata(kwargs, config, self.image_reserve)
            extra = kwargs.get("extra_body") or {}
            messages = extra.get("messages", kwargs.get("messages", []))
            last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
            continuation = any(m.get("role") == "tool" for m in messages[last_user + 1 :])
            payload.update(
                attempt=self.attempts,
                call=self.completed + 1,
                stage="tool" if continuation else "first",
                boundary="sdk_create",
            )
            started = time.monotonic()
            try:
                response = await create(**kwargs)
            except BaseException as exc:
                payload.update(
                    status="aborted" if isinstance(exc, asyncio.CancelledError) else "error",
                    error_type=type(exc).__name__,
                    elapsed=time.monotonic() - started,
                )
                await self._record(payload)
                raise
            self.completed += 1
            payload.update(status="completed", elapsed=time.monotonic() - started, **_usage(response))
            await self._record(payload)
            return response

        return observed
