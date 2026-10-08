"""Request-local token estimates learned from completed first model calls.

These are budget estimates, not a tokenizer. Images retain their original
reserve; image observations may only increase it. No prompt text is retained.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256

ESTIMATE_VERSION = 1
MIN_SAMPLES = 8
SAFETY_MARGIN = 1.2


def text_tokens(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return math.ceil(sum(2 if ord(c) > 127 else 1 / 3 for c in text))


def profile_key(config, model=None):
    # Endpoint is hashed too: it may contain private routing information.
    value = [config.get("id"), config.get("api_base"), model or config.get("model")]
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class BudgetEstimator:
    text_scale: float = 1.0
    image_scale: float = 1.0
    text_samples: int = 0
    image_samples: int = 0

    def text(self, value, overhead=0):
        return math.ceil((text_tokens(value) + overhead) * self.text_scale)

    def image(self, reserve):
        return math.ceil(reserve * self.image_scale)


def _upper(values):
    # Maximum of the bounded recent window avoids discarding rare underestimates.
    return max(values) * SAFETY_MARGIN


def calibrated(config, samples, model=None):
    # The deployed GLM single-call audit found old estimates >= 1.55x
    # actual usage. 0.75 keeps >=16% headroom until live samples are sufficient.
    model = model or config.get("model", "")
    bootstrap = 0.75 if model == "glm-5.3-flash" else 1.0
    valid = [
        s
        for s in samples
        if s.get("estimate_version") == ESTIMATE_VERSION
        and s.get("status") == "completed"
        and s.get("stage") == "first"
        and isinstance(s.get("input_tokens"), int)
        and s["input_tokens"] >= 512
        and isinstance(s.get("raw_text_tokens"), int)
        and s["raw_text_tokens"] >= 512
    ]
    text = [s["input_tokens"] / s["raw_text_tokens"] for s in valid if not s.get("images")]
    scale = max(0.55, _upper(text)) if len(text) >= MIN_SAMPLES else bootstrap
    # Increase immediately when even one observed call exceeds the estimate.
    if text:
        scale = max(scale, _upper(text))
    images = [
        max(0, s["input_tokens"] - scale * s["raw_text_tokens"]) / s["raw_image_tokens"]
        for s in valid
        if s.get("images") and s.get("raw_image_tokens", 0) > 0
    ]
    image_scale = max(1.0, _upper(images)) if images else 1.0
    return BudgetEstimator(scale, image_scale, len(text), len(images))
