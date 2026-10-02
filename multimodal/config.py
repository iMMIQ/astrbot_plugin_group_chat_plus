"""Validate configuration once at the boundary; preserve v2.0 behavior on upgrade."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

SCHEMA = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
CONFIG_VERSION = 2
PROBABILITIES = {
    "auto_candidate_probability",
    "poke_reverse_on_poke_probability",
    "poke_after_reply_probability",
}
NONNEGATIVE = {
    "background_images",
    "mention_wait_seconds",
    "media_wait_seconds",
    "auto_reply_cooldown",
    "poke_after_reply_delay",
}


def migrate_config(raw):
    result = copy.deepcopy(dict(raw))
    version = result.get("config_version", 1)
    if type(version) is not int or version > CONFIG_VERSION:
        raise ValueError("不支持此配置版本")
    # Existing v2.0 settings take priority over defaults, especially participation/pokes.
    result["config_version"] = CONFIG_VERSION
    return result


class Settings(dict):
    def __init__(self, raw):
        super().__init__()
        self.update(migrate_config(raw))

    def update(self, other=(), **kwargs):
        values = {key: copy.deepcopy(rule["default"]) for key, rule in SCHEMA.items()}
        values.update(self)
        values.update(dict(other, **kwargs))
        for key, rule in SCHEMA.items():
            value = values[key]
            kind = rule["type"]
            valid = (
                type(value) is bool
                if kind == "bool"
                else type(value) is int
                if kind == "int"
                else type(value) in (int, float) and math.isfinite(value)
                if kind == "float"
                else isinstance(value, str)
                if kind == "string"
                else isinstance(value, list)
            )
            if not valid:
                raise ValueError(f"配置 {key} 类型不正确")
            if "options" in rule and value not in rule["options"]:
                raise ValueError(f"配置 {key} 取值不正确")
            if key in PROBABILITIES and not 0 <= value <= 1:
                raise ValueError(f"配置 {key} 必须在 0 到 1 之间")
            if kind in {"int", "float"} and key not in PROBABILITIES:
                if value < 0 or (value == 0 and key not in NONNEGATIVE):
                    raise ValueError(f"配置 {key} 超出允许范围")
        if values["background_images"] > values["max_images"]:
            raise ValueError("背景图片上限不能大于总图片上限")
        if values["input_token_budget"] < 256:
            raise ValueError("输入预算不能小于 256")
        super().update(values)
