"""Validate configuration once at the boundary; preserve v2.0 behavior on upgrade."""

from __future__ import annotations

import copy
import json
import math
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
CONFIG_VERSION = 2
PROBABILITIES = {
    "auto_candidate_probability",
    "poke_reverse_on_poke_probability",
    "poke_after_reply_probability",
    "nickname_candidate_probability",
    "plus_one_probability",
    "plus_one_break_probability",
}
NONNEGATIVE = {
    "background_images",
    "mention_wait_seconds",
    "media_wait_seconds",
    "auto_reply_cooldown",
    "poke_after_reply_delay",
    "poke_reply_cooldown",
    "mute_backoff_seconds",
    "bubble_typing_delay_max",
    "thread_continuation_seconds",
    "thread_max_turns",
    "thread_min_interval_seconds",
    "input_debounce_seconds",
    "auto_stale_max_messages",
    "auto_stale_max_seconds",
    "poke_after_reply_max_chars",
    "plus_one_cooldown",
    "emoji_react_cooldown",
    "room_memory_token_budget",
}


def resolve_tz(config):
    """Return configured ZoneInfo, falling back safely to +08:00 or UTC."""
    name = str((config or {}).get("timezone_name", "") or "").strip()
    if not name:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        if name in {"Asia/Shanghai", "Asia/Chongqing", "PRC", "CST", "+08:00"}:
            return timezone(timedelta(hours=8))
        return timezone.utc


def parse_overrides(raw) -> dict[str, dict]:
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise ValueError("配置 group_overrides 不是合法 JSON") from exc
    else:
        raise ValueError("配置 group_overrides 类型不正确")
    if not isinstance(parsed, dict):
        raise ValueError("配置 group_overrides 必须是 JSON 对象")
    result = {}
    for gid, overrides in parsed.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"配置 group_overrides[{gid}] 必须是对象")
        result[str(gid)] = copy.deepcopy(overrides)
    return result


def migrate_config(raw):
    result = copy.deepcopy(dict(raw))
    version = result.get("config_version", 1)
    if type(version) is not int or version > CONFIG_VERSION:
        raise ValueError("不支持此配置版本")
    # Existing v2.0 settings take priority over defaults, especially participation/pokes.
    result["config_version"] = CONFIG_VERSION
    return result


def _validate_value(key, value, rule):
    kind = rule["type"]
    if key == "group_overrides":
        parse_overrides(value)
        return
    valid = (
        type(value) is bool
        if kind == "bool"
        else type(value) is int
        if kind == "int"
        else type(value) in (int, float) and math.isfinite(value)
        if kind == "float"
        else isinstance(value, str)
        if kind in {"string", "text"}
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


class Settings(dict):
    def __init__(self, raw):
        super().__init__()
        self._group_overrides: dict[str, dict] = {}
        self.update(migrate_config(raw))

    def update(self, other=(), **kwargs):
        values = {key: copy.deepcopy(rule["default"]) for key, rule in SCHEMA.items()}
        values.update(self)
        values.update(dict(other, **kwargs))
        for key, rule in SCHEMA.items():
            _validate_value(key, values[key], rule)
        overrides = parse_overrides(values["group_overrides"])
        for gid, group_vals in overrides.items():
            for okey, oval in group_vals.items():
                if okey in SCHEMA and okey != "group_overrides":
                    _validate_value(okey, oval, SCHEMA[okey])
        if values["background_images"] > values["max_images"]:
            raise ValueError("背景图片上限不能大于总图片上限")
        if values["input_token_budget"] < 256:
            raise ValueError("输入预算不能小于 256")
        if values["plus_one_min_users"] < 2:
            raise ValueError("plus_one_min_users 不能小于 2")
        self._group_overrides = overrides
        super().update(values)

    def for_group(self, group_id) -> dict:
        """Return effective config dictionary with per-group overrides applied."""
        gid = str(group_id or "")
        group_vals = self._group_overrides.get(gid)
        if not group_vals:
            return self
        merged = dict(self)
        for key, value in group_vals.items():
            if key in SCHEMA and key != "group_overrides":
                merged[key] = copy.deepcopy(value)
        return merged
