"""Authenticated OneBot group poke notices and bounded platform actions."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping


def notice(event):
    if event.get_platform_name() != "aiocqhttp" or not event.get_group_id():
        return None
    raw = getattr(event.message_obj, "raw_message", None)
    if not isinstance(raw, Mapping) or (
        raw.get("post_type"), raw.get("notice_type"), raw.get("sub_type")
    ) != ("notice", "notify", "poke"):
        return None
    actor, target = str(raw.get("user_id", "")), str(raw.get("target_id", ""))
    if (not actor or not target or actor != str(event.get_sender_id())
            or str(raw.get("self_id", "")) != str(event.get_self_id())
            or str(raw.get("group_id", "")) != str(event.get_group_id())):
        return None
    return {"type": "poke", "actor_id": actor, "target_id": target}


def in_scope(config, event):
    groups = [str(g) for g in config.get("poke_enabled_groups", [])]
    return not groups or str(event.get_group_id()) in groups


def accepted(config, event, poke):
    mode = config.get("poke_message_mode", "bot_only")
    return in_scope(config, event) and (
        mode == "all" or (mode == "bot_only" and poke["target_id"] == str(event.get_self_id()))
    )


def probability(config, name, default):
    return max(0.0, min(1.0, float(config.get(name, default))))


async def send(event, target):
    """Raise on failure; callers decide logging. Never touch the event's sender."""
    group, target = str(event.get_group_id()), str(target)
    bot = getattr(event, "bot", None)
    if event.get_platform_name() != "aiocqhttp" or bot is None or not group.isdecimal() or not target.isdecimal():
        return False
    await asyncio.wait_for(bot.api.call_action("send_poke", group_id=int(group), user_id=int(target)), 2)
    return True
