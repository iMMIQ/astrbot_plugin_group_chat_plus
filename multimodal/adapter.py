"""Normalize copies of platform segments; never flatten the live event in place."""
from __future__ import annotations

import asyncio
import json
import time
import uuid

from .poke import notice as poke_notice


def room_key(event):
    return json.dumps([str(event.get_platform_id()), str(event.get_self_id()),
                       "group", str(event.get_group_id())], ensure_ascii=False, separators=(",", ":"))


def field(part, name, default=None):
    if isinstance(part, dict):
        data = part.get("data", part)
        return data.get(name, default)
    return getattr(part, name, default)


def part_type(part):
    if isinstance(part, dict):
        return str(part.get("type", "unknown")).lower()
    return type(part).__name__.lower()


def image_ids(parts):
    ids = []
    for part in parts:
        if part["type"] == "image":
            ids.append(part["media_id"])
        elif part["type"] == "forward":
            for node in part.get("nodes", []):
                ids.extend(image_ids(node["parts"]))
        elif part["type"] == "protocol":
            for message in part["messages"]:
                if isinstance(message.get("content"), list):
                    ids.extend(p["media_id"] for p in message["content"] if p.get("type") == "journal_image")
    return ids


class PlatformAdapter:
    def __init__(self, journal, media):
        self.journal, self.media = journal, media
        self.locks = {}

    async def _action(self, event, action, **kwargs):
        bot = getattr(event, "bot", None)
        if bot is None:
            return None
        try:
            return await asyncio.wait_for(bot.api.call_action(action, **kwargs), 2)
        except Exception:
            return None

    async def normalize(self, event, parts, room, depth=0, quota=None):
        quota = {"images": 0, "nodes": 0} if quota is None else quota
        result = []
        for part in list(parts or [])[:100]:
            kind = part_type(part)
            if kind in {"plain", "text"}:
                result.append({"type": "text", "text": str(field(part, "text", ""))})
            elif kind == "image":
                source = str(field(part, "path") or field(part, "url") or field(part, "file") or "")
                quota["images"] += 1
                if source and quota["images"] <= 12:
                    result.append({"type": "image", "media_id": self.media.capture(room, source)})
                else:
                    result.append({"type": "unavailable", "kind": "image", "reason": "capture_limit" if source else "missing_source"})
            elif kind in {"at", "atall"}:
                result.append({"type": "mention", "target_id": str(field(part, "qq", "all"))})
            elif kind == "reply":
                target = str(field(part, "id", ""))
                result.append({"type": "reply", "event_id": target})
                if target and depth < 3 and not self.journal.get(room, target):
                    chain = field(part, "chain", [])
                    sender = str(field(part, "sender_id", ""))
                    name = str(field(part, "sender_nickname", ""))
                    timestamp = field(part, "time", 0)
                    if not chain:
                        fetched = await self._action(event, "get_msg", message_id=target)
                        if fetched and str(fetched.get("group_id", "")) == str(event.get_group_id()):
                            chain = fetched.get("message", [])
                            sender = str(fetched.get("sender", {}).get("user_id", ""))
                            name = str(fetched.get("sender", {}).get("nickname", ""))
                            timestamp = fetched.get("time", 0)
                    if isinstance(chain, list) and chain:
                        normalized = await self.normalize(event, chain, room, depth + 1, quota)
                        self.journal.add(room, target, sender, name, normalized, kind="quoted",
                                         received=float(timestamp or time.time()))
            elif kind in {"forward", "nodes", "node"}:
                nodes = field(part, "nodes", [])
                if kind == "node":
                    nodes = [part]
                if kind == "forward" and depth < 3:
                    fetched = await self._action(event, "get_forward_msg", message_id=str(field(part, "id", "")))
                    nodes = fetched.get("messages", []) if fetched else []
                output = []
                for node in nodes or []:
                    quota["nodes"] += 1
                    if quota["nodes"] > 20 or depth >= 3:
                        break
                    sender = node.get("sender", {}) if isinstance(node, dict) else {}
                    content = field(node, "content", None)
                    if content is None and isinstance(node, dict):
                        content = node.get("message", [])
                    if not isinstance(content, list):
                        content = [{"type": "text", "data": {"text": str(content or "")}}]
                    output.append({"sender_id": str(sender.get("user_id") or field(node, "uin", "")),
                                   "name": str(sender.get("nickname") or field(node, "name", "")),
                                   "parts": await self.normalize(event, content, room, depth + 1, quota)})
                result.append({"type": "forward", "nodes": output, "available": bool(output)})
            elif kind == "face":
                result.append({"type": "text", "text": "[QQ表情 id=" + str(field(part, "id", "")) + "]"})
            elif kind == "poke" and (poke := poke_notice(event)):
                result.append(poke)
            else:
                result.append({"type": "unavailable", "kind": kind, "reason": "unsupported_modality"})
        return result

    async def ingest(self, event):
        room = room_key(event)
        async with self.locks.setdefault(room, asyncio.Lock()):
            return await self._ingest(event, room)

    async def _ingest(self, event, room):
        eid = str(getattr(event.message_obj, "message_id", "") or uuid.uuid4().hex)
        if existing := self.journal.get(room, eid):
            return existing, False
        # Reserve a local arrival time before bounded quote/forward resolution.
        received = time.time()
        poke = poke_notice(event)
        parts = [poke] if poke else await self.normalize(event, event.get_messages(), room)
        return self.journal.add(room, eid, event.get_sender_id(), event.get_sender_name(), parts, received=received)
