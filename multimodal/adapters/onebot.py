"""Normalize copies of platform segments; never flatten the live event in place."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from ..models import RoomId
from ..qq_faces import face_label
from .poke import notice as poke_notice


def _safe_http_url(value) -> str:
    s = str(value or "").strip()
    return s[:400] if s.startswith(("http://", "https://")) else ""


def _safe_basename(value) -> str:
    s = str(value or "").strip()
    if not s or s.startswith(("http://", "https://", "data:", "base64://")):
        return ""
    return os.path.basename(s.replace("\\", "/"))[:120]


def _human_size(value) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return ""


def extract_json_card(raw) -> str | None:
    """Extract readable title/desc/URL from a QQ JSON share card (pure data)."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except ValueError:
            return None
    elif isinstance(raw, dict):
        obj = raw
    else:
        return None
    if not isinstance(obj, dict):
        return None

    prompt = str(obj.get("prompt") or "").strip()
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    tag = ""
    title = ""
    desc = ""
    urls: list[str] = []

    def _collect(node, depth=0):
        nonlocal tag, title, desc
        if depth > 4 or not isinstance(node, dict):
            return
        if not tag:
            for k in ("tag", "sourceName", "appName"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    tag = v.strip()[:40]
                    break
        if not title:
            for k in ("title", "news_title", "subject"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    title = v.strip()[:120]
                    break
        if not desc:
            for k in ("desc", "summary", "content", "subtitle"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    desc = v.strip()[:180]
                    break
        for k in ("qqdocurl", "jumpUrl", "pcJumpUrl", "url", "musicUrl", "source_url", "docurl"):
            u = _safe_http_url(node.get(k))
            if u and u not in urls:
                urls.append(u)
        for v in node.values():
            if isinstance(v, dict):
                _collect(v, depth + 1)

    _collect(meta)
    parts: list[str] = []
    if tag:
        parts.append(tag)
    if title and title != tag:
        parts.append(title)
    if desc and desc not in (title, tag):
        parts.append(desc)
    if not parts and prompt:
        parts.append(prompt[:160])
    if urls:
        parts.append("链接: " + urls[0])
    if not parts:
        return None
    return "[QQ分享卡片] " + " | ".join(parts)[:500]


def extract_file_label(part, *, fetched_url: str = "") -> str:
    name = _safe_basename(field(part, "name", "") or field(part, "file", "")) or "未命名文件"
    size = _human_size(field(part, "file_size", "") or field(part, "size", ""))
    fid = str(field(part, "file_id", "") or "").strip()[:80]
    url = _safe_http_url(fetched_url or field(part, "url", ""))
    bits = [f"name={name}"]
    if size:
        bits.append(f"size={size}")
    if fid:
        bits.append(f"file_id={fid}")
    if url:
        bits.append(f"url={url}")
    return "[群文件 " + " ".join(bits) + "]"


def extract_media_label(kind: str, part) -> str:
    label = "群视频" if kind == "video" else "群语音"
    name = _safe_basename(field(part, "name", "") or field(part, "file", ""))
    url = _safe_http_url(field(part, "url", ""))
    bits = []
    if name:
        bits.append(f"file={name}")
    if url:
        bits.append(f"url={url}")
    return f"[{label} " + " ".join(bits) + "]" if bits else f"[{label}]"


def room_key(event) -> RoomId:
    return RoomId(
        json.dumps(
            [str(event.get_platform_id()), str(event.get_self_id()), "group", str(event.get_group_id())],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


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
        self.inflight = {}

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
                    result.append({"type": "image", "media_id": await self.media.capture(room, source)})
                else:
                    result.append(
                        {
                            "type": "unavailable",
                            "kind": "image",
                            "reason": "capture_limit" if source else "missing_source",
                        }
                    )
            elif kind in {"at", "atall"}:
                result.append({"type": "mention", "target_id": str(field(part, "qq", "all"))})
            elif kind == "reply":
                target = str(field(part, "id", ""))
                result.append({"type": "reply", "event_id": target})
                if target and depth < 3 and not await self.journal.get(room, target):
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
                        await self.journal.add(
                            room,
                            target,
                            sender,
                            name,
                            normalized,
                            kind="quoted",
                            received=float(timestamp or time.time()),
                        )
            elif kind in {"forward", "nodes", "node"}:
                nodes = field(part, "nodes", [])
                if kind == "node":
                    nodes = [part]
                if kind == "forward" and depth < 3:
                    fetched = await self._action(
                        event, "get_forward_msg", message_id=str(field(part, "id", ""))
                    )
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
                    output.append(
                        {
                            "sender_id": str(sender.get("user_id") or field(node, "uin", "")),
                            "name": str(sender.get("nickname") or field(node, "name", "")),
                            "parts": await self.normalize(event, content, room, depth + 1, quota),
                        }
                    )
                result.append({"type": "forward", "nodes": output, "available": bool(output)})
            elif kind == "face":
                result.append({"type": "text", "text": face_label(field(part, "id", ""))})
            elif kind == "poke" and (poke := poke_notice(event)):
                result.append(poke)
            elif kind == "json":
                card = extract_json_card(field(part, "data", "") or part)
                if card:
                    result.append({"type": "text", "text": card})
                else:
                    result.append({"type": "unavailable", "kind": kind, "reason": "empty_card"})
            elif kind == "file":
                fetched_url = _safe_http_url(field(part, "url", ""))
                fid = str(field(part, "file_id", "") or "").strip()
                gid = str(event.get_group_id() or "")
                if not fetched_url and fid and gid.isdecimal():
                    resp = await self._action(event, "get_group_file_url", group_id=int(gid), file_id=fid)
                    if isinstance(resp, dict):
                        fetched_url = _safe_http_url(resp.get("url", ""))
                result.append({"type": "text", "text": extract_file_label(part, fetched_url=fetched_url)})
            elif kind in {"video", "record"}:
                result.append({"type": "text", "text": extract_media_label(kind, part)})
            else:
                result.append({"type": "unavailable", "kind": kind, "reason": "unsupported_modality"})
        return result

    async def ingest(self, event):
        room = room_key(event)
        token = uuid.uuid4().hex
        received = time.time()
        done = asyncio.get_running_loop().create_future()
        self.inflight[token] = (room, str(event.get_sender_id()), received, done)
        try:
            async with self.locks.setdefault(room, asyncio.Lock()):
                return await self._ingest(event, room, received)
        finally:
            done.set_result(None)
            self.inflight.pop(token, None)

    async def settle_arrivals(self, room, sender, since, until, timeout):
        pending = [
            done
            for r, s, t, done in self.inflight.values()
            if r == room and s == sender and since < t <= until
        ]
        if pending:
            await asyncio.wait(pending, timeout=timeout)

    async def _ingest(self, event, room, received):
        eid = str(getattr(event.message_obj, "message_id", "") or uuid.uuid4().hex)
        if existing := await self.journal.get(room, eid):
            return existing, False
        # Reserve a local arrival time before bounded quote/forward resolution.
        poke = poke_notice(event)
        parts = [poke] if poke else await self.normalize(event, event.get_messages(), room)
        sender = str(event.get_sender_id())
        name = str(event.get_sender_name() or "")
        if poke and (not name or name == sender):
            known = await self.journal.latest_sender_name(room, sender)
            if known:
                name = known
        return await self.journal.add(room, eid, sender, name, parts, received=received)

    def cleanup(self):
        active = {entry[0] for entry in self.inflight.values()}
        self.locks = {room: lock for room, lock in self.locks.items() if room in active}
