"""Selection is deterministic; the model resolves meaning from typed messages."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .adapter import image_ids


class ContextLimit(ValueError):
    pass


def text_tokens(value):
    # Conservative without a model tokenizer: CJK may consume >1 token/character.
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return sum(2 if ord(c) > 127 else 1 / 3 for c in text).__ceil__()


@dataclass
class Selection:
    anchor: dict
    events: list
    protected: set
    primary_media: set
    reasons: dict
    chosen_media: set = field(default_factory=set)
    tokens: int = 0


class ContextSelector:
    def __init__(self, journal, media, config):
        self.journal, self.media, self.config = journal, media, config

    def candidates(self, anchor):
        room, snapshot = anchor["room"], anchor["seq"]
        recent = self.journal.recent(room, snapshot, anchor["received"] - float(self.config.get("context_minutes", 10)) * 60,
                                     int(self.config.get("max_context_messages", 40)))
        selected = {e["seq"]: e for e in recent}
        selected[snapshot] = anchor
        protected = {snapshot}
        primary = set(image_ids(anchor["parts"]))
        reasons = {e["seq"]: "recent_room" for e in recent}
        reasons[snapshot] = "anchor"
        references = []
        parent = anchor
        for _ in range(3):
            reply = next((p for p in parent["parts"] if p["type"] == "reply"), None)
            if not reply:
                break
            parent = self.journal.get(room, reply["event_id"])
            if not parent or parent["seq"] > snapshot or parent["seq"] <= self.journal.floor(room):
                break
            selected[parent["seq"]] = parent
            protected.add(parent["seq"])
            reasons[parent["seq"]] = "explicit_reply"
            references.append(parent)
            primary.update(image_ids(parent["parts"]))
        # Explicit cross-sender references outrank implicit same-sender images.
        if not references and not primary:
            same = self.journal.same_sender(anchor, float(self.config.get("association_seconds", 120)), limit=8, images_only=True)
            for event in same:
                if event["seq"] == snapshot:
                    continue
                mids = image_ids(event["parts"])
                if mids:
                    selected[event["seq"]] = event
                    protected.add(event["seq"])
                    reasons[event["seq"]] = "same_sender_recent_image"
                    primary.update(mids)
                    if len(primary) > int(self.config.get("max_images", 6)):
                        break
        return Selection(anchor, sorted(selected.values(), key=lambda e: e["seq"]), protected, primary, reasons)

    def _image_cost(self, mid, room):
        media = self.journal.media(mid, room)
        if not media or media["status"] != "ready":
            return 64
        tiles = math.ceil(media["width"] / 512) * math.ceil(media["height"] / 512)
        # A configurable conservative reserve, not a claim about provider pricing.
        return max(int(self.config.get("image_token_reserve", 1600)), min(tiles, 64) * 256)

    async def _parts(self, parts, room, selected_media, emitted):
        output = []
        for part in parts:
            kind = part["type"]
            if kind == "text":
                output.append({"type": "text", "text": part["text"]})
            elif kind == "image":
                mid = part["media_id"]
                output.append({"type": "text", "text": "[media_id=" + mid + "]"})
                if mid in emitted:
                    output.append({"type": "text", "text": "[与此前同一附件，原图已在上下文中]"})
                elif mid in selected_media:
                    uri = await self.media.data_uri(mid, room)
                    if uri:
                        output.append({"type": "image_url", "image_url": {"url": uri}})
                        emitted.add(mid)
                    else:
                        state = self.journal.media(mid, room)
                        output.append({"type": "text", "text": "[已收到图片，但附件状态=" + str(state["status"] if state else "missing") + "; 尚未看到图片内容]"})
                else:
                    output.append({"type": "text", "text": "[背景图片未纳入本次视觉预算]"})
            elif kind == "mention":
                output.append({"type": "text", "text": "[mention_id=" + json.dumps(part["target_id"], ensure_ascii=False) + "]"})
            elif kind == "reply":
                output.append({"type": "text", "text": "[reply_to=" + json.dumps(part["event_id"], ensure_ascii=False) + "]"})
            elif kind == "poke":
                output.append({"type": "text", "text": "[戳一戳事件=" + json.dumps(
                    {"actor_id": part["actor_id"], "target_id": part["target_id"]}, ensure_ascii=False) + "]"})
            elif kind == "forward":
                output.append({"type": "text", "text": "[合并转发开始]" if part["available"] else "[合并转发内容无法获取]"})
                for node in part["nodes"]:
                    output.append({"type": "text", "text": "[forward_sender=" + json.dumps({"id": node["sender_id"], "name": node["name"]}, ensure_ascii=False) + "]"})
                    output.extend(await self._parts(node["parts"], room, selected_media, emitted))
                if part["available"]:
                    output.append({"type": "text", "text": "[合并转发结束]"})
            elif kind == "unavailable":
                output.append({"type": "text", "text": "[附件不可用: " + part["kind"] + "; " + part["reason"] + "]"})
        return output

    async def assemble(self, selection, fixed_tokens=0, total_limit=None, reserved_images=0):
        limit = min(int(self.config.get("input_token_budget", 32768)), total_limit or 10**9)
        budget = limit - fixed_tokens
        if budget <= 0:
            raise ContextLimit("人格和工具已占满输入预算，请提高预算或缩小工具集合。")
        max_images = int(self.config.get("max_images", 6)) - reserved_images
        if len(selection.primary_media) > max_images:
            raise ContextLimit("关联图片超过本次上限，请引用具体图片或分批提问。")
        selected_media = set(selection.primary_media)
        ambient = int(self.config.get("background_images", 2))
        for event in reversed(selection.events):
            for mid in image_ids(event["parts"]):
                if mid not in selected_media and ambient > 0 and len(selected_media) < max_images:
                    selected_media.add(mid)
                    ambient -= 1
        def cost(event):
            # Stored protocol messages contain no wire images or reasoning blobs.
            return text_tokens(event["parts"]) + 120 + sum(
                self._image_cost(mid, event["room"]) for mid in image_ids(event["parts"]) if mid in selected_media
            )
        mandatory = [e for e in selection.events if e["seq"] in selection.protected]
        required = sum(cost(e) for e in mandatory)
        if required > budget:
            raise ContextLimit("当前问题和关联图片超出输入预算，请分批提问。")
        chosen = {e["seq"]: e for e in mandatory}
        used = required
        for event in reversed(selection.events):
            if event["seq"] not in chosen and used + cost(event) <= budget:
                chosen[event["seq"]] = event
                used += cost(event)
        # Remove media reserved for background events that did not fit.
        selected_media &= {mid for e in chosen.values() for mid in image_ids(e["parts"])}
        selection.chosen_media, selection.tokens = selected_media, used + fixed_tokens
        emitted = set()
        messages = []
        current = []
        for event in sorted(chosen.values(), key=lambda e: e["seq"]):
            if event["kind"] == "self":
                for part in event["parts"]:
                    if part["type"] == "protocol":
                        # Keep whole tool exchanges as one selected journal event.
                        for message in copy.deepcopy(part["messages"]):
                            if isinstance(message.get("content"), list):
                                blocks = []
                                for block in message["content"]:
                                    if block.get("type") == "journal_image":
                                        blocks.extend(await self._parts([{"type": "image", "media_id": block["media_id"]}], event["room"], selected_media, emitted))
                                    else:
                                        blocks.append(block)
                                message["content"] = blocks
                            messages.append(message)
                continue
            metadata = {"event_id": event["event_id"], "sender_id": event["sender"], "name": event["name"],
                        "time": datetime.fromtimestamp(event["received"], timezone.utc).isoformat(), "source": event["kind"]}
            blocks = [{"type": "text", "text": "[message_metadata=" + json.dumps(metadata, ensure_ascii=False) + "]"}]
            blocks.extend(await self._parts(event["parts"], event["room"], selected_media, emitted))
            if event["seq"] == selection.anchor["seq"]:
                current = blocks
            else:
                messages.append({"role": "user", "content": blocks})
        selection.events = sorted(chosen.values(), key=lambda e: e["seq"])
        return messages, current
