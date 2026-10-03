"""Causal selection and immutable event frames; images stay symbolic in SQLite."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone

from .adapters.onebot import image_ids
from .models import Selection, event_order
from .output import strip_headers


class ContextLimit(ValueError):
    pass


def text_tokens(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return sum(2 if ord(c) > 127 else 1 / 3 for c in text).__ceil__()


class ContextSelector:
    def __init__(self, journal, media, config):
        self.journal, self.media, self.config = journal, media, config

    async def candidates(self, anchor):
        room, trigger = anchor["room"], anchor["seq"]
        recent, watermark = await self.journal.view(
            anchor,
            anchor["received"] - float(self.config.get("context_minutes", 10)) * 60,
            int(self.config.get("max_context_messages", 40)),
        )
        selected = {e["seq"]: e for e in recent}
        selected[trigger] = anchor
        protected = {trigger}
        primary = set(image_ids(anchor["parts"]))
        reasons = {e["seq"]: "recent_room" for e in recent}
        reasons[trigger] = "anchor"
        references = []
        parent = anchor
        for _ in range(3):
            reply = next((p for p in parent["parts"] if p["type"] == "reply"), None)
            if not reply:
                break
            parent = await self.journal.get(room, reply["event_id"])
            if not parent or parent["seq"] > trigger or parent["seq"] <= await self.journal.floor(room):
                break
            selected[parent["seq"]] = parent
            protected.add(parent["seq"])
            reasons[parent["seq"]] = "explicit_reply"
            references.append(parent)
            primary.update(image_ids(parent["parts"]))
        if not references and not primary:
            same = await self.journal.same_sender(
                anchor, float(self.config.get("association_seconds", 120)), limit=8, images_only=True
            )
            for event in same:
                if event["seq"] == trigger:
                    continue
                mids = image_ids(event["parts"])
                if mids:
                    selected[event["seq"]] = event
                    protected.add(event["seq"])
                    reasons[event["seq"]] = "same_sender_recent_image"
                    primary.update(mids)
        # A quote of a bot answer also restores the actual sources used in that turn.
        queue = list(references)
        seen = set()
        while queue and len(seen) < 32:
            event = queue.pop(0)
            if event["seq"] in seen:
                continue
            seen.add(event["seq"])
            parent_seq = event.get("causal_anchor")
            sources = await self.journal.dependencies(room, parent_seq or event["seq"])
            if parent_seq:
                if not sources and event["event_id"].startswith("generation:"):
                    # v2 did not store dependencies. Recover only the bounded
                    # implicit same-sender image association of that old turn.
                    question = await self.journal.by_seq(room, parent_seq)
                    if question:
                        legacy = await self.journal.same_sender(
                            question,
                            float(self.config.get("association_seconds", 120)),
                            limit=8,
                            images_only=True,
                        )
                        sources = [e["seq"] for e in legacy]
                sources = [parent_seq] + sources
            for seq in sources[:32]:
                source = await self.journal.by_seq(room, seq)
                if source and source["seq"] <= trigger:
                    selected[seq] = source
                    protected.add(seq)
                    primary.update(image_ids(source["parts"]))
                    reasons[seq] = "referenced_turn_source"
                    queue.append(source)
        # A public reply and its trigger remain a complete turn when trimming.
        seed = {e["seq"] for e in recent} | {trigger}
        for event in list(selected.values()):
            if event.get("causal_anchor") is not None:
                parent = await self.journal.by_seq(room, event["causal_anchor"])
                if parent:
                    selected[parent["seq"]] = parent
                    if event["seq"] in seed:
                        seed.add(parent["seq"])
        return Selection(
            anchor,
            sorted(selected.values(), key=event_order),
            protected,
            primary,
            reasons,
            watermark,
            seed_seqs=seed,
        )

    async def _parts(self, parts, room, selected_media, records):
        output = []
        for part in parts:
            kind = part["type"]
            if kind == "text":
                output.append({"type": "text", "text": part["text"]})
            elif kind == "image":
                mid = part["media_id"]
                output.append({"type": "text", "text": "[media_id=" + mid + "]"})
                record = records.get(mid)
                if mid in selected_media and record and record["status"] == "ready":
                    output.append({"type": "journal_image", "media_id": mid})
                elif not record or record["status"] != "ready":
                    output.append(
                        {
                            "type": "text",
                            "text": "[已收到图片，但附件状态="
                            + str(record["status"] if record else "missing")
                            + "; 尚未看到图片内容]",
                        }
                    )
                else:
                    output.append({"type": "text", "text": "[背景图片未纳入此历史帧的视觉预算]"})
            elif kind == "mention":
                output.append(
                    {
                        "type": "text",
                        "text": "[mention_id=" + json.dumps(part["target_id"], ensure_ascii=False) + "]",
                    }
                )
            elif kind == "reply":
                output.append(
                    {
                        "type": "text",
                        "text": "[reply_to=" + json.dumps(part["event_id"], ensure_ascii=False) + "]",
                    }
                )
            elif kind == "poke":
                output.append(
                    {
                        "type": "text",
                        "text": "[戳一戳事件="
                        + json.dumps(
                            {"actor_id": part["actor_id"], "target_id": part["target_id"]}, ensure_ascii=False
                        )
                        + "]",
                    }
                )
            elif kind == "forward":
                output.append(
                    {
                        "type": "text",
                        "text": "[合并转发开始]" if part["available"] else "[合并转发内容无法获取]",
                    }
                )
                for node in part["nodes"]:
                    output.append(
                        {
                            "type": "text",
                            "text": "[forward_sender="
                            + json.dumps({"id": node["sender_id"], "name": node["name"]}, ensure_ascii=False)
                            + "]",
                        }
                    )
                    output.extend(await self._parts(node["parts"], room, selected_media, records))
                if part["available"]:
                    output.append({"type": "text", "text": "[合并转发结束]"})
            elif kind == "unavailable":
                output.append(
                    {"type": "text", "text": "[附件不可用: " + part["kind"] + "; " + part["reason"] + "]"}
                )
        return output

    async def _frame(self, event, chosen_media, records):
        metadata = {
            "event_id": event["event_id"],
            "sender_id": event["sender"],
            "name": event["name"],
            "time": datetime.fromtimestamp(event["received"], timezone.utc).isoformat(),
            "source": event["kind"],
        }
        blocks = [
            {"type": "text", "text": "[message_metadata=" + json.dumps(metadata, ensure_ascii=False) + "]"}
        ]
        if event["kind"] != "self":
            blocks.extend(await self._parts(event["parts"], event["room"], chosen_media, records))
            return {"messages": [{"role": "user", "content": blocks}]}
        # Platform identity/reply/mention/attachment facts are data, never an
        # assistant output example. Keep the actual sent text in its own role.
        facts = [p for p in event["parts"] if p["type"] != "text"]
        blocks.extend(await self._parts(facts, event["room"], chosen_media, records))
        raw_body = "".join(p["text"] for p in event["parts"] if p["type"] == "text")
        body = strip_headers(raw_body)
        if body != raw_body:
            blocks.append(
                {
                    "type": "text",
                    "text": "[平台记录：此历史消息曾夹带内部格式标记，此处已省略，仅保留回答正文]",
                }
            )
        messages = [{"role": "user", "content": blocks}]
        if body.strip():
            blocks.append(
                {"type": "text", "text": "[下一条 assistant 是此机器人消息的正文，平台记录不属于正文]"}
            )
            messages.append({"role": "assistant", "content": [{"type": "text", "text": body}]})
        return {"messages": messages}

    @staticmethod
    def _frame_media(frame):
        return {
            p["media_id"]
            for message in frame["messages"]
            if isinstance(message.get("content"), list)
            for p in message["content"]
            if p.get("type") == "journal_image"
        }

    async def assemble(
        self,
        selection,
        fixed_tokens=0,
        total_limit=None,
        reserved_images=0,
        *,
        freeze=True,
        frames_override=None,
        contiguous=False,
    ):
        mids = {mid for event in selection.events for mid in image_ids(event["parts"])}
        if mids:
            self.media.lease(mids)
        try:
            return await self._assemble(
                selection,
                fixed_tokens,
                total_limit,
                reserved_images,
                freeze=freeze,
                frames_override=frames_override,
                contiguous=contiguous,
            )
        finally:
            if mids:
                self.media.release(mids)

    async def _assemble(
        self,
        selection,
        fixed_tokens=0,
        total_limit=None,
        reserved_images=0,
        *,
        freeze=True,
        frames_override=None,
        contiguous=False,
    ):
        limit = min(int(self.config.get("input_token_budget", 32768)), total_limit or 10**9)
        budget = limit - fixed_tokens
        if budget <= 0:
            raise ContextLimit("人格和工具已占满输入预算，请提高预算或缩小工具集合。")
        room = selection.anchor["room"]
        records, frozen = await self.journal.context_data(
            room,
            [e["seq"] for e in selection.events],
            {mid for event in selection.events for mid in image_ids(event["parts"])},
        )

        if frames_override is not None:
            frozen = frames_override

        def key(mid):
            record = records.get(mid)
            return record["sha"] if record and record.get("sha") else mid

        def keys(mids):
            return {key(mid) for mid in mids}

        max_images = int(self.config.get("max_images", 6)) - reserved_images
        if len(keys(selection.primary_media)) > max_images:
            raise ContextLimit("关联图片超过本次上限，请引用具体图片或分批提问。")
        image_costs = {}
        for mid, record in records.items():
            tiles = (
                math.ceil(record["width"] / 512) * math.ceil(record["height"] / 512)
                if record and record["status"] == "ready"
                else 1
            )
            image_costs[key(mid)] = max(
                int(self.config.get("image_token_reserve", 1600)), min(tiles, 64) * 256
            )
        preferred = set(selection.primary_media)
        ambient = int(self.config.get("background_images", 2))
        # Keep the oldest prefix stable, rather than rotating background images each turn.
        for event in selection.events:
            for mid in image_ids(event["parts"]):
                if key(mid) not in keys(preferred) and ambient > 0 and len(keys(preferred)) < max_images:
                    preferred.add(mid)
                    ambient -= 1
        frames = {}
        for event in selection.events:
            frames[event["seq"]] = frozen.get(event["seq"]) or await self._frame(event, preferred, records)

        def frame_keys(seq):
            return keys(self._frame_media(frames[seq]))

        def text_cost(seq):
            return text_tokens(frames[seq]) + 120

        groups = {}
        for event in selection.events:
            parent = event.get("causal_anchor") or event["seq"]
            groups.setdefault(parent, []).append(event["seq"])
        mandatory = {seq for group in groups.values() if set(group) & selection.protected for seq in group}
        chosen = set(mandatory)
        used_images = keys(selection.primary_media) | set().union(*(frame_keys(seq) for seq in chosen))
        used_text = sum(text_cost(seq) for seq in chosen)

        def cost(text, images):
            return text + sum(image_costs.get(k, 1600) for k in images)

        if len(used_images) > max_images or cost(used_text, used_images) > budget:
            raise ContextLimit("当前问题和关联图片超出输入预算，请分批提问。")
        for group in reversed(list(groups.values())):
            additional = set(group) - chosen
            new_images = used_images | set().union(*(frame_keys(seq) for seq in additional))
            new_text = used_text + sum(text_cost(seq) for seq in additional)
            if len(new_images) <= max_images and cost(new_text, new_images) <= budget:
                chosen.update(additional)
                used_images, used_text = new_images, new_text
            elif contiguous:
                break
        selected = [event for event in selection.events if event["seq"] in chosen]
        selection.chosen_events = selected
        selection.chosen_media = {mid for mid in records if key(mid) in used_images}
        selection.tokens = cost(used_text, used_images) + fixed_tokens
        selection.rebases = 0
        emitted = set()

        async def materialize(blocks):
            output = []
            for block in blocks:
                if block.get("type") != "journal_image":
                    output.append(copy.deepcopy(block))
                    continue
                mid = block["media_id"]
                asset = key(mid)
                if asset in emitted:
                    output.append({"type": "text", "text": "[与此前同一附件，原图已在上下文中]"})
                    continue
                uri = await self.media.data_uri(mid, room)
                if uri:
                    output.append({"type": "image_url", "image_url": {"url": uri}})
                    emitted.add(asset)
                else:
                    selection.rebases += 1
                    output.append({"type": "text", "text": "[原图附件已失效，尚未看到图片内容]"})
            return output

        if freeze:
            new = {event["seq"]: frames[event["seq"]] for event in selected if not frozen.get(event["seq"])}
            if new:
                frames.update(await self.journal.freeze_many(new))
        selection.frames = {seq: copy.deepcopy(frames[seq]) for seq in chosen}
        selection.canonical_current = []
        history, current = [], []
        for event in selected:
            frame = frames[event["seq"]]
            for message in frame["messages"]:
                wire = copy.deepcopy(message)
                if isinstance(wire.get("content"), list):
                    wire["content"] = await materialize(wire["content"])
                if event["seq"] == selection.anchor["seq"]:
                    current = wire["content"]
                    selection.canonical_current = copy.deepcopy(current)
                else:
                    history.append(wire)
        # An old frame may have omitted an image or captured its pending state.
        # Add the now-required original at the tail, without rewriting that frame.
        for mid in sorted(selection.primary_media):
            if key(mid) not in emitted and records.get(mid) and records[mid]["status"] == "ready":
                current.append({"type": "text", "text": "[本轮关联原图 media_id=" + mid + "]"})
                current.extend(await materialize([{"type": "journal_image", "media_id": mid}]))
        return history, current
