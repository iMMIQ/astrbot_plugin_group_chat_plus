"""One coordinator owns a turn from capture through pipeline completion."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace

from .adapters.onebot import PlatformAdapter, image_ids, part_type, room_key
from .adapters.poke import accepted as accept_poke
from .adapters.poke import in_scope as poke_scope
from .adapters.poke import notice as poke_notice
from .adapters.poke import probability
from .adapters.poke import send as send_poke
from .bridge import GATE_RULES, additions, protocol_messages, request_snapshot, rewrite
from .context import ContextLimit, ContextSelector
from .models import TurnState
from .participation import Participation
from .routing import TurnRoute, classify, identity
from .segments import SegmentManager, digest
from .storage.assets import MediaStore
from .storage.sqlite import Journal

OWNER = "_native_multimodal"


@dataclass
class RoomSlot:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: int = 0
    touched: float = field(default_factory=time.monotonic)


def image_count(history, current):
    return sum(
        p.get("type") == "image_url"
        for msg in history
        if isinstance(msg.get("content"), list)
        for p in msg["content"]
        if isinstance(p, dict)
    ) + sum((p.get("type") if isinstance(p, dict) else p.type) == "image_url" for p in current)


class ConversationService:
    def __init__(self, gateway, config):
        self.gateway, self.config = gateway, config
        self.logger = getattr(gateway, "logger", logging.getLogger(__name__))
        self.root = gateway.data_root()
        self.journal = Journal(self.root)
        self.media = MediaStore(self.journal, self.root / "media", config)
        self.adapter = PlatformAdapter(self.journal, self.media)
        self.selector = ContextSelector(self.journal, self.media, config)
        self.segments = SegmentManager(self.selector)
        self.participation = Participation(config)
        self.rooms = {}
        self.running_tasks = set()
        self.cleanup_task = None
        self.closing = False

    async def initialize(self):
        await self.journal.ready()
        await self.media.start()
        restored = self.gateway.restore_wrappers()
        self.cleanup_task = asyncio.create_task(self._cleanup())
        self.logger.info(
            "[NativeMM] v2.2.1 已加载；旧包装恢复=%s，主动参与=%s",
            restored,
            self.config["auto_reply_enabled"],
        )

    async def _cleanup(self):
        while True:
            try:
                await self.media.cleanup()
                self.participation.cleanup()
                self.adapter.cleanup()
                cutoff = time.monotonic() - 300
                self.rooms = {
                    room: slot for room, slot in self.rooms.items() if slot.pending or slot.touched >= cutoff
                }
            except Exception as exc:
                self.logger.warning("[NativeMM] 清理失败 type=%s", type(exc).__name__)
            await asyncio.sleep(300)

    async def terminate(self):
        self.closing = True
        if self.cleanup_task:
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task
        active = [task for task in self.running_tasks if task is not asyncio.current_task()]
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        await self.media.close()
        await self.journal.close()

    @asynccontextmanager
    async def room(self, key):
        slot = self.rooms.setdefault(key, RoomSlot())
        if slot.pending >= self.config["max_pending_turns"]:
            raise ContextLimit("本群回复队列已满，请稍后提问。")
        slot.pending += 1
        acquired = False
        try:
            await asyncio.wait_for(slot.lock.acquire(), self.config["queue_wait_seconds"])
            acquired = True
            yield
        except TimeoutError as exc:
            raise ContextLimit("本群回复排队超时，请稍后提问。") from exc
        finally:
            if acquired:
                slot.lock.release()
            slot.pending -= 1
            slot.touched = time.monotonic()

    def enabled(self, event):
        groups = [str(g) for g in self.config["enabled_groups"]]
        return (
            not self.closing
            and self.config["enable_group_chat"]
            and bool(event.get_group_id())
            and (not groups or str(event.get_group_id()) in groups)
            and str(event.get_sender_id()) != str(event.get_self_id())
        )

    async def capture(self, event):
        if not self.enabled(event):
            return
        task = asyncio.current_task()
        self.running_tasks.add(task)
        try:
            poke = poke_notice(event)
            raw = getattr(event.message_obj, "raw_message", None)
            if poke:
                if not accept_poke(self.config, event, poke):
                    return
            elif hasattr(raw, "get") and raw.get("post_type") == "notice":
                return
            event.set_extra("_group_context_flow_injected", True)
            anchor, inserted = await self.adapter.ingest(event)
            event.set_extra(OWNER + "_anchor", anchor)
            event.set_extra(OWNER + "_duplicate", not inserted)
            reply_to_bot = False
            for part in anchor["parts"]:
                if part["type"] == "reply":
                    referenced = await self.journal.get(anchor["room"], part["event_id"])
                    reply_to_bot = bool(referenced and referenced["sender"] == str(event.get_self_id()))
                    break
            route = classify(
                anchor,
                event.get_self_id(),
                framework_wake=bool(event.is_at_or_wake_command),
                reply_to_bot=reply_to_bot,
            )
            event.set_extra(OWNER + "_route", route)
            if route:
                event.is_at_or_wake_command = True
            pure = (
                len(event.get_messages()) == 1
                and part_type(event.get_messages()[0]) == "at"
                and str(event.get_messages()[0].qq) == str(event.get_self_id())
            )
            if poke and poke["target_id"] == str(event.get_self_id()):
                event.is_at_or_wake_command = True
                if inserted and random.random() < probability(
                    self.config, "poke_reverse_on_poke_probability", 0
                ):
                    if await self.poke(event, anchor, "reverse"):
                        event.set_extra(OWNER + "_reverse_poke", True)
            elif not pure:
                return
            async with contextlib.aclosing(self.reply(event, anchor, pure_mention=pure)) as flow:
                async for result in flow:
                    yield result
            event.stop_event()
        finally:
            self.running_tasks.discard(task)

    async def respond(self, event):
        anchor = event.get_extra(OWNER + "_anchor")
        if not anchor or self.gateway.already_handled(event):
            return
        explicit = event.get_extra(OWNER + "_route") is not None or bool(event.is_at_or_wake_command)
        if not explicit and not self.participation.candidate(anchor["room"]):
            return
        async with contextlib.aclosing(self.reply(event, anchor, auto=not explicit)) as flow:
            async for result in flow:
                yield result

    async def reply(self, event, anchor, pure_mention=False, auto=False):
        event.call_llm = True
        if event.get_extra(OWNER + "_duplicate", False):
            return
        route = event.get_extra(OWNER + "_route") or TurnRoute(
            str(event.get_self_id()), anchor["event_id"], "auto_approved" if auto else "framework_wake"
        )
        task = asyncio.current_task()
        self.running_tasks.add(task)
        leased, gid, completed, state = [], None, False, None
        try:
            async with self.room(anchor["room"]):
                if self.closing:
                    return
                if pure_mention:
                    await asyncio.sleep(self.config["mention_wait_seconds"])
                    deadline = time.time()
                    await self.adapter.settle_arrivals(
                        anchor["room"],
                        anchor["sender"],
                        anchor["received"],
                        deadline,
                        self.config["media_wait_seconds"],
                    )
                    after = await self.journal.recent(anchor["room"], 2**63 - 1, anchor["received"], 20)
                    batch = [
                        e
                        for e in after
                        if e["sender"] == anchor["sender"]
                        and e["seq"] > anchor["seq"]
                        and e["kind"] == "member"
                        and e["received"] <= deadline
                    ]
                    if batch:
                        anchor = batch[-1]
                        route = replace(route, reason="mention_followup")
                        event.set_extra(OWNER + "_anchor", anchor)
                selection = await self.selector.candidates(anchor)
                leased = list({mid for e in selection.events for mid in image_ids(e["parts"])})
                self.media.lease(leased)
                await self.media.wait(leased)
                provider = await self.gateway.provider(event)
                if provider is None:
                    raise ContextLimit("没有可用的聊天模型。")
                if auto:
                    if not self.participation.claim(anchor["room"]):
                        return
                    history, current = await self.selector.assemble(
                        selection,
                        fixed_tokens=1000,
                        total_limit=self.config["gate_token_budget"],
                        freeze=False,
                    )
                    if image_count(history, current):
                        self.gateway.validate_images(event, provider)
                    await self.journal.diagnose(
                        anchor["room"],
                        digest({"provider": provider.provider_config.get("id")}),
                        "gate",
                        {
                            "system": digest(GATE_RULES + "\n" + identity(event.get_self_id())),
                            "tools": digest([]),
                            "messages": [digest(m) for m in history + [{"role": "user", "content": current}]],
                            "stable": [],
                            "segment": "gate",
                        },
                    )
                    started = time.monotonic()
                    gate = None
                    try:
                        gate = await self.gateway.gate(event, provider, history, current)
                    finally:
                        await self.record_usage(
                            selection,
                            "gate",
                            provider,
                            getattr(gate, "usage", None),
                            time.monotonic() - started,
                            image_count(history, current),
                            "completed" if gate else "error",
                        )
                    if not self.participation.accepts(gate.completion_text):
                        return
                gid = await self.journal.begin(anchor["room"], anchor["seq"], event.unified_msg_origin)
                if gid is None:
                    return
                state = TurnState(
                    selection,
                    gid,
                    provider,
                    auto=auto,
                    route=route,
                    started=time.monotonic(),
                    media_leases=leased,
                )

                async def public_sent(chain, receipt, platform_id):
                    await self.public_sent(event, state, chain, receipt, platform_id)

                state.transport = self.gateway.track_transport(event, self.journal, gid, public_sent)
                event.set_extra(OWNER, state)
                text = "\n".join(p["text"] for p in anchor["parts"] if p["type"] == "text").strip()
                if not text:
                    poke = next((p for p in anchor["parts"] if p["type"] == "poke"), None)
                    text = (
                        "[戳一戳事件=" + json.dumps(poke, ensure_ascii=False) + "]"
                        if poke
                        else "[仅附件或@消息]"
                    )
                event.set_extra(OWNER + "_retrieval", text)
                request = self.gateway.prepare(
                    event,
                    text,
                    await self.gateway.conversation(event),
                    event.get_extra(OWNER + "_reverse_poke", False),
                )

                async def record_stats(stats, response, aborted):
                    await self.record_usage(
                        selection,
                        "reply",
                        provider,
                        stats.token_usage,
                        time.monotonic() - state.started,
                        image_count(state.request.contexts, state.request.extra_user_content_parts)
                        if state.request
                        else 0,
                        "aborted"
                        if aborted
                        else "error"
                        if response and response.role == "err"
                        else "completed",
                    )

                async with contextlib.aclosing(
                    self.gateway.execute(event, request, provider, record_stats)
                ) as execution:
                    async for _ in execution:
                        yield None
                completed = True
                await self.journal.settle(gid, completed=True)
        except ContextLimit as exc:
            if not auto:
                yield event.plain_result(str(exc))
            else:
                self.logger.info("[NativeMM] 主动候选跳过，预算或能力不满足")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.error("[NativeMM] 回复失败 type=%s", type(exc).__name__, exc_info=True)
            if gid:
                await self.journal.finish(gid, "failed")
            if not auto:
                yield event.plain_result("群聊上下文处理失败，请查看 NativeMM 日志。")
            completed = True
        finally:
            if state and state.transport:
                state.transport.restore()
            if gid:
                await self.journal.settle(gid, completed=completed)
            self.media.release(leased)
            self.running_tasks.discard(task)

    async def record_usage(self, selection, phase, provider, usage, elapsed, images, status):
        try:
            await self.journal.trace(
                room=selection.anchor["room"],
                anchor=selection.anchor["seq"],
                phase=phase,
                provider=provider.provider_config.get("id"),
                input_tokens=(usage.input_other + usage.input_cached) if usage else None,
                cached_tokens=usage.input_cached if usage else None,
                output_tokens=usage.output if usage else None,
                elapsed=elapsed,
                images=images,
                selected=len(selection.chosen_events),
                dropped=len(selection.events) - len(selection.chosen_events),
                rebases=selection.rebases,
                status=status,
            )
        except Exception as exc:
            self.logger.warning("[NativeMM] 诊断记录失败 type=%s", type(exc).__name__)

    async def snapshot(self, event, req):
        if state := event.get_extra(OWNER):
            state.snapshot = request_snapshot(req)

    async def inject(self, event, req):
        state = event.get_extra(OWNER)
        if not state:
            return
        try:
            maximum = int(state.provider.provider_config.get("max_context_tokens", 0) or 0)

            async def summarize(rules, data):
                response, start = None, time.monotonic()
                try:
                    response = await self.gateway.summarize(event, state.provider, rules, data)
                    await self.journal.diagnose(
                        state.selection.anchor["room"],
                        digest(event.unified_msg_origin),
                        "summary",
                        {
                            "system": digest(rules),
                            "tools": digest([]),
                            "messages": [digest(data)],
                            "stable": [],
                            "segment": state.selection.segment_id,
                        },
                    )
                    return response.completion_text
                finally:
                    await self.record_usage(
                        state.selection,
                        "summary",
                        state.provider,
                        getattr(response, "usage", None),
                        time.monotonic() - start,
                        0,
                        "completed" if response else "error",
                    )

            self.gateway.filter_tools(event, req)
            _, current = await rewrite(
                req,
                state.selection,
                self.selector,
                state.snapshot,
                event.get_extra(OWNER + "_retrieval"),
                maximum,
                segments=self.segments,
                scope_policy=self.gateway.scope_policy(event, state.provider),
                summarize=summarize,
                route=state.route,
            )
            await self.journal.link(
                state.selection.anchor["room"],
                state.selection.anchor["seq"],
                [
                    e["seq"]
                    for e in state.selection.chosen_events
                    if e["seq"] in state.selection.protected
                    or bool(set(image_ids(e["parts"])) & state.selection.chosen_media)
                ],
            )
            extra_leases = state.selection.chosen_media - set(state.media_leases)
            self.media.lease(extra_leases)
            state.media_leases.extend(extra_leases)
            count = image_count(req.contexts, req.extra_user_content_parts)
            if count:
                self.gateway.validate_images(event, state.provider)
            state.request = req
            state.current = [
                {"type": "text", "text": p["text"]}
                if p["type"] == "text"
                else {"type": "text", "text": "[图片已保存于插件附件日志]"}
                for p in current
            ]
            self.logger.info(
                "[NativeMM] anchor=%s trigger_seq=%s view_seq=%s events=%s images=%s tokens_est=%s rebases=%s trigger=%s",
                state.selection.anchor["event_id"],
                state.selection.anchor["seq"],
                state.selection.view_seq,
                [e["event_id"] for e in state.selection.chosen_events],
                count,
                state.selection.tokens,
                state.selection.rebases,
                state.route.reason if state.route else "unknown",
            )
        except Exception as exc:
            await self.journal.finish(state.gid, "failed")
            self.logger.error("[NativeMM] 请求组装失败 type=%s", type(exc).__name__)
            await event.send(
                event.plain_result(str(exc) if isinstance(exc, ContextLimit) else "群聊请求组装失败。")
            )
            event.stop_event()

    async def agent_begin(self, event, run_context):
        if state := event.get_extra(OWNER):
            state.run_context = run_context
            state.baseline_protocol = protocol_messages(run_context.messages)

    async def agent_done(self, event, run_context, response):
        state = event.get_extra(OWNER)
        if not state or state.baseline_protocol is None:
            return
        event.set_extra(OWNER + "_done", True)
        tail = additions(protocol_messages(run_context.messages), state.baseline_protocol)
        for message in tail:
            if isinstance(message.get("content"), list):
                for index, part in enumerate(message["content"]):
                    if part.get("type") == "image_url":
                        mid = await self.media.capture(
                            state.selection.anchor["room"], part["image_url"]["url"]
                        )
                        message["content"][index] = {"type": "journal_image", "media_id": mid}
        success = response is not None and response.role in {"assistant", "tool"}
        await self.journal.finish(
            state.gid, "generated" if success else "failed", tail, str(event.get_self_id())
        )
        self.gateway.mirror(
            run_context,
            [{"role": "user", "content": state.current or [{"type": "text", "text": "[群聊消息]"}]}]
            + state.public_messages,
        )

    async def public_sent(self, event, state, chain, receipt, platform_id):
        parts = await self.adapter.normalize(event, list(chain.chain), state.selection.anchor["room"])
        public = await self.journal.publish(state.gid, receipt, parts, str(event.get_self_id()), platform_id)
        if public:
            # The core mirror must also contain only observed public deliveries.
            texts = [p["text"] for p in public["parts"] if p["type"] == "text"]
            state.public_messages.append(
                {"role": "assistant", "content": "\n".join(texts) or "[机器人已发送附件]"}
            )
            if state.run_context and event.get_extra(OWNER + "_done", False):
                self.gateway.mirror(
                    state.run_context,
                    [{"role": "user", "content": state.current or [{"type": "text", "text": "[群聊消息]"}]}]
                    + state.public_messages,
                )

    async def delivered(self, event):
        state = event.get_extra(OWNER)
        result = self.gateway.receipt(event)
        if (
            not state
            or not state.transport
            or state.transport.successes == 0
            or result is None
            or any(receipt is result for receipt in state.receipts)
        ):
            return
        state.receipts.append(result)
        if not state.after_poke_attempted:
            state.after_poke_attempted = True
            if (
                self.config["enable_poke_after_reply"]
                and poke_scope(self.config, event)
                and random.random() < probability(self.config, "poke_after_reply_probability", 0.15)
            ):
                await asyncio.sleep(self.config["poke_after_reply_delay"])
                if not self.closing:
                    await self.poke(event, state.selection.anchor, "after_reply", state.gid)

    async def poke(self, event, anchor, reason, generation=None):
        try:
            if not self.closing and await send_poke(event, anchor["sender"]):
                await self.journal.add(
                    anchor["room"],
                    "poke:" + (generation or anchor["event_id"]) + ":" + reason,
                    event.get_self_id(),
                    "bot",
                    [{"type": "poke", "actor_id": str(event.get_self_id()), "target_id": anchor["sender"]}],
                    kind="action",
                )
                return True
        except Exception as exc:
            self.logger.warning("[NativeMM] 戳一戳失败 type=%s", type(exc).__name__)
        return False

    async def status(self, event):
        if self.enabled(event):
            yield event.plain_result(
                "NativeMM v2.2.1\n"
                + json.dumps(await self.journal.status(room_key(event)), ensure_ascii=False)
            )
            event.stop_event()

    async def reset(self, event):
        if self.enabled(event):
            async with self.room(room_key(event)):
                await self.journal.reset(room_key(event))
            yield event.plain_result("本群 NativeMM 上下文已重置；原始日志和附件按保留期清理。")
            event.stop_event()
