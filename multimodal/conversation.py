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
from .adapters.poke import description as poke_description
from .adapters.poke import in_scope as poke_scope
from .adapters.poke import notice as poke_notice
from .adapters.poke import probability
from .adapters.poke import send as send_poke
from .bridge import additions, build_gate_rules, protocol_messages, request_snapshot, rewrite
from .budget import BudgetEstimator, calibrated, profile_key, text_tokens
from .config import resolve_tz
from .context import ContextLimit, ContextSelector
from .history import event_text, filter_eligible, identity_part, search, semantic_select
from .models import TurnState
from .output import strip_headers
from .participation import Participation
from .qq_faces import resolve_react_emoji, send_emoji_like
from .routing import TurnRoute, classify, identity
from .segments import MEMORY_RULES, SegmentManager, digest
from .storage.assets import MediaStore
from .storage.sqlite import Journal
from .wire import WireObserver

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


def _plain_text(parts) -> str:
    return "\n".join(p["text"] for p in parts if p["type"] == "text").strip()


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
        self._muted_until = {}
        self._debounce_seq = {}
        self._memory_inflight = set()

    def _gcfg(self, event_or_gid):
        gid = event_or_gid.get_group_id() if hasattr(event_or_gid, "get_group_id") else event_or_gid
        if hasattr(self.config, "for_group"):
            return self.config.for_group(gid)
        return self.config

    async def initialize(self):
        await self.journal.ready()
        await self.media.start()
        restored = self.gateway.restore_wrappers()
        self.cleanup_task = asyncio.create_task(self._cleanup())
        self.logger.info(
            "[NativeMM] v2.2.6 已加载；旧包装恢复=%s，主动参与=%s",
            restored,
            self.config["auto_reply_enabled"],
        )

    async def _cleanup(self):
        while True:
            try:
                await self.media.cleanup()
                self.participation.cleanup()
                self.adapter.cleanup()
                now = time.monotonic()
                cutoff = now - 300
                self.rooms = {
                    room: slot for room, slot in self.rooms.items() if slot.pending or slot.touched >= cutoff
                }
                self._muted_until = {room: until for room, until in self._muted_until.items() if until > now}
                if len(self._debounce_seq) > 2048:
                    self._debounce_seq.clear()
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
        gid = str(event.get_group_id() or "")
        groups = [str(g) for g in self.config["enabled_groups"]]
        disabled = {str(g) for g in self.config.get("disabled_groups", [])}
        return (
            not self.closing
            and self.config["enable_group_chat"]
            and bool(gid)
            and gid not in disabled
            and (not groups or gid in groups)
            and str(event.get_sender_id()) != str(event.get_self_id())
        )

    def is_muted(self, room):
        return time.monotonic() < self._muted_until.get(room, 0.0)

    def mark_muted(self, room, exc):
        backoff = int(self.config.get("mute_backoff_seconds", 1800) or 0)
        if backoff <= 0:
            return
        msg = str(exc)
        if "retcode=1200" in msg and ('"result": 120' in msg or '"result":120' in msg or "sendMsg" in msg):
            self._muted_until[room] = time.monotonic() + backoff
            self.logger.warning("[NativeMM] 群发送受限(retcode=1200)，退避 %ss room=%s", backoff, room)

    def _aux_provider(self, config_key, fallback):
        pid = str(self.config.get(config_key, "") or "").strip()
        if pid:
            found = self.gateway.provider_by_id(pid)
            if found is not None:
                return found
            self.logger.warning("[NativeMM] 辅助模型 %s=%s 不可用，回退主模型", config_key, pid)
        return fallback

    async def capture(self, event):
        if not self.enabled(event):
            return
        task = asyncio.current_task()
        self.running_tasks.add(task)
        try:
            gcfg = self._gcfg(event)
            poke = poke_notice(event)
            raw = getattr(event.message_obj, "raw_message", None)
            if poke:
                if not accept_poke(gcfg, event, poke):
                    return
            elif hasattr(raw, "get") and raw.get("post_type") == "notice":
                return
            event.set_extra("_group_context_flow_injected", True)
            anchor, inserted = await self.adapter.ingest(event)
            event.set_extra(OWNER + "_anchor", anchor)
            event.set_extra(OWNER + "_duplicate", not inserted)
            if inserted and not poke:
                ignored = {str(x) for x in gcfg.get("ignored_bot_ids", [])}
                if str(anchor["sender"]) not in ignored:
                    self.participation.observe_member(
                        anchor["room"],
                        anchor["sender"],
                        _plain_text(anchor["parts"]),
                        anchor["received"],
                    )
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
                if not poke:
                    key = (anchor["room"], str(anchor["sender"]))
                    if anchor["seq"] > self._debounce_seq.get(key, 0):
                        self._debounce_seq[key] = anchor["seq"]
            pure = (
                len(event.get_messages()) == 1
                and part_type(event.get_messages()[0]) == "at"
                and str(event.get_messages()[0].qq) == str(event.get_self_id())
            )
            if poke and poke["target_id"] == str(event.get_self_id()):
                poke_cd = float(gcfg.get("poke_reply_cooldown", 15.0) or 0.0)
                last_map = getattr(self, "_last_poke_reply_ts", None)
                if last_map is None:
                    last_map = {}
                    self._last_poke_reply_ts = last_map
                now_ts = time.monotonic()
                rkey = str(anchor.get("room") or event.get_group_id() or "")
                if poke_cd > 0 and now_ts - last_map.get(rkey, 0.0) < poke_cd:
                    event.stop_event()
                    return
                last_map[rkey] = now_ts
                event.is_at_or_wake_command = True
                if inserted and random.random() < probability(gcfg, "poke_reverse_on_poke_probability", 0):
                    if await self.poke(event, anchor, "reverse"):
                        event.set_extra(OWNER + "_reverse_poke", True)
            elif not pure:
                return
            if self.is_muted(anchor["room"]):
                event.stop_event()
                return
            async with contextlib.aclosing(self.reply(event, anchor, pure_mention=pure)) as flow:
                async for result in flow:
                    yield result
            event.stop_event()
        finally:
            self.running_tasks.discard(task)

    async def _send_plus_one(self, event, anchor, text: str, gcfg):
        room = anchor["room"]
        self.participation.mark_plus_one(room, text)
        await asyncio.sleep(random.uniform(0.9, 2.2))
        if self.closing or self.is_muted(room):
            return
        newer = await self.journal.count_member_after(room, anchor["seq"])
        if newer > 1:
            return
        try:
            platform_id = await self.gateway.send_bare_text(event, text)
        except Exception as exc:
            self.mark_muted(room, exc)
            self.logger.warning("[NativeMM] +1 复读发送失败 type=%s", type(exc).__name__)
            return
        eid = str(platform_id) if platform_id else ("plus1:" + str(anchor["seq"]))
        await self.journal.add(
            room,
            eid,
            str(event.get_self_id()),
            "bot",
            [{"type": "text", "text": text}],
            kind="self",
        )
        self.participation.observe_bot_reply(room, anchor["sender"], anchor["seq"], auto=True)
        self.logger.info("[NativeMM] +1 跟队复读 room=%s text=%s", room, text[:40])

    async def respond(self, event):
        anchor = event.get_extra(OWNER + "_anchor")
        if not anchor or self.gateway.already_handled(event) or self.is_muted(anchor["room"]):
            return
        gcfg = self._gcfg(event)
        explicit = event.get_extra(OWNER + "_route") is not None or bool(event.is_at_or_wake_command)
        if not explicit:
            if any(p["type"] == "poke" for p in anchor["parts"]):
                return
            plain = _plain_text(anchor["parts"])
            plus_mode = self.participation.detect_plus_one(anchor["room"], anchor["sender"], plain, gcfg)
            if plus_mode == "repeat":
                await self._send_plus_one(event, anchor, plain, gcfg)
                return
            if plus_mode == "break":
                self.participation.mark_plus_one(anchor["room"], plain)
                async with contextlib.aclosing(
                    self.reply(
                        event,
                        anchor,
                        auto=True,
                        auto_reason="plus_one_break",
                        trigger_hint="群里大家正在+1复读刷屏，如果你有好玩的吐槽可以花式打断队形(reply)，否则跳过(skip)。",
                    )
                ) as flow:
                    async for result in flow:
                        yield result
                return
            signal = self.participation.classify_passive(
                anchor["room"], anchor, str(event.get_self_id()), gcfg
            )
            if not signal:
                return
            hint_map = {
                "thread_continue": "续聊判断：你刚和该群友聊过，对方没@你但可能仍在接着跟你聊；如果是在接着聊或值得接茬选reply，否则skip。",
                "nickname_mention": "外号提及：群友句子里提到了你的外号，判断是否适合接话(reply)、顺手贴表情(react)或跳过(skip)。",
                "ambient": "",
            }
            async with contextlib.aclosing(
                self.reply(
                    event,
                    anchor,
                    auto=True,
                    auto_reason=signal,
                    trigger_hint=hint_map.get(signal, ""),
                )
            ) as flow:
                async for result in flow:
                    yield result
            return
        async with contextlib.aclosing(self.reply(event, anchor, auto=False)) as flow:
            async for result in flow:
                yield result

    async def _absorb_followup(self, anchor, deadline: float):
        await self.adapter.settle_arrivals(
            anchor["room"],
            anchor["sender"],
            anchor["received"],
            deadline,
            self.config["media_wait_seconds"],
        )
        after = await self.journal.recent(anchor["room"], 2**63 - 1, anchor["received"], 20)
        return [
            e
            for e in after
            if e["sender"] == anchor["sender"]
            and e["seq"] > anchor["seq"]
            and e["kind"] == "member"
            and e["received"] <= deadline
        ]

    async def reply(
        self,
        event,
        anchor,
        pure_mention=False,
        auto=False,
        auto_reason=None,
        trigger_hint="",
    ):
        event.call_llm = True
        if event.get_extra(OWNER + "_duplicate", False):
            return
        gcfg = self._gcfg(event)
        is_poke = any(p["type"] == "poke" for p in anchor["parts"])
        route_reason = (
            auto_reason
            if (auto and auto_reason in {"thread_continue", "nickname_mention", "plus_one_break"})
            else ("auto_approved" if auto else "framework_wake")
        )
        route_hint = (
            "群里大家正在+1刷屏，简短有趣地打断队形或吐槽一句。" if auto_reason == "plus_one_break" else ""
        )
        route = event.get_extra(OWNER + "_route") or TurnRoute(
            str(event.get_self_id()), anchor["event_id"], route_reason, hint=route_hint
        )
        task = asyncio.current_task()
        self.running_tasks.add(task)
        leased, gid, completed, state = [], None, False, None
        released = asyncio.Event()
        try:
            if self.closing or self.is_muted(anchor["room"]):
                return

            # Explicit short debounce/coalescing for same-sender multi-bubble bursts (e.g. text + image).
            if not auto and not pure_mention and not is_poke:
                debounce = float(gcfg.get("input_debounce_seconds", 0.0) or 0.0)
                if debounce > 0:
                    await asyncio.sleep(debounce)
                    if self.closing or self.is_muted(anchor["room"]):
                        return
                    key = (anchor["room"], str(anchor["sender"]))
                    if self._debounce_seq.get(key, 0) > anchor["seq"]:
                        return
                    deadline = time.time()
                    batch = await self._absorb_followup(anchor, deadline)
                    if batch:
                        anchor = batch[-1]
                        event.set_extra(OWNER + "_anchor", anchor)

            if auto:
                signal = auto_reason or "ambient"
                if not self.participation.claim_signal(anchor["room"], signal, gcfg):
                    return
                main_provider = await self.gateway.provider(event)
                if main_provider is None:
                    raise ContextLimit("没有可用的聊天模型。")
                gate_provider = self._aux_provider("gate_provider_id", main_provider)
                gate_selection = await self.selector.candidates(anchor)
                gate_images = bool(gcfg.get("gate_include_images", False))
                gate_leased = (
                    list({mid for e in gate_selection.events for mid in image_ids(e["parts"])})
                    if gate_images
                    else []
                )
                if gate_leased:
                    self.media.lease(gate_leased)
                try:
                    if gate_leased:
                        await self.media.wait(gate_leased)
                    gate_model = gate_provider.get_model()
                    gate_profile = profile_key(gate_provider.provider_config, gate_model)
                    gate_selection.estimator = calibrated(
                        gate_provider.provider_config,
                        await self.journal.budget_samples(gate_profile),
                        gate_model,
                    )
                    history, current = await self.selector.assemble(
                        gate_selection,
                        fixed_tokens=1000,
                        total_limit=gcfg["gate_token_budget"],
                        freeze=False,
                        include_images=gate_images,
                    )
                    if gate_images and image_count(history, current):
                        self.gateway.validate_images(event, gate_provider)
                    react_allowed = bool(
                        gcfg.get("emoji_react_enabled", False)
                    ) and self.participation.can_react(anchor["room"], gcfg)
                    persona_hint = str(gcfg.get("gate_persona_hint", "") or "")
                    gate_rules = build_gate_rules(
                        persona_hint=persona_hint,
                        react_enabled=react_allowed,
                        trigger_hint=trigger_hint,
                    )
                    await self.journal.diagnose(
                        anchor["room"],
                        digest({"provider": gate_provider.provider_config.get("id")}),
                        "gate",
                        {
                            "system": digest(gate_rules + "\n" + identity(event.get_self_id())),
                            "tools": digest([]),
                            "messages": [digest(m) for m in history + [{"role": "user", "content": current}]],
                            "stable": [],
                            "segment": "gate",
                            "signal": signal,
                        },
                    )
                    started = time.monotonic()
                    gate = None
                    try:
                        gate = await self.gateway.gate(
                            event,
                            gate_provider,
                            history,
                            current,
                            persona_hint=persona_hint,
                            react_enabled=react_allowed,
                            trigger_hint=trigger_hint,
                        )
                    finally:
                        await self.record_usage(
                            gate_selection,
                            "gate",
                            gate_provider,
                            getattr(gate, "usage", None),
                            time.monotonic() - started,
                            image_count(history, current),
                            "completed" if gate else "error",
                        )
                    decision = self.participation.parse_gate(gate.completion_text if gate else "")
                    if decision["action"] == "react":
                        if react_allowed:
                            eid = resolve_react_emoji(decision.get("emoji"))
                            if eid and await send_emoji_like(event, anchor["event_id"], eid):
                                self.participation.mark_reacted(anchor["room"])
                                self.logger.info(
                                    "[NativeMM] 贴表情回应 room=%s msg=%s emoji=%s(%s)",
                                    anchor["room"],
                                    anchor["event_id"],
                                    decision.get("emoji"),
                                    eid,
                                )
                        return
                    if decision["action"] != "reply":
                        return
                finally:
                    if gate_leased:
                        self.media.release(gate_leased)

            async with self.room(anchor["room"]):
                if self.closing or self.is_muted(anchor["room"]):
                    return
                if auto:
                    base_max = int(gcfg.get("auto_stale_max_messages", 4) or 0)
                    max_msgs = (
                        (base_max + 2) if (base_max > 0 and auto_reason == "thread_continue") else base_max
                    )
                    max_sec = float(gcfg.get("auto_stale_max_seconds", 20.0) or 0.0)
                    newer = await self.journal.count_member_after(anchor["room"], anchor["seq"])
                    age = time.time() - float(anchor["received"])
                    if (max_msgs > 0 and newer >= max_msgs) or (max_sec > 0 and age >= max_sec):
                        self.logger.info(
                            "[NativeMM] 主动插话已过期放弃 room=%s newer=%s age=%.1fs",
                            anchor["room"],
                            newer,
                            age,
                        )
                        return
                elif not is_poke and not pure_mention:
                    key = (anchor["room"], str(anchor["sender"]))
                    if self._debounce_seq.get(key, 0) > anchor["seq"]:
                        return

                if pure_mention:
                    await asyncio.sleep(gcfg["mention_wait_seconds"])
                    deadline = time.time()
                    batch = await self._absorb_followup(anchor, deadline)
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
                estimates, budget_profiles = [], []
                for candidate in self.gateway.budget_providers(event, provider):
                    model = candidate.get_model()
                    profile = profile_key(candidate.provider_config, model)
                    estimates.append(
                        calibrated(
                            candidate.provider_config,
                            await self.journal.budget_samples(profile),
                            model,
                        )
                    )
                    budget_profiles.append({"profile": profile, **vars(estimates[-1])})
                selection.estimator = BudgetEstimator(
                    max(e.text_scale for e in estimates),
                    max(e.image_scale for e in estimates),
                    estimates[0].text_samples,
                    estimates[0].image_samples,
                )
                gid = await self.journal.begin(anchor["room"], anchor["seq"], event.unified_msg_origin)
                if gid is None:
                    return
                state = TurnState(
                    selection,
                    gid,
                    provider,
                    auto=auto,
                    route=route,
                    poke_succeeded=bool(event.get_extra(OWNER + "_reverse_poke", False)),
                    started=time.monotonic(),
                    media_leases=leased,
                    is_poke_trigger=is_poke,
                )

                async def public_sent(chain, receipt, platform_id):
                    await self.public_sent(event, state, chain, receipt, platform_id)

                async def intervening_count():
                    return await self.journal.count_member_after(anchor["room"], anchor["seq"])

                state.transport = self.gateway.track_transport(
                    event,
                    self.journal,
                    gid,
                    public_sent,
                    on_send_error=lambda exc: self.mark_muted(anchor["room"], exc),
                    turn_state=state,
                    config=gcfg,
                    intervening_fn=intervening_count,
                )
                event.set_extra(OWNER, state)
                event.set_extra("meme_manager_defer_selection", True)

                async def send_meme_later(chain):
                    t = asyncio.current_task()
                    self.running_tasks.add(t)
                    try:
                        await released.wait()
                        if self.closing or await self.journal.floor(anchor["room"]) >= anchor["seq"]:
                            return False
                        await state.transport.send_late(chain)
                        return True
                    finally:
                        if not self.closing:
                            await self.journal.settle(gid, completed=True)
                        self.running_tasks.discard(t)

                event.set_extra("meme_manager_deferred_send", send_meme_later)
                text = _plain_text(anchor["parts"])
                if not text:
                    poke = next((p for p in anchor["parts"] if p["type"] == "poke"), None)
                    text = poke_description(poke) if poke else "[仅附件或@消息]"
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

                async def record_wire(payload):
                    payload.update(
                        anchor=selection.anchor["seq"],
                        generation=gid,
                        segment=selection.segment_id,
                        rollover=selection.rollover,
                        text_scale=selection.estimator.text_scale,
                        image_scale=selection.estimator.image_scale,
                        budget_profiles=budget_profiles,
                    )
                    await self.journal.diagnose(
                        selection.anchor["room"],
                        payload["profile"],
                        "wire_reply_" + payload["stage"],
                        payload,
                    )

                observer = WireObserver(record_wire, int(gcfg["image_token_reserve"]))

                async with contextlib.aclosing(
                    self.gateway.execute(event, request, provider, record_stats, observer)
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
            released.set()

    async def _distill_room_memory(self, room, archived_events, archived_summary, fallback_provider, gcfg):
        if room in self._memory_inflight or self.closing:
            return
        self._memory_inflight.add(room)
        task = asyncio.current_task()
        self.running_tasks.add(task)
        try:
            provider = self._aux_provider("summary_provider_id", fallback_provider)
            if provider is None:
                return
            ttl = float(gcfg.get("room_memory_ttl_days", 7) or 7) * 86400
            existing = await self.journal.room_memory(room, ttl) or {"facts": [], "members": {}}
            snippets = []
            for e in (archived_events or [])[-40:]:
                txt = _plain_text(e.get("parts") or [])
                if e.get("kind") == "self":
                    txt = strip_headers(txt)
                if txt:
                    snippets.append(
                        {"sender": str(e["sender"]), "name": str(e["name"])[:40], "text": txt[:200]}
                    )
            if not snippets and not archived_summary:
                return
            payload = {
                "existing_memory": existing,
                "recent_summary": (archived_summary or [])[-8:],
                "messages": snippets,
            }
            raw = await self.gateway.distill_memory(provider, MEMORY_RULES, payload)
            cleaned_raw = (raw or "").strip()
            if cleaned_raw.startswith("```"):
                cleaned_raw = cleaned_raw.strip("`").lstrip("jsonJSON").strip()
            decoded = json.loads(cleaned_raw)
            if not isinstance(decoded, dict):
                return
            facts = [
                str(x).strip()[:60] for x in (decoded.get("facts") or []) if isinstance(x, str) and x.strip()
            ][:6]
            members = {}
            for uid, note in (decoded.get("members") or {}).items():
                if isinstance(uid, str) and isinstance(note, str) and uid.strip() and note.strip():
                    members[uid.strip()] = note.strip()[:40]
                    if len(members) >= 12:
                        break
            result = {"facts": facts, "members": members}
            cap = int(gcfg.get("room_memory_token_budget", 480) or 480)
            # Trim members and facts fairly so facts are not wiped to zero by member notes.
            while text_tokens(result) > cap:
                if len(result["members"]) > 6 and len(result["members"]) >= len(result["facts"]) * 2:
                    first_k = next(iter(result["members"]))
                    result["members"].pop(first_k, None)
                elif len(result["facts"]) > 2:
                    result["facts"].pop(0)
                elif result["members"]:
                    first_k = next(iter(result["members"]))
                    result["members"].pop(first_k, None)
                elif result["facts"]:
                    result["facts"].pop(0)
                else:
                    break
            if result["facts"] or result["members"]:
                await self.journal.save_room_memory(room, result)
                self.logger.info(
                    "[NativeMM] 群印象已更新 room=%s facts=%d members=%d",
                    room,
                    len(result["facts"]),
                    len(result["members"]),
                )
        except Exception as exc:
            self.logger.debug("[NativeMM] 群印象提炼跳过 type=%s", type(exc).__name__)
        finally:
            self._memory_inflight.discard(room)
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
        gcfg = self._gcfg(event)
        try:
            maximum = int(state.provider.provider_config.get("max_context_tokens", 0) or 0)

            async def summarize(rules, data):
                summary_provider = self._aux_provider("summary_provider_id", state.provider)
                response, start = None, time.monotonic()
                try:
                    response = await self.gateway.summarize(event, summary_provider, rules, data)
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
                        summary_provider,
                        getattr(response, "usage", None),
                        time.monotonic() - start,
                        0,
                        "completed" if response else "error",
                    )

            self.gateway.filter_tools(event, req)
            anchor = state.selection.anchor
            senders = {anchor["sender"]}
            senders.update(e["sender"] for e in state.selection.events[-12:] if e["kind"] == "member")
            senders.update(p["target_id"] for p in anchor["parts"] if p["type"] == "mention")
            profiles = await self.journal.member_identities(anchor, senders)
            tz = resolve_tz(gcfg)

            async def history(source="group", mode="literal", **kwargs):
                if await self.journal.floor(anchor["room"]) >= anchor["seq"]:
                    return {"error": "本群上下文已重置，请重新发起查询。"}
                if mode not in {"literal", "semantic"}:
                    return {"error": "mode 必须为 literal 或 semantic。"}
                if source == "group":
                    kw = str(kwargs.get("keyword") or "").strip()
                    if mode == "semantic" and kw and hasattr(self.gateway, "semantic_rank"):
                        try:
                            lim = max(1, min(50, int(kwargs.get("limit", 15) or 15)))
                        except (TypeError, ValueError):
                            return {"error": "limit 必须是整数。"}
                        before_id = kwargs.get("before_id")
                        if before_id is not None:
                            try:
                                if isinstance(before_id, bool) or int(before_id) <= 0:
                                    raise ValueError
                                before_id = int(before_id)
                            except (TypeError, ValueError):
                                return {"error": "before_id 必须是正整数。"}
                        sender_q = str(kwargs.get("sender") or "").strip().casefold()
                        all_events, all_profiles = await self.journal.group_events(anchor)
                        eligible, resolved = filter_eligible(
                            all_events, all_profiles, before_id=before_id, sender=sender_q
                        )
                        items = []
                        for ev in eligible:
                            txt = f"{ev['name']}: {event_text(ev)}".strip()
                            if len(txt) > len(ev["name"]) + 2:
                                items.append((ev["seq"], txt))
                        ranked = await self.gateway.semantic_rank(kw, items, top_k=lim)
                        if ranked:
                            return semantic_select(
                                eligible,
                                all_profiles,
                                ranked,
                                source="group",
                                limit=lim,
                                keyword=kw,
                                resolved=resolved,
                                tz=tz,
                            )
                    return await self.journal.group_history(anchor, tz=tz, **kwargs)
                if source != "session":
                    return {"error": "source 必须为 group 或 session。"}
                floor = await self.journal.floor(anchor["room"])
                if floor:
                    return {"error": "本群已设置上下文边界，请使用 source=group 查询边界之后的记录。"}
                legacy = await self.gateway.session_history(event, anchor)
                if legacy is None:
                    return {"error": "框架会话历史未启用，请使用 source=group。"}
                events, legacy_profiles = legacy
                for uid, profile in profiles.items():
                    other = legacy_profiles.get(uid, {"sender_id": uid, "names": []})
                    legacy_profiles[uid] = {
                        "sender_id": uid,
                        "names": list(dict.fromkeys(profile["names"] + other["names"])),
                    }
                return search(events, legacy_profiles, source="session", tz=tz, **kwargs)

            self.gateway.add_history_tool(event, req, history)

            async def image(media_id):
                visible = {mid for item in state.selection.chosen_events for mid in image_ids(item["parts"])}
                if media_id not in visible:
                    return None
                self.media.lease([media_id])
                try:
                    return await self.media.data_uri(media_id, anchor["room"])
                finally:
                    self.media.release([media_id])

            self.gateway.add_image_tool(event, req, image)
            if (
                event.get_platform_name() == "aiocqhttp"
                and str(event.get_group_id()).isdecimal()
                and str(state.selection.anchor["sender"]).isdecimal()
                and poke_scope(gcfg, event)
                and getattr(event, "bot", None) is not None
            ):

                async def model_poke():
                    return await self.model_poke(event, state)

                self.gateway.add_poke_tool(event, req, model_poke)

            room_memory = None
            on_archive = None
            if gcfg.get("room_memory_enabled", False):
                ttl = float(gcfg.get("room_memory_ttl_days", 7) or 7) * 86400
                room_memory = await self.journal.room_memory(anchor["room"], ttl)

                def _schedule_archive(evts, summ):
                    if not self.closing and anchor["room"] not in self._memory_inflight:
                        asyncio.create_task(
                            self._distill_room_memory(anchor["room"], evts, summ, state.provider, gcfg)
                        )

                on_archive = _schedule_archive

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
                member_context=identity_part(profiles, anchor["sender"]),
                room_memory=room_memory,
                on_archive=on_archive,
            )
            await self.journal.link(
                state.selection.anchor["room"],
                state.selection.anchor["seq"],
                [
                    e["seq"]
                    for e in state.selection.chosen_events
                    if e["seq"] in state.selection.protected
                    or bool(set(image_ids(e["parts"])) & state.selection.primary_media)
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
        state.agent_finished = True
        event.set_extra(OWNER + "_done", True)
        tail = additions(protocol_messages(run_context.messages), state.baseline_protocol)
        for message in tail:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    fname = (tc.get("function") or {}).get("name") if isinstance(tc, dict) else None
                    if fname and fname != "native_poke_current_sender":
                        state.used_serious_tool = True
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
            texts = [p["text"] for p in public["parts"] if p["type"] == "text"]
            if any(text.strip() for text in texts):
                event.set_extra("meme_manager_text_delivered", True)
                state.reply_text_len += sum(len(t.strip()) for t in texts)
                self.participation.observe_bot_reply(
                    state.selection.anchor["room"],
                    state.selection.anchor["sender"],
                    state.selection.anchor["seq"],
                    auto=state.auto,
                    is_thread=bool(state.route and state.route.reason == "thread_continue"),
                )
            state.public_messages.append(
                {"role": "assistant", "content": "\n".join(texts) or "[机器人已发送附件]"}
            )
            if state.run_context and event.get_extra(OWNER + "_done", False):
                self.gateway.mirror(
                    state.run_context,
                    [{"role": "user", "content": state.current or [{"type": "text", "text": "[群聊消息]"}]}]
                    + state.public_messages,
                )

    async def decorate(self, event):
        if event.get_extra(OWNER):
            self.gateway.guard_result(event)
            self.gateway.decorate_bubbles(event, self._gcfg(event))

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
        gcfg = self._gcfg(event)
        if not state.after_poke_attempted:
            state.after_poke_attempted = True
            max_chars = int(gcfg.get("poke_after_reply_max_chars", 35) or 0)
            casual_enough = (not state.used_serious_tool) and (
                max_chars == 0 or state.reply_text_len <= max_chars
            )
            if (
                gcfg["enable_poke_after_reply"]
                and casual_enough
                and not state.poke_succeeded
                and not state.action_poke_attempted
                and poke_scope(gcfg, event)
                and random.random() < probability(gcfg, "poke_after_reply_probability", 0.15)
            ):
                await asyncio.sleep(gcfg["poke_after_reply_delay"])
                if not self.closing:
                    if not state.poke_succeeded and not state.action_poke_attempted:
                        state.action_poke_attempted = True
                        state.poke_succeeded = await self.poke(
                            event, state.selection.anchor, "after_reply", state.gid
                        )

    async def model_poke(self, event, state):
        gcfg = self._gcfg(event)
        if (
            self.closing
            or event.get_extra(OWNER) is not state
            or event.get_extra(OWNER + "_done", False)
            or not poke_scope(gcfg, event)
        ):
            return "戳一戳未执行：本轮已结束或不在启用范围内。"
        if state.poke_succeeded:
            return "本轮已成功戳过当前发送者，不重复执行。"
        if state.action_poke_attempted:
            return "本轮已尝试戳一戳，不重复执行；仅成功结果表示动作已完成。"
        state.action_poke_attempted = True
        state.poke_succeeded = await self.poke(event, state.selection.anchor, "model", state.gid)
        return "已成功向当前发送者戳一戳。" if state.poke_succeeded else "戳一戳失败，未确认成功。"

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
                "NativeMM v2.2.6\n"
                + json.dumps(await self.journal.status(room_key(event)), ensure_ascii=False)
            )
            event.stop_event()

    async def reset(self, event):
        if self.enabled(event):
            async with self.room(room_key(event)):
                await self.journal.reset(room_key(event))
            yield event.plain_result("本群 NativeMM 上下文已重置；原始日志和附件按保留期清理。")
            event.stop_event()
