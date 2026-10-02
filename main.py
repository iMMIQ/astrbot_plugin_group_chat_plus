"""Native multimodal Chat Plus. Legacy utils/private_chat/web are not loaded."""
from __future__ import annotations

import asyncio
import contextlib
import json
import random
import sys
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import Message
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.star.star_handler import star_handlers_registry
from astrbot.core.star.star_tools import StarTools

from .multimodal.adapter import PlatformAdapter, image_ids, part_type, room_key
from .multimodal.bridge import GROUP_RULES, additions, protocol_messages, request_snapshot, restore_legacy_wrappers, rewrite
from .multimodal.context import ContextLimit, ContextSelector
from .multimodal.media import MediaStore
from .multimodal.runner import execute, strict_provider
from .multimodal.store import Journal

OWNER = "_native_multimodal"


class ChatPlus(Star):
    """原生多模态群聊。/mmstatus 查看房间状态；管理员 /mmreset 重置房间上下文。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.root = StarTools.get_data_dir("astrbot_plugin_group_chat_plus") / "multimodal_v1"
        self.journal = Journal(self.root)
        self.media = MediaStore(self.journal, self.root / "media", config)
        self.adapter = PlatformAdapter(self.journal, self.media)
        self.selector = ContextSelector(self.journal, self.media, config)
        self.locks, self.last_auto = {}, {}
        self.cleanup_task = None
        self.running_tasks = set()
        self.closing = False

    async def initialize(self):
        await self.media.start()
        restored = restore_legacy_wrappers(star_handlers_registry)
        self.cleanup_task = asyncio.create_task(self._cleanup())
        logger.info("[NativeMM] v2.0.0 原图上下文已加载；旧钩子包装恢复=%s，主动参与=%s",
                    restored, bool(self.config.get("auto_reply_enabled", False)))

    async def _cleanup(self):
        while True:
            try:
                self.media.cleanup()
            except Exception as exc:
                logger.warning("[NativeMM] 清理失败 type=%s", type(exc).__name__)
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
        self.journal.close()

    def _enabled(self, event):
        groups = [str(g) for g in self.config.get("enabled_groups", [])]
        return (not self.closing and bool(self.config.get("enable_group_chat", True))
                and bool(event.get_group_id()) and (not groups or str(event.get_group_id()) in groups)
                and str(event.get_sender_id()) != str(event.get_self_id()))

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=sys.maxsize + 10)
    async def capture(self, event: AstrMessageEvent):
        if not self._enabled(event):
            return
        event.set_extra("_group_context_flow_injected", True)
        anchor, inserted = await self.adapter.ingest(event)
        event.set_extra(OWNER + "_anchor", anchor)
        event.set_extra(OWNER + "_duplicate", not inserted)
        messages = event.get_messages()
        # Own pure @ before the builtin group waiter can capture another user.
        if (len(messages) == 1 and part_type(messages[0]) == "at"
                and str(messages[0].qq) == str(event.get_self_id())):
            async with contextlib.aclosing(self._reply(event, anchor, pure_mention=True)) as flow:
                async for result in flow:
                    yield result
            event.stop_event()

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-sys.maxsize)
    async def respond(self, event: AstrMessageEvent):
        anchor = event.get_extra(OWNER + "_anchor")
        if not anchor or event.is_stopped() or event._has_send_oper or event.call_llm:
            return
        explicit = bool(event.is_at_or_wake_command)
        if not explicit:
            if not self.config.get("auto_reply_enabled", False):
                return
            if time.monotonic() - self.last_auto.get(anchor["room"], -1e9) < float(self.config.get("auto_reply_cooldown", 60)):
                return
            if random.random() >= float(self.config.get("auto_candidate_probability", 0.02)):
                return
        async with contextlib.aclosing(self._reply(event, anchor, auto=not explicit)) as flow:
            async for result in flow:
                yield result

    async def _provider(self, event):
        provider_id = self.config.get("provider_id", "")
        if provider_id:
            event.set_extra("selected_provider", provider_id)
            return self.context.get_provider_by_id(provider_id)
        return await self.context.get_using_provider_async(event.unified_msg_origin)

    async def _conversation(self, event):
        manager = self.context.conversation_manager
        cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
        if not cid:
            cid = await manager.new_conversation(event.unified_msg_origin, platform_id=event.get_platform_id())
        return await manager.get_conversation(event.unified_msg_origin, cid)

    async def _reply(self, event, anchor, pure_mention=False, auto=False):
        event.call_llm = True
        if event.get_extra(OWNER + "_duplicate", False):
            return
        task = asyncio.current_task()
        self.running_tasks.add(task)
        leased, gid = [], None
        try:
            async with self.locks.setdefault(anchor["room"], asyncio.Lock()):
                if self.closing:
                    return
                if pure_mention:
                    await asyncio.sleep(float(self.config.get("mention_wait_seconds", 1)))
                    after = self.journal.recent(anchor["room"], 2**63 - 1, anchor["received"], 20)
                    batch = [e for e in after if e["sender"] == anchor["sender"] and e["seq"] > anchor["seq"]]
                    if batch:
                        anchor = batch[-1]
                        event.set_extra(OWNER + "_anchor", anchor)
                selection = self.selector.candidates(anchor)
                leased = list({mid for e in selection.events for mid in image_ids(e["parts"])})
                self.media.lease(leased)
                await self.media.wait(leased)
                provider = await self._provider(event)
                if provider is None:
                    raise ContextLimit("没有可用的聊天模型。")
                if leased:
                    if "image" not in provider.provider_config.get("modalities", []):
                        raise ContextLimit("当前模型未声明图片输入能力，请选择支持图片的模型。")
                    cfg = self.context.get_config(umo=event.unified_msg_origin)
                    fallback_ids = cfg.get("agent_runner", {}).get("config", {}).get("model", {}).get("fallback_provider_ids", [])
                    for pid in fallback_ids:
                        fallback = self.context.get_provider_by_id(pid)
                        if fallback and "image" not in fallback.provider_config.get("modalities", []):
                            raise ContextLimit("备用模型不支持图片，请先调整备用模型配置。")
                if auto:
                    if time.monotonic() - self.last_auto.get(anchor["room"], -1e9) < float(self.config.get("auto_reply_cooldown", 60)):
                        return
                    self.last_auto[anchor["room"]] = time.monotonic()
                    history, current = await self.selector.assemble(selection, fixed_tokens=1000)
                    gate = await asyncio.wait_for(strict_provider(provider).text_chat(
                        prompt="", contexts=history + [{"role": "user", "content": current}],
                        system_prompt=GROUP_RULES + '\n判断是否值得主动参与，严格只返回 JSON {"reply":true或false}。不使用工具。',
                    ), 15)
                    if json.loads(gate.completion_text).get("reply") is not True:
                        return
                gid = self.journal.begin(anchor["room"], anchor["seq"])
                if gid is None:
                    return
                event.set_extra(OWNER, {"selection": selection, "gid": gid, "leased": leased,
                                        "provider": provider, "auto": auto})
                text = "\n".join(p["text"] for p in anchor["parts"] if p["type"] == "text").strip() or "[仅附件或@消息]"
                event.set_extra(OWNER + "_retrieval", text)
                request = event.request_llm(prompt=text, contexts=[], conversation=await self._conversation(event))
                async with contextlib.aclosing(execute(event, self.context, request, provider)) as execution:
                    async for _ in execution:
                        yield None
                self.journal.settle(gid)
        except ContextLimit as exc:
            if not auto:
                yield event.plain_result(str(exc))
            else:
                logger.info("[NativeMM] 主动候选跳过，预算或能力不满足")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[NativeMM] 回复失败 type=%s", type(exc).__name__, exc_info=True)
            if not auto:
                yield event.plain_result("群聊上下文处理失败，请查看 NativeMM 日志。")
        finally:
            if gid:
                self.journal.settle(gid)
            self.media.release(leased)
            self.running_tasks.discard(task)

    @filter.on_llm_request(priority=sys.maxsize)
    async def snapshot(self, event: AstrMessageEvent, req: ProviderRequest):
        if state := event.get_extra(OWNER):
            state["snapshot"] = request_snapshot(req)

    @filter.on_llm_request(priority=-sys.maxsize)
    async def inject(self, event: AstrMessageEvent, req: ProviderRequest):
        state = event.get_extra(OWNER)
        if not state:
            return
        try:
            maximum = int(state["provider"].provider_config.get("max_context_tokens", 0) or 0)
            _, current = await rewrite(req, state["selection"], self.selector, state["snapshot"],
                                       event.get_extra(OWNER + "_retrieval"), maximum)
            state["request"] = req
            state["current"] = [{"type": "text", "text": p["text"]} if p["type"] == "text"
                                else {"type": "text", "text": "[图片已保存于插件附件日志]"} for p in current]
            selection = state["selection"]
            count = sum(p.get("type") == "image_url" for msg in req.contexts if isinstance(msg.get("content"), list) for p in msg["content"] if isinstance(p, dict)) + sum(
                (p.get("type") if isinstance(p, dict) else p.type) == "image_url" for p in req.extra_user_content_parts)
            logger.info("[NativeMM] anchor=%s snapshot=%s events=%s images=%s tokens_est=%s",
                        selection.anchor["event_id"], selection.anchor["seq"],
                        [e["event_id"] for e in selection.events], count, selection.tokens)
        except Exception as exc:
            self.journal.finish(state["gid"], "failed")
            logger.error("[NativeMM] 请求组装失败 type=%s", type(exc).__name__)
            await event.send(event.plain_result(str(exc) if isinstance(exc, ContextLimit) else "群聊请求组装失败。"))
            event.stop_event()

    @filter.on_agent_begin(priority=-sys.maxsize)
    async def agent_begin(self, event, run_context):
        if state := event.get_extra(OWNER):
            state["baseline_protocol"] = protocol_messages(run_context.messages)

    @filter.on_agent_done(priority=-sys.maxsize)
    async def agent_done(self, event, run_context, response):
        state = event.get_extra(OWNER)
        if not state or "baseline_protocol" not in state:
            return
        event.set_extra(OWNER + "_done", True)
        tail = additions(protocol_messages(run_context.messages), state["baseline_protocol"])
        mirror_tail = json.loads(json.dumps(tail))
        for message in tail:
            if isinstance(message.get("content"), list):
                for index, part in enumerate(message["content"]):
                    if part.get("type") == "image_url":
                        mid = self.media.capture(state["selection"].anchor["room"], part["image_url"]["url"])
                        message["content"][index] = {"type": "journal_image", "media_id": mid}
        for message in mirror_tail:
            if isinstance(message.get("content"), list):
                message["content"] = [{"type": "text", "text": "[工具图片保存在插件附件日志]"} if p.get("type") == "image_url" else p for p in message["content"]]
        success = response is not None and response.role in {"assistant", "tool"}
        self.journal.finish(state["gid"], "generated" if success else "failed", tail, str(event.get_self_id()))
        # Agent has completed: keep a latest-turn framework mirror without wire images.
        mirror = [{"role": "user", "content": state.get("current", [{"type": "text", "text": "[群聊消息]"}])}] + mirror_tail
        run_context.messages[:] = [Message.model_validate(msg) for msg in mirror]

    @filter.after_message_sent(priority=-sys.maxsize)
    async def delivered(self, event):
        state = event.get_extra(OWNER)
        result = event.get_result()
        if state and result and result.is_model_result():
            self.journal.sent(state["gid"])

    @filter.command("mmstatus")
    async def status(self, event):
        if not self._enabled(event):
            return
        yield event.plain_result("NativeMM v2.0.0\n" + json.dumps(self.journal.status(room_key(event)), ensure_ascii=False))
        event.stop_event()

    @filter.command("mmreset")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def reset(self, event):
        if not self._enabled(event):
            return
        room = room_key(event)
        async with self.locks.setdefault(room, asyncio.Lock()):
            self.journal.reset(room)
        yield event.plain_result("本群 NativeMM 上下文已重置；原始日志和附件按保留期清理。")
        event.stop_event()
