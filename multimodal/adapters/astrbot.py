"""Use AstrBot's agent with per-request providers; never change global providers."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

from astrbot.api.star import Context
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.astr_main_agent import build_main_agent, collect_initial_request
from astrbot.core.pipeline.context import PipelineContext
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.pipeline.process_stage.follow_up import register_active_runner, unregister_active_runner
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
    _record_internal_agent_stats,
)
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.star.session_llm_manager import SessionServiceManager
from astrbot.core.star.star_handler import EventType


class StrictImageOpenAI(ProviderOpenAIOfficial):
    async def _fallback_to_text_only_and_retry(self, *args, **kwargs):
        raise ValueError("图片输入被服务商拒绝；NativeMM 已停止去图降级，请检查图片或模型配置。")


STRICT_CLASSES = {ProviderOpenAIOfficial: StrictImageOpenAI}


def strict_provider(provider):
    if not isinstance(provider, ProviderOpenAIOfficial):
        return provider
    # Reuse the owned HTTP client without opening/closing a global resource.
    # The isolated object overrides only text-only fallback, not a live handler.
    original_type = type(provider)
    if original_type not in STRICT_CLASSES:
        STRICT_CLASSES[original_type] = type(
            "NativeMM" + original_type.__name__,
            (original_type,),
            {"_fallback_to_text_only_and_retry": StrictImageOpenAI._fallback_to_text_only_and_retry},
        )
    clone = object.__new__(STRICT_CLASSES[original_type])
    clone.__dict__.update(provider.__dict__)
    clone.provider_config = copy.deepcopy(provider.provider_config)
    clone.api_keys = list(provider.api_keys)
    return clone


class RequestContext(Context):
    """SDK requires Context identity; normal attribute lookup preserves overrides."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def get_llm_tool_manager(self):
        return self.original.get_llm_tool_manager()

    def get_config(self, *args, **kwargs):
        return self.original.get_config(*args, **kwargs)

    def get_provider_by_id(self, provider_id):
        provider = self.original.get_provider_by_id(provider_id)
        return strict_provider(provider) if provider else None

    async def get_using_provider_async(self, *args, **kwargs):
        provider = await self.original.get_using_provider_async(*args, **kwargs)
        return strict_provider(provider) if provider else None


async def execute(event, context, request, provider, on_stats=None):
    """Yield at the same response boundary as the core local-agent stage."""
    scoped = RequestContext(context)
    config = scoped.get_config(umo=event.unified_msg_origin)
    if not config.get("provider_settings", {}).get("enable", True):
        return
    if not await SessionServiceManager.should_process_llm_request(event):
        return
    if config.get("agent_runner", {}).get("runner_type", "local") != "local":
        raise RuntimeError("NativeMM requires AstrBot local Agent")
    native = InternalAgentSubStage()
    await native.initialize(PipelineContext(config, SimpleNamespace(context=scoped), "native-mm"))
    if await call_event_hook(event, EventType.OnWaitingLLMRequestEvent):
        return
    event.set_extra("provider_request", request)
    collected, _ = await collect_initial_request(event, scoped, native.main_agent_cfg)
    cfg = replace(native.main_agent_cfg, streaming_response=False)
    built = await build_main_agent(
        event=event,
        plugin_context=scoped,
        config=cfg,
        provider=strict_provider(provider),
        req=collected,
        apply_reset=False,
    )
    if built is None:
        raise RuntimeError("agent_build_failed")
    if await call_event_hook(event, EventType.OnLLMRequestEvent, built.provider_request):
        if built.reset_coro:
            built.reset_coro.close()
        return
    await built.reset_coro
    runner = built.agent_runner
    register_active_runner(event.unified_msg_origin, runner)
    try:
        # NapCat group replies are buffered; preserve the core's /stop integration.
        async for result in run_agent(
            runner,
            native.max_step,
            native.show_tool_use,
            native.show_tool_call_result,
            show_reasoning=native.show_reasoning,
            buffer_intermediate_messages=native.buffer_intermediate_messages,
        ):
            yield result
        # A tool that sends directly / returns None ends the core loop without
        # on_agent_done. Collect its actual call/result, without inventing a reply.
        if runner.done() and not event.get_extra("_native_multimodal_done", False):
            await runner.agent_hooks.on_agent_done(
                runner.run_context,
                runner.get_final_llm_resp() or LLMResponse(role="tool", completion_text=""),
            )
        # The core stage normally records stats after consuming run_agent. Since
        # we own that boundary, preserve its usage/cache accounting explicitly.
        # This helper handles storage errors without affecting the response.
        if not event.is_stopped() or runner.was_aborted():
            await native._save_to_history(
                event,
                built.provider_request,
                runner.get_final_llm_resp(),
                runner.run_context.messages,
                runner.stats,
                user_aborted=runner.was_aborted(),
            )
    finally:
        try:
            # Usage exists even when the send pipeline is closed after yielding.
            final_response = runner.get_final_llm_resp()
            stats_runner = (
                runner
                if runner.done()
                else SimpleNamespace(provider=runner.provider, stats=runner.stats, was_aborted=lambda: True)
            )
            await _record_internal_agent_stats(event, built.provider_request, stats_runner, final_response)
            if on_stats:
                await on_stats(runner.stats, final_response, stats_runner.was_aborted())
        finally:
            unregister_active_runner(event.unified_msg_origin, runner)


class AstrBotGateway:
    """The single compatibility boundary for the tested AstrBot 4.28.2 local Agent.

    Keep the native builder for persona/KB/tools. Public tool_loop_agent lacks
    those pipeline responsibilities; replacement requires the contract probe.
    """

    def __init__(self, context, config):
        from astrbot.api import logger

        self.context, self.config, self.logger = context, config, logger

    @staticmethod
    def data_root():
        from astrbot.core.star.star_tools import StarTools

        return StarTools.get_data_dir("astrbot_plugin_group_chat_plus") / "multimodal_v1"

    @staticmethod
    def restore_wrappers():
        from astrbot.core.star.star_handler import star_handlers_registry

        from ..bridge import restore_legacy_wrappers

        return restore_legacy_wrappers(star_handlers_registry)

    async def provider(self, event):
        provider_id = self.config["provider_id"]
        if provider_id:
            event.set_extra("selected_provider", provider_id)
            return self.context.get_provider_by_id(provider_id)
        return await self.context.get_using_provider_async(event.unified_msg_origin)

    async def conversation(self, event):
        manager = self.context.conversation_manager
        cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
        if not cid:
            cid = await manager.new_conversation(
                event.unified_msg_origin, platform_id=event.get_platform_id()
            )
        return await manager.get_conversation(event.unified_msg_origin, cid)

    def validate_images(self, event, provider):
        from ..context import ContextLimit

        providers = [provider]
        cfg = self.context.get_config(umo=event.unified_msg_origin)
        fallback_ids = (
            cfg.get("agent_runner", {}).get("config", {}).get("model", {}).get("fallback_provider_ids", [])
        )
        providers.extend(self.context.get_provider_by_id(pid) for pid in fallback_ids)
        if any(p and "image" not in p.provider_config.get("modalities", []) for p in providers):
            raise ContextLimit("当前或备用模型未声明图片输入能力，请调整模型配置。")

    def prepare(self, event, text, conversation, reverse_poke=False):
        from astrbot.core.agent.message import TextPart

        request = event.request_llm(prompt=text, contexts=[], conversation=conversation)
        if reverse_poke:
            request.extra_user_content_parts.append(
                TextPart(text="[平台动作]机器人已向本轮戳人者戳回一次，此动作已成功执行。")
            )
        return request

    @staticmethod
    def track_transport(event, journal, gid, on_sent=None):
        return TransportTracker(event, journal, gid, on_sent)

    def execute(self, event, request, provider, on_stats=None):
        return execute(event, self.context, request, provider, on_stats)

    @staticmethod
    def already_handled(event):
        return event.is_stopped() or event._has_send_oper or event.call_llm

    @staticmethod
    def mirror(run_context, messages):
        from astrbot.core.agent.message import Message

        run_context.messages[:] = [Message.model_validate(msg) for msg in messages]

    @staticmethod
    def receipt(event):
        result = event.get_result()
        if not result or not result.is_model_result():
            return None
        # The SDK has no durable per-segment send receipt here. Retain object
        # references in TurnState so Python cannot reuse identities mid-turn.
        return result

    def scope_policy(self, event, provider):
        cfg = self.context.get_config(umo=event.unified_msg_origin)
        return {
            "provider": provider.provider_config.get("id"),
            "model": provider.get_model(),
            "fallbacks": cfg.get("agent_runner", {})
            .get("config", {})
            .get("model", {})
            .get("fallback_provider_ids", []),
            "role": getattr(event, "role", None),
        }

    async def summarize(self, event, provider, rules, data):
        import asyncio
        import json
        import time

        from astrbot.core.agent.response import AgentStats

        response = None
        stats = AgentStats()
        stats.start_time = time.time()
        try:
            response = await asyncio.wait_for(
                strict_provider(provider).text_chat(
                    prompt=json.dumps(data, ensure_ascii=False),
                    contexts=[],
                    system_prompt=rules,
                    max_tokens=min(2048, int(self.config["summary_token_budget"])),
                ),
                15,
            )
            if response.usage:
                stats.token_usage = response.usage
            return response
        finally:
            stats.end_time = time.time()
            await _record_internal_agent_stats(
                event,
                None,
                SimpleNamespace(provider=provider, stats=stats, was_aborted=lambda: False),
                response or LLMResponse(role="err", completion_text="summary_failed"),
            )

    async def gate(self, event, provider, history, current):
        import asyncio
        import time

        from astrbot.core.agent.response import AgentStats

        from ..bridge import GATE_RULES

        stats = AgentStats()
        started = time.time()
        response = None
        try:
            response = await asyncio.wait_for(
                strict_provider(provider).text_chat(
                    prompt="",
                    contexts=history + [{"role": "user", "content": current}],
                    system_prompt=GATE_RULES,
                ),
                15,
            )
            if response.usage:
                stats.token_usage = response.usage
            return response
        finally:
            stats.start_time = started
            stats.end_time = time.time()
            fake_runner = SimpleNamespace(provider=provider, stats=stats, was_aborted=lambda: False)
            await _record_internal_agent_stats(
                event, None, fake_runner, response or LLMResponse(role="err", completion_text="gate_failed")
            )


class EventBot:
    """Per-event delegate captures OneBot ACKs that SDK send() discards."""

    def __init__(self, original, tracker):
        self.original, self.tracker = original, tracker

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def send_group_msg(self, **kwargs):
        return await self.tracker.platform_send(self.original.send_group_msg, kwargs)

    async def call_action(self, action, **kwargs):
        if action == "send_group_forward_msg":

            async def original(**params):
                return await self.original.call_action(action, **params)

            return await self.tracker.platform_send(original, kwargs, forward=True)
        return await self.original.call_action(action, **kwargs)

    async def send(self, event, message, **kwargs):
        async def original(**params):
            return await self.original.send(event=event, message=message, **kwargs)

        return await self.tracker.platform_send(
            original, {"group_id": event.get("group_id"), "message": message}
        )


class TransportTracker:
    """Observe this event's real sends; core after-send hooks also fire on failure.

    No global platform/client wrapping. Each actual send gets a turn-local key;
    callbacks never manufacture transport success. Exceptions keep their normal
    framework handling while the journal records incomplete delivery.
    """

    def __init__(self, event, journal, gid, on_sent=None):
        self.event, self.journal, self.gid = event, journal, gid
        self.on_sent = on_sent
        self.original = event.send
        self.previous_override = event.__dict__.get("send")
        self.had_override = "send" in event.__dict__
        self.attempts = 0
        self.successes = 0
        self.wrapper = self.send
        event.send = self.wrapper
        self.bot_original = None
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent

        if isinstance(event, AiocqhttpMessageEvent):
            self.bot_original = event.bot
            self.bot_proxy = EventBot(event.bot, self)
            event.bot = self.bot_proxy
        self.platform_attempts = 0

    async def _ack(self, receipt, result, chain):
        import asyncio

        from astrbot.api import logger

        platform_id = (
            str(result["message_id"])
            if isinstance(result, dict) and result.get("message_id") is not None
            else None
        )
        self.successes += 1

        async def persist():
            try:
                await self.journal.sent(self.gid, receipt, platform_id)
                if self.on_sent:
                    await self.on_sent(chain, receipt, platform_id)
            except Exception:
                logger.exception("[NativeMM] 已发送消息保存失败")

        task = asyncio.create_task(persist())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def platform_send(self, original, kwargs, forward=False):
        from astrbot.core.message.message_event_result import MessageChain

        if str(kwargs.get("group_id")) != str(self.event.get_group_id()):
            return await original(**kwargs)
        self.platform_attempts += 1
        receipt = self.gid + ":onebot:" + str(self.platform_attempts)
        await self.journal.delivery_attempt(self.gid, receipt)
        try:
            result = await original(**kwargs)
        except BaseException:
            await self.journal.delivery_failed(self.gid, receipt)
            raise
        parts = kwargs.get("messages", kwargs.get("nodes", [])) if forward else kwargs.get("message", [])
        if forward:
            parts = [{"type": "nodes", "nodes": parts}]
        await self._ack(
            receipt,
            result,
            MessageChain(parts if isinstance(parts, list) else [{"type": "text", "text": str(parts)}]),
        )
        return result

    async def send(self, chain):
        self.attempts += 1
        receipt = self.gid + ":transport:" + str(self.attempts)
        if not self.bot_original:
            await self.journal.delivery_attempt(self.gid, receipt)
        before = self.platform_attempts
        try:
            result = await self.original(chain)
        except BaseException:
            if not self.bot_original or self.platform_attempts == before:
                await self.journal.delivery_attempt(self.gid, receipt)
                await self.journal.delivery_failed(self.gid, receipt)
            raise
        else:
            if not self.bot_original:
                await self._ack(receipt, result, chain)
            return result

    def restore(self):
        if self.bot_original is not None and self.event.bot is self.bot_proxy:
            self.event.bot = self.bot_original
        if self.event.send is self.wrapper:
            if self.had_override:
                self.event.send = self.previous_override
            else:
                del self.event.send
