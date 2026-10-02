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
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage, _record_internal_agent_stats,
)
from astrbot.core.pipeline.process_stage.follow_up import register_active_runner, unregister_active_runner
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.star.star_handler import EventType
from astrbot.core.star.session_llm_manager import SessionServiceManager


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
        STRICT_CLASSES[original_type] = type("NativeMM" + original_type.__name__, (original_type,),
                                             {"_fallback_to_text_only_and_retry": StrictImageOpenAI._fallback_to_text_only_and_retry})
    clone = object.__new__(STRICT_CLASSES[original_type])
    clone.__dict__.update(provider.__dict__)
    clone.provider_config = copy.deepcopy(provider.provider_config)
    clone.api_keys = list(provider.api_keys)
    return clone


class RequestContext(Context):
    def __init__(self, original):
        self.original = original

    def __getattribute__(self, name):
        if name.startswith("__") or name in {"original", "get_config", "get_provider_by_id", "get_using_provider_async"}:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "original"), name)

    def get_config(self, *args, **kwargs):
        return self.original.get_config(*args, **kwargs)

    def get_provider_by_id(self, provider_id):
        provider = self.original.get_provider_by_id(provider_id)
        return strict_provider(provider) if provider else None

    async def get_using_provider_async(self, *args, **kwargs):
        provider = await self.original.get_using_provider_async(*args, **kwargs)
        return strict_provider(provider) if provider else None


async def execute(event, context, request, provider):
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
    built = await build_main_agent(event=event, plugin_context=scoped, config=cfg,
                                   provider=strict_provider(provider), req=collected, apply_reset=False)
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
        async for result in run_agent(runner, native.max_step, native.show_tool_use, native.show_tool_call_result,
                                      show_reasoning=native.show_reasoning,
                                      buffer_intermediate_messages=native.buffer_intermediate_messages):
            yield result
        # A tool that sends directly / returns None ends the core loop without
        # on_agent_done. Collect its actual call/result, without inventing a reply.
        if runner.done() and not event.get_extra("_native_multimodal_done", False):
            await runner.agent_hooks.on_agent_done(runner.run_context,
                                                   runner.get_final_llm_resp() or LLMResponse(role="tool", completion_text=""))
        # The core stage normally records stats after consuming run_agent. Since
        # we own that boundary, preserve its usage/cache accounting explicitly.
        # This helper handles storage errors without affecting the response.
        await _record_internal_agent_stats(event, built.provider_request, runner,
                                           runner.get_final_llm_resp())
        if not event.is_stopped() or runner.was_aborted():
            await native._save_to_history(event, built.provider_request, runner.get_final_llm_resp(),
                                          runner.run_context.messages, runner.stats, user_aborted=runner.was_aborted())
    finally:
        unregister_active_runner(event.unified_msg_origin, runner)
