"""Run inside AstrBot v4.28.2: real builder/runner/hooks, offline provider.

Usage: python tests/multimodal/sdk_probe.py /path/to/plugin
Does not load production configuration, contact providers or send platform messages.
"""

import asyncio
import base64
import copy
import importlib
import json
import pathlib
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiocqhttp import Event as OneBotEvent
from astrbot.api.star import Context
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.config.agent_runner import normalize_agent_runner
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.message.components import At, Image, Plain, Poke
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import AiocqhttpAdapter
from astrbot.core.provider.entities import LLMResponse, TokenUsage
from astrbot.core.provider.provider import Provider
from astrbot.core.star.star import StarMetadata, star_map
from astrbot.core.star.star_handler import EventType, star_handlers_registry
from mcp.types import CallToolResult, ImageContent, TextContent

root = pathlib.Path(sys.argv[1]).resolve()
package = types.ModuleType("native_mm_probe")
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
module = importlib.import_module("native_mm_probe.main")
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)


def event(eid, parts, sender="a", group="fixture-group"):
    msg = AstrBotMessage()
    msg.type = MessageType.GROUP_MESSAGE
    msg.self_id, msg.group_id, msg.message_id = "fixture-bot", group, eid
    msg.sender = MessageMember(user_id=sender, nickname=sender)
    msg.message = parts
    msg.message_str = "".join(p.text for p in parts if isinstance(p, Plain))
    meta = SimpleNamespace(
        id="fixture-platform",
        name="aiocqhttp",
        support_proactive_message=False,
        support_streaming_message=False,
    )
    ev = AstrMessageEvent(msg.message_str, msg, meta, sender + "_" + group)
    ev.send = AsyncMock()
    ev.is_at_or_wake_command = any(isinstance(p, At) for p in parts)
    return ev


async def record(plugin, ev):
    assert [result async for result in plugin.capture(ev)] == []


async def main():
    captures = []
    tool_mode = {"enabled": False, "step": 0}
    provider = Mock(spec=Provider)
    provider.provider_config = {
        "id": "fixture-provider",
        "modalities": ["text", "image", "tool_use"],
        "max_context_tokens": 128000,
    }
    provider.get_model.return_value = "fixture-model"
    provider.meta.return_value = SimpleNamespace(type="openai_chat_completion", id="fixture-provider")

    async def reply(**kwargs):
        captures.append(kwargs)
        if "判断是否值得主动参与" in kwargs.get("system_prompt", ""):
            return LLMResponse(
                role="assistant",
                completion_text='{"reply":true}',
                usage=TokenUsage(input_other=32, input_cached=64, output=4),
            )
        if tool_mode["enabled"] and tool_mode["step"] == 0:
            tool_mode["step"] += 1
            return LLMResponse(
                role="assistant",
                completion_text="",
                tools_call_name=["fixture_lookup"],
                tools_call_args=[{}],
                tools_call_ids=["call-fixture"],
                usage=TokenUsage(input_other=64, input_cached=128, output=8),
            )
        return LLMResponse(
            role="assistant",
            completion_text="fixture_reply",
            usage=TokenUsage(input_other=64, input_cached=128, output=8),
        )

    provider.text_chat = AsyncMock(side_effect=reply)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["agent_runner"] = normalize_agent_runner(cfg.get("agent_runner"))
    cfg["provider_ltm_settings"]["group_message_history_enable"] = False
    cfg["provider_settings"]["proactive_capability"] = {"add_cron_tools": False}
    cfg["subagent_orchestrator"] = {}
    cfg["timezone"] = "UTC"
    cfg["provider_settings"]["enable"] = True
    context = Mock(spec=Context)
    context.get_config.return_value = cfg
    context.get_using_provider_async = AsyncMock(return_value=provider)
    context.get_provider_by_id.return_value = provider
    context.subagent_orchestrator = None
    context.kb_manager = SimpleNamespace()
    context.persona_manager = SimpleNamespace(
        resolve_selected_persona=AsyncMock(
            return_value=(
                "fixture-persona",
                {"prompt": "PERSONA_MUST_SURVIVE", "tools": [], "skills": []},
                None,
                False,
            )
        )
    )
    context.get_llm_tool_manager.return_value = SimpleNamespace(get_full_tool_set=lambda: ToolSet())
    conversation = SimpleNamespace(
        history=json.dumps([{"role": "user", "content": "OLD_CORE_HISTORY"}]),
        cid="fixture-conversation",
        persona_id="fixture-persona",
        token_usage=0,
    )
    context.conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value=conversation.cid),
        get_conversation=AsyncMock(return_value=conversation),
        update_conversation=AsyncMock(),
    )
    results = {}
    with (
        tempfile.TemporaryDirectory() as td,
        patch("astrbot.core.star.star_tools.StarTools.get_data_dir", return_value=pathlib.Path(td)),
        patch(
            "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal.db_helper.insert_provider_stat",
            new=AsyncMock(),
        ) as stat_sink,
    ):
        plugin = module.ChatPlus(
            context,
            {"input_token_budget": 32768, "mention_wait_seconds": 1.0, "enable_poke_after_reply": False},
        )
        star_map[module.__name__] = StarMetadata(name="native-mm-probe", activated=True)
        for handler in star_handlers_registry:
            if handler.handler_module_path == module.__name__:
                handler.handler = getattr(plugin, handler.handler.__name__)
        await plugin.initialize()
        try:
            source = event("image", [Image.fromBytes(PNG)])
            await record(plugin, source)
            initial = source.get_extra(module.OWNER + "_anchor")
            interrupt = event("interruption", [Plain("B 插话")], sender="b")
            await record(plugin, interrupt)
            target = event("question", [At(qq="fixture-bot"), Plain("看前一张图")])
            await record(plugin, target)
            flow = plugin.respond(target)
            with patch(
                "astrbot.core.astr_main_agent.retrieve_knowledge_base",
                new=AsyncMock(return_value="KB_MUST_SURVIVE"),
            ) as kb:
                await anext(flow)
                assert kb.call_args.kwargs["query"] == "看前一张图"
            state = target.get_extra(module.OWNER)
            assert "OLD_CORE_HISTORY" in json.dumps(state.snapshot["contexts"])
            actual = state.request
            assert actual.prompt == "" and "PERSONA_MUST_SURVIVE" in actual.system_prompt
            assert "KB_MUST_SURVIVE" in json.dumps(actual.extra_user_content_parts)
            assert "OLD_CORE_HISTORY" not in json.dumps(actual.contexts)
            assert len(captures) == 1
            body = json.dumps(captures[0], default=str, ensure_ascii=False)
            assert body.count("data:image/png;base64,") == 1
            assert "PERSONA_MUST_SURVIVE" in body and "KB_MUST_SURVIVE" in body
            assert '"sender_id": "a"' in body or '\\"sender_id\\": \\"a\\"' in body
            assert state.transport.original.await_count == 0
            # on_agent_done used real SDK Message objects and generated our journal turn.
            assert (await plugin.service.journal.status(initial["room"]))["last_generation"][
                "status"
            ] == "generated"
            target.set_result(
                target.plain_result("fixture_reply").set_result_content_type(ResultContentType.LLM_RESULT)
            )
            await target.send(target.get_result())
            await call_event_hook(target, EventType.OnAfterMessageSentEvent)
            try:
                await anext(flow)
            except StopAsyncIteration:
                pass
            assert "data:image" not in json.dumps(
                context.conversation_manager.update_conversation.call_args.kwargs["history"]
            )
            assert (await plugin.service.journal.status(initial["room"]))["last_generation"][
                "status"
            ] == "sent"
            results["actual_builder_history_reload_and_final_image_count"] = 1
            results["persona_and_kb_preserved"] = True
            results["agent_completion_persisted_and_mirror_has_no_base64"] = True

            # Per-room lock survives the yield boundary until the framework finishes.
            one = event("one", [At(qq="fixture-bot"), Plain("one")])
            two = event("two", [At(qq="fixture-bot"), Plain("two")], sender="b")
            await record(plugin, one)
            await record(plugin, two)
            first = plugin.respond(one)
            second = plugin.respond(two)
            await anext(first)
            pending = asyncio.create_task(anext(second))
            await asyncio.sleep(0.02)
            assert not pending.done()
            await first.aclose()
            await asyncio.wait_for(pending, 20)
            prior_gid = one.get_extra(module.OWNER).gid
            assert any(
                e["event_id"] == "generation:" + prior_gid
                for e in two.get_extra(module.OWNER).selection.events
            )
            await second.aclose()
            results["room_lock_serializes_different_senders"] = True
            results["queued_turn_includes_completed_previous_answer"] = True

            # A pure @ may collect A's next line, but never takes B's line as A's.
            mention = event("mention", [At(qq="fixture-bot")])
            pure = plugin.capture(mention)
            pending = asyncio.create_task(anext(pure))
            await asyncio.sleep(0.01)
            other = event("other", [Plain("B unrelated")], sender="b")
            other_task = asyncio.create_task(record(plugin, other))
            own = event("own-followup", [Plain("A followup")])
            own_task = asyncio.create_task(record(plugin, own))
            await asyncio.gather(other_task, own_task)
            await asyncio.wait_for(pending, 20)
            state = mention.get_extra(module.OWNER)
            assert state.selection.anchor["event_id"] == "own-followup"
            assert state.selection.anchor["sender"] == "a"
            await pure.aclose()
            results["pure_mention_keeps_sender_identity"] = True

            # Exercise a real SDK tool loop and the plugin's completion collector.
            async def lookup(event: AstrMessageEvent):
                return CallToolResult(
                    content=[
                        TextContent(type="text", text="fixture_tool_result"),
                        ImageContent(type="image", data=base64.b64encode(PNG).decode(), mimeType="image/png"),
                    ]
                )

            tool = FunctionTool(
                name="fixture_lookup",
                description="fixture only",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
            )
            context.persona_manager.resolve_selected_persona.return_value = (
                "fixture-persona",
                {"prompt": "PERSONA_MUST_SURVIVE", "tools": None, "skills": []},
                None,
                False,
            )
            toolset = ToolSet()
            toolset.add_tool(tool)
            context.get_llm_tool_manager.return_value = SimpleNamespace(get_full_tool_set=lambda: toolset)
            tool_mode["enabled"] = True
            query = event("tool-probe", [At(qq="fixture-bot"), Plain("tool-probe")])
            await record(plugin, query)
            with patch(
                "astrbot.core.astr_main_agent.retrieve_knowledge_base", new=AsyncMock(return_value=None)
            ):
                async for _ in plugin.respond(query):
                    pass
            protocol = (
                await plugin.service.journal.get(
                    initial["room"], "generation:" + query.get_extra(module.OWNER).gid
                )
            )["parts"][0]["messages"]
            assert [p["role"] for p in protocol] == ["assistant", "tool", "user", "assistant"]
            assert protocol[0]["tool_calls"][0]["id"] == protocol[1]["tool_call_id"] == "call-fixture"
            assert "fixture_tool_result" in json.dumps(protocol)
            assert "journal_image" in json.dumps(protocol)
            assert "data:image" not in json.dumps(protocol)
            results["real_tool_loop_ids_and_complete_reply_saved"] = True

            # SDK also supports terminal tools with no final assistant response.
            async def terminal(event: AstrMessageEvent):
                return None

            tool.handler = terminal
            tool_mode["step"] = 0
            query = event("tool-terminal", [At(qq="fixture-bot"), Plain("tool-terminal")])
            await record(plugin, query)
            with patch(
                "astrbot.core.astr_main_agent.retrieve_knowledge_base", new=AsyncMock(return_value=None)
            ):
                async for _ in plugin.respond(query):
                    pass
            protocol = (
                await plugin.service.journal.get(
                    initial["room"], "generation:" + query.get_extra(module.OWNER).gid
                )
            )["parts"][0]["messages"]
            assert [p["role"] for p in protocol] == ["assistant", "tool"]
            assert protocol[0]["tool_calls"][0]["id"] == protocol[1]["tool_call_id"]
            results["terminal_tool_result_persisted_without_fake_reply"] = True

            # Per-request strict adapter refuses the core's image-removal retry.
            runner_module = importlib.import_module("native_mm_probe.multimodal.adapters.astrbot")
            raw = object.__new__(runner_module.ProviderOpenAIOfficial)
            raw.provider_config = {"id": "fixture"}
            raw.api_keys = ["fixture"]
            raw.client = SimpleNamespace()
            strict = runner_module.strict_provider(raw)
            assert strict is not raw and strict.client is raw.client
            try:
                await strict._fallback_to_text_only_and_retry()
            except ValueError:
                pass
            else:
                raise AssertionError("image removal allowed")
            results["image_removal_retry_blocked_without_global_mutation"] = True
            # Real core stats helper, mocked storage: no probe rows in production.
            assert stat_sink.await_count == 6
            recorded = [call.kwargs for call in stat_sink.await_args_list]
            assert all(row["stats"]["token_usage"]["input_cached"] == 128 for row in recorded[1:4])
            recorded = [recorded[0], recorded[4], recorded[5]]
            assert all(
                row["provider_id"] == "fixture-provider" and row["status"] == "completed" for row in recorded
            )
            assert [row["stats"]["token_usage"] for row in recorded] == [
                {"input_other": 64, "input_cached": 128, "output": 8},
                {"input_other": 128, "input_cached": 256, "output": 16},
                {"input_other": 64, "input_cached": 128, "output": 8},
            ]
            results["core_cache_stats_preserved_including_tool_rounds"] = True

            # Gate and reply both reach core statistics and local phase traces.
            tool_mode["enabled"] = False
            plugin.config.update(auto_reply_enabled=True, auto_candidate_probability=1, auto_reply_cooldown=0)
            auto_event = event("active-probe", [Plain("a topic")], group="active-fixture-group")
            auto_event.is_at_or_wake_command = False
            await record(plugin, auto_event)
            before_stats = stat_sink.await_count
            async for _ in plugin.respond(auto_event):
                pass
            assert stat_sink.await_count == before_stats + 2
            auto_status = await plugin.service.journal.status(
                auto_event.get_extra(module.OWNER + "_anchor")["room"]
            )
            assert {t["phase"]: t["cached_tokens"] for t in auto_status["trace"]} == {
                "gate": 64,
                "reply": 128,
            }
            results["proactive_gate_and_reply_both_accounted"] = True

            # The real RespondStage swallows per-segment errors and still calls
            # after_message_sent. Only observed successful sends count as sent.
            from astrbot.core.pipeline.context import PipelineContext
            from astrbot.core.pipeline.respond.stage import RespondStage
            from native_mm_probe.multimodal.models import TurnState

            for failures, expected in (
                ([None, RuntimeError("fixture segment failure")], "partial"),
                ([RuntimeError("fixture failure 1"), RuntimeError("fixture failure 2")], "uncertain"),
            ):
                delivery = event("segmented-" + expected, [Plain("question")], group="delivery-" + expected)
                await record(plugin, delivery)
                anchor = delivery.get_extra(module.OWNER + "_anchor")
                selection = await plugin.service.selector.candidates(anchor)
                gid = await plugin.service.journal.begin(anchor["room"], anchor["seq"])
                await plugin.service.journal.finish(
                    gid, "generated", [{"role": "assistant", "content": "part1 part2"}], "bot"
                )
                delivery.send = AsyncMock(side_effect=failures)
                state = TurnState(selection, gid, provider)
                state.transport = plugin.service.gateway.track_transport(
                    delivery, plugin.service.journal, gid
                )
                delivery.set_extra(module.OWNER, state)
                result = delivery.plain_result("part1").set_result_content_type(ResultContentType.LLM_RESULT)
                result.chain.append(Plain("part2"))
                delivery.set_result(result)
                segmented_cfg = copy.deepcopy(cfg)
                segmented_cfg["platform_settings"]["segmented_reply"]["enable"] = True
                segmented_cfg["platform_settings"]["segmented_reply"]["interval"] = "0,0"
                responder = RespondStage()
                await responder.initialize(
                    PipelineContext(segmented_cfg, SimpleNamespace(context=context), "probe")
                )
                responder._calc_comp_interval = AsyncMock(return_value=0)
                try:
                    await responder.process(delivery)
                    await plugin.service.journal.settle(gid, completed=True)
                    status = (await plugin.service.journal.status(anchor["room"]))["last_generation"]
                    assert status["delivery"] == expected
                    assert state.transport.attempts == 2
                    assert status["sent_parts"] == (1 if expected == "partial" else 0)
                finally:
                    state.transport.restore()
            results["real_segmented_send_failures_are_not_success"] = True

            # Real notice shape: no text and no @; bypass the ordinary auto gate.
            tool_mode["enabled"] = False
            plugin.config.update(
                auto_reply_enabled=False,
                auto_candidate_probability=0,
                enable_poke_after_reply=True,
                poke_after_reply_probability=1,
                poke_after_reply_delay=0,
            )
            notice_raw = dict(
                post_type="notice",
                notice_type="notify",
                sub_type="poke",
                self_id=67890,
                user_id=12345,
                target_id=67890,
                group_id=24680,
            )
            converter = object.__new__(AiocqhttpAdapter)
            converted = await converter._convert_handle_notice_event(OneBotEvent(notice_raw))
            assert converted.message_str == "" and isinstance(converted.message[0], Poke)
            poke_event = event("poke-notice", [], sender="12345", group="24680")
            poke_event.message_obj = converted
            poke_event.message_str = poke_event.message_obj.message_str = ""
            poke_event.is_at_or_wake_command = False
            poke_event.bot = SimpleNamespace(api=SimpleNamespace(call_action=AsyncMock()))
            original_count = len(captures)
            flow = plugin.capture(poke_event)
            await anext(flow)
            assert len(captures) == original_count + 1
            assert "戳一戳事件" in json.dumps(captures[-1], default=str, ensure_ascii=False)
            assert "unsupported_modality" not in json.dumps(captures[-1], default=str)
            poke_event.set_result(
                poke_event.plain_result("fixture_reply").set_result_content_type(ResultContentType.LLM_RESULT)
            )
            await poke_event.send(poke_event.get_result())
            await call_event_hook(poke_event, EventType.OnAfterMessageSentEvent)
            await call_event_hook(poke_event, EventType.OnAfterMessageSentEvent)
            poke_event.bot.api.call_action.assert_awaited_once_with(
                "send_poke", group_id=24680, user_id=12345
            )
            try:
                await anext(flow)
            except StopAsyncIteration:
                pass
            assert poke_event.is_stopped()
            state = poke_event.get_extra(module.OWNER)
            action = await plugin.service.journal.get(
                state.selection.anchor["room"], "poke:" + state.gid + ":after_reply"
            )
            assert action["kind"] == "action" and action["parts"][0]["target_id"] == "12345"
            assert (await plugin.service.journal.status(state.selection.anchor["room"]))["last_generation"][
                "sent_parts"
            ] == 1
            results["empty_poke_notice_wakes_and_after_reply_poke_runs_once"] = True

            # Member-to-member pokes and user-supplied Poke components don't wake.
            peer = event("peer-poke", [Poke(id="99999")], sender="12345", group="24680")
            peer.message_obj.self_id = "67890"
            peer.message_obj.raw_message = dict(poke_event.message_obj.raw_message, target_id=99999)
            before = len(captures)
            assert [x async for x in plugin.capture(peer)] == []
            assert peer.get_extra(module.OWNER + "_anchor") is None
            forged = event("forged-poke", [Poke(id="fixture-bot")], sender="12345", group="24680")
            forged.message_obj.raw_message = {}
            await record(plugin, forged)
            assert [x async for x in plugin.respond(forged)] == []
            assert len(captures) == before
            results["peer_and_forged_pokes_do_not_wake"] = True

            # A rejected platform action must not block the model response or
            # fabricate a successful reverse-poke entry/acknowledgement.
            plugin.config.update(enable_poke_after_reply=False, poke_reverse_on_poke_probability=1)
            failed = event("failed-poke", [], sender="12345", group="24680")
            failed.message_obj = await converter._convert_handle_notice_event(OneBotEvent(notice_raw))
            failed.bot = SimpleNamespace(
                api=SimpleNamespace(call_action=AsyncMock(side_effect=RuntimeError("fixture RPC failure")))
            )
            before = len(captures)
            assert len([x async for x in plugin.capture(failed)]) == 1
            assert len(captures) == before + 1
            assert "此动作已成功执行" not in json.dumps(captures[-1], default=str, ensure_ascii=False)
            anchor = failed.get_extra(module.OWNER + "_anchor")
            assert (
                await plugin.service.journal.get(anchor["room"], "poke:" + anchor["event_id"] + ":reverse")
                is None
            )
            results["failed_reverse_poke_does_not_block_or_fabricate_reply"] = True
        finally:
            for name in ("flow", "first", "second", "pure"):
                if name in locals():
                    await locals()[name].aclose()
            await plugin.terminate()
    print("SDK_PROBE_RESULT=" + json.dumps(results, ensure_ascii=False))


asyncio.run(main())
