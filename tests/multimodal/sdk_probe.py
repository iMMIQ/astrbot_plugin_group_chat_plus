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
from astrbot.core.message.components import At, Image, Plain, Poke, Reply
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
        if "将群聊数据压缩为 JSON" in kwargs.get("system_prompt", ""):
            data = json.loads(kwargs["prompt"])
            first = data["messages"][0]
            return LLMResponse(
                role="assistant",
                completion_text=json.dumps(
                    {
                        "items": [
                            {"text": first["sender"] + ": 讨论旅行计划，路线待定", "sources": [first["seq"]]}
                        ]
                    }
                ),
                usage=TokenUsage(input_other=48, input_cached=16, output=24),
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
            assert "[native_bot_identity=" in actual.system_prompt
            assert actual.extra_user_content_parts[-1]["text"].startswith("[native_turn_control]")
            assert state.route.reason == "mention_self"
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
            assert "native_turn_control" not in json.dumps(
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
            await one.send(one.plain_result("first-visible-answer"))
            await first.aclose()
            await asyncio.wait_for(pending, 20)
            assert any(
                e.get("causal_anchor") == one.get_extra(module.OWNER).selection.anchor["seq"]
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
            assert state.route.reason == "mention_followup" and state.route.source_event_id == "mention"
            control = json.loads(state.request.extra_user_content_parts[-1]["text"].split("]", 1)[1])
            assert control["source_event_id"] == "mention" and control["anchor_event_id"] == "own-followup"
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
            context.get_llm_tool_manager.return_value = SimpleNamespace(
                get_full_tool_set=lambda: ToolSet(list(toolset.tools)), get_builtin_tool=lambda cls: cls()
            )
            tool_mode["enabled"] = True
            query = event("tool-probe", [At(qq="fixture-bot"), Plain("tool-probe")])
            await record(plugin, query)
            with patch(
                "astrbot.core.astr_main_agent.retrieve_knowledge_base", new=AsyncMock(return_value=None)
            ):
                async for _ in plugin.respond(query):
                    pass
            protocol = await plugin.service.journal.execution(
                query.get_extra(module.OWNER).gid, query.unified_msg_origin
            )
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
            protocol = await plugin.service.journal.execution(
                query.get_extra(module.OWNER).gid, query.unified_msg_origin
            )
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
            assert auto_event.get_extra(module.OWNER).route.reason == "auto_approved"
            assert "[native_bot_identity=" in captures[-2]["system_prompt"]
            assert "participation" in json.dumps(captures[-1], default=str)

            # Explicit platform facts win over quoted peers and a rejecting gate.
            plugin.config.update(auto_reply_enabled=False)
            quoted = event(
                "quoted-peer-and-at",
                [Reply(id="interruption"), At(qq="fixture-bot"), Plain("三香包是什么馅")],
            )
            quoted.is_at_or_wake_command = False
            await record(plugin, quoted)
            with patch.object(plugin.service.gateway, "gate", new=AsyncMock()) as gate:
                async for _ in plugin.respond(quoted):
                    pass
                gate.assert_not_awaited()
            assert quoted.get_extra(module.OWNER).route.reason == "mention_self"
            seed = event("reply-seed", [Plain("背景")], group="reply-fixture")
            await record(plugin, seed)
            room = seed.get_extra(module.OWNER + "_anchor")["room"]
            await plugin.service.journal.add(
                room, "bot-response", "fixture-bot", "bot", [{"type": "text", "text": "answer"}]
            )
            follow = event(
                "reply-bot-confirmed", [Reply(id="bot-response"), Plain("继续说")], group="reply-fixture"
            )
            await record(plugin, follow)
            assert follow.get_extra(module.OWNER + "_route").reason == "reply_self"
            with patch.object(plugin.service.gateway, "gate", new=AsyncMock()) as gate:
                async for _ in plugin.respond(follow):
                    pass
                gate.assert_not_awaited()
            results["explicit_at_and_reply_self_bypass_gate"] = True

            # Mirror the actual SDK's permissions before building instructions;
            # retain unrelated tools and never mutate the shared tool manager.
            from astrbot.core.tools.computer_tools import ShellSessionTool

            cfg["provider_settings"].update(computer_use_runtime="local", computer_use_require_admin=True)
            shell = ShellSessionTool()
            toolset.add_tool(shell)
            for eid, admin, required, allowed in [
                ("ordinary-tools", False, True, False),
                ("admin-tools", True, True, True),
                ("public-local-tools", False, False, True),
            ]:
                cfg["provider_settings"]["computer_use_require_admin"] = required
                query = event(
                    eid, [At(qq="fixture-bot"), Plain("三香包是什么馅")], group="permission-fixture"
                )
                query.role = "admin" if admin else "member"
                await record(plugin, query)
                with patch(
                    "astrbot.core.astr_main_agent.retrieve_knowledge_base", new=AsyncMock(return_value=None)
                ):
                    async for _ in plugin.respond(query):
                        pass
                req = query.get_extra(module.OWNER).request
                names = {t.name for t in req.func_tool.tools}
                assert ("astrbot_shell_session" in names) is allowed
                assert ("astrbot_shell_session" in req.system_prompt) is allowed
                assert "fixture_lookup" in names and shell in toolset.tools
                assert cfg["provider_settings"]["computer_use_runtime"] == "local"
            cfg["provider_settings"].update(computer_use_runtime="none", computer_use_require_admin=True)
            ordinary = event("late-tool-filter", [Plain("question")])
            req = SimpleNamespace(func_tool=toolset)
            plugin.service.gateway.filter_tools(ordinary, req)
            assert req.func_tool is not toolset and shell in toolset.tools
            assert {t.name for t in req.func_tool.tools} == {"fixture_lookup"}
            toolset.remove_tool("astrbot_shell_session")
            results["actual_computer_permissions_filter_tools_and_prompt_without_global_mutation"] = True

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

                async def ack(chain, receipt, platform_id):
                    await plugin.service.public_sent(delivery, state, chain, receipt, platform_id)

                state.transport = plugin.service.gateway.track_transport(
                    delivery, plugin.service.journal, gid, ack
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
                    view, _ = await plugin.service.journal.view(anchor, 0, 100)
                    public = [e for e in view if e["kind"] == "self"]
                    # Current generation follows its anchor; inspect full room snapshot.
                    public = [
                        e
                        for e in await plugin.service.journal.recent(anchor["room"], 2**63 - 1, 0, 100)
                        if e["kind"] == "self"
                    ]
                    assert len(public) == (1 if expected == "partial" else 0)
                    if public:
                        assert public[0]["parts"] == [{"type": "text", "text": "part1"}]
                finally:
                    state.transport.restore()
            results["real_segmented_send_failures_are_not_success"] = True

            # Real aiocqhttp send() returns None, but its underlying API ACK
            # carries the ID needed by a later quote. Delegate only this event.
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent

            source = event("real-onebot-source", [Image.fromBytes(PNG)], sender="12345", group="34567")
            source.message_obj.self_id = "67890"
            await record(plugin, source)
            original = event("real-onebot-question", [Plain("看这张图")], sender="12345", group="34567")
            original.message_obj.self_id = "67890"
            source.message_obj.self_id = "67890"
            bot = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": 998877}))
            real = AiocqhttpMessageEvent(
                original.message_str, original.message_obj, original.platform_meta, original.session_id, bot
            )
            await record(plugin, real)
            anchor = real.get_extra(module.OWNER + "_anchor")
            photo = source.get_extra(module.OWNER + "_anchor")
            await plugin.service.journal.link(anchor["room"], anchor["seq"], [photo["seq"]])
            gid = await plugin.service.journal.begin(anchor["room"], anchor["seq"], real.unified_msg_origin)
            state = TurnState(await plugin.service.selector.candidates(anchor), gid, provider)

            async def actual_ack(chain, receipt, platform_id):
                await plugin.service.public_sent(real, state, chain, receipt, platform_id)

            tracker = plugin.service.gateway.track_transport(real, plugin.service.journal, gid, actual_ack)
            try:
                assert real.bot is not bot
                assert await real.send(real.plain_result("actual-platform-reply")) is None
                stored = await plugin.service.journal.get(anchor["room"], "998877")
                assert stored["parts"] == [{"type": "text", "text": "actual-platform-reply"}]
                assert stored["causal_anchor"] == anchor["seq"]
                assert bot.send_group_msg.await_count == 1
                assert tracker.successes == 1
            finally:
                tracker.restore()
            assert real.bot is bot
            late = (
                await plugin.service.journal.add(
                    anchor["room"],
                    "late-real-quote",
                    "23456",
                    "B",
                    [{"type": "reply", "event_id": "998877"}, {"type": "text", "text": "这张图呢"}],
                    received=anchor["received"] + 901,
                )
            )[0]
            history, current = await plugin.service.selector.assemble(
                await plugin.service.selector.candidates(late)
            )
            assert json.dumps([history, current]).count("data:image/png;base64,") == 1
            results["real_onebot_discarded_ack_id_preserved_without_shared_mutation"] = True

            # Compression goes through a real provider contract and core stats,
            # only at segment rollover; it is not an extra per-message rewrite.
            tool_mode["enabled"] = False
            plugin.config.update(input_token_budget=5000, summary_token_budget=400)
            summary_before = sum("将群聊数据压缩为 JSON" in c.get("system_prompt", "") for c in captures)
            for index in range(8):
                query = event(
                    "summary-" + str(index),
                    [At(qq="fixture-bot"), Plain("旅行计划" * 60)],
                    group="summary-room",
                )
                await record(plugin, query)
                with patch(
                    "astrbot.core.astr_main_agent.retrieve_knowledge_base", new=AsyncMock(return_value=None)
                ):
                    flow = plugin.respond(query)
                    async for _ in flow:
                        await query.send(query.plain_result("visible-summary-room-reply"))
            summary_after = sum("将群聊数据压缩为 JSON" in c.get("system_prompt", "") for c in captures)
            assert 0 < summary_after - summary_before < 8
            report = await plugin.service.journal.status(
                query.get_extra(module.OWNER).selection.anchor["room"]
            )
            phase = next(p for p in report["trace"] if p["phase"] == "summary")
            assert phase["calls"] == summary_after - summary_before
            assert phase["cached_tokens"] == phase["calls"] * 16
            assert report["diagnostics"] and report["segments"] == 1
            assert any(
                row["provider_id"] == "fixture-provider" and row["stats"]["token_usage"]["input_cached"] == 16
                for row in [call.kwargs for call in stat_sink.await_args_list]
            )
            plugin.config.update(input_token_budget=32768, summary_token_budget=1024)
            results["real_summary_only_on_rollover_and_core_cache_stats_preserved"] = True

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
            assert "平台动作记录（已发生）" in json.dumps(captures[-1], default=str, ensure_ascii=False)
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

            # Force the observed provider fault through real hooks and delivery.
            # Private execution keeps the raw model result, public history only
            # contains the guarded text actually sent to the platform.
            from native_mm_probe.multimodal.adapters.astrbot import guard_chain

            leaked = (
                '[message_metadata={"event_id":"invented","sender_id":"fixture-bot","name":"bot",'
                '"time":"fake","source":"self"}][reply_to="output-guard"][mention_id="a"]正文一'
                '\n\n[message_metadata={"event_id":"invented2","sender_id":"fixture-bot","name":"bot",'
                '"time":"fake","source":"self"}]正文二'
            )
            query = event("output-guard", [At(qq="fixture-bot"), Plain("具体问题")], group="output-fixture")
            await record(plugin, query)
            with patch.object(
                provider,
                "text_chat",
                new=AsyncMock(return_value=LLMResponse(role="assistant", completion_text=leaked)),
            ):
                output_flow = plugin.respond(query)
                try:
                    await anext(output_flow)
                    state = query.get_extra(module.OWNER)
                    protocol = await plugin.service.journal.execution(state.gid, query.unified_msg_origin)
                    assert "invented" in json.dumps(protocol)
                    await call_event_hook(query, EventType.OnDecoratingResultEvent)
                    assert query.get_result().get_plain_text() == "正文一\n\n正文二"
                    await query.send(query.get_result())
                    sent_chain = state.transport.original.call_args.args[0]
                    assert sent_chain.get_plain_text() == "正文一\n\n正文二"
                    try:
                        await anext(output_flow)
                    except StopAsyncIteration:
                        pass
                finally:
                    await output_flow.aclose()
            room = state.selection.anchor["room"]
            visible = [
                e for e in await plugin.service.journal.recent(room, 2**63 - 1, 0, 20) if e["kind"] == "self"
            ]
            assert len(visible) == 1 and visible[0]["parts"][-1]["text"] == "正文一\n\n正文二"
            assert "message_metadata" not in json.dumps(
                context.conversation_manager.update_conversation.call_args.kwargs["history"][-1]
            )

            # The transport catches late additions, even if decoration was
            # skipped. It joins text fragments to parse a split JSON header.
            from astrbot.core.message.message_event_result import MessageChain

            fragmented = MessageChain(
                [Reply(id="question"), At(qq="a"), Plain(leaked[:40]), Plain(leaked[40:])]
            )
            guarded = guard_chain(fragmented)
            assert guarded.chain[:2] == fragmented.chain[:2]
            assert guarded.get_plain_text() == "正文一\n\n正文二"
            assert "message_metadata" in fragmented.get_plain_text()
            literal = "```json\n" + leaked + "\n```"
            assert guard_chain(MessageChain([Plain(literal)])).get_plain_text() == literal
            header_only = MessageChain([Plain(leaked.split("正文一", 1)[0])])
            assert "格式异常" in guard_chain(header_only).get_plain_text()
            late = event("late-output", [At(qq="fixture-bot"), Plain("late")], group="late-output-fixture")
            await record(plugin, late)
            anchor = late.get_extra(module.OWNER + "_anchor")
            gid = await plugin.service.journal.begin(anchor["room"], anchor["seq"])
            tracker = plugin.service.gateway.track_transport(late, plugin.service.journal, gid)
            try:
                await late.send(fragmented)
                assert tracker.original.call_args.args[0].get_plain_text() == "正文一\n\n正文二"
                assert tracker.successes == 1
            finally:
                tracker.restore()
            results["model_metadata_filtered_before_send_and_public_history_with_private_audit_intact"] = True
            results["late_transport_guard_preserves_reply_at_and_literal_code"] = True

            # Replay real core send_message_to_user + native poke calls. The
            # platform is mocked, but routing, runner, tools, hooks and ACK
            # publication use the actual SDK. No production messages or stats.
            from astrbot.core.tools.message_tools import SendMessageToUserTool
            from native_mm_probe.multimodal.adapters.astrbot import RequestContext

            marker = '[戳一戳事件={"actor_id":"67890","target_id":"12345"}]'
            toolset.add_tool(SendMessageToUserTool())
            context.send_message = AsyncMock(return_value=True)
            plugin.config.update(
                enable_poke_after_reply=True, poke_after_reply_probability=1, poke_after_reply_delay=0
            )
            for failure in (False, True):
                seed = event(
                    "native-tool-" + str(failure),
                    [At(qq="67890"), Plain("戳我")],
                    sender="12345",
                    group="86420" if not failure else "86421",
                )
                seed.message_obj.self_id = "67890"
                bot = SimpleNamespace(
                    api=SimpleNamespace(
                        call_action=AsyncMock(side_effect=RuntimeError("fixture") if failure else None)
                    ),
                    send_group_msg=AsyncMock(side_effect=[{"message_id": 445566}, {"message_id": 445567}]),
                )
                query = AiocqhttpMessageEvent(
                    seed.message_str, seed.message_obj, seed.platform_meta, seed.session_id, bot
                )
                query.is_at_or_wake_command = True
                await record(plugin, query)
                sequence = [
                    LLMResponse(
                        role="assistant",
                        completion_text="",
                        tools_call_name=["send_message_to_user"],
                        tools_call_args=[{"messages": [{"type": "plain", "text": marker + "正文"}]}],
                        tools_call_ids=["call-send"],
                    ),
                    LLMResponse(
                        role="assistant",
                        completion_text="",
                        tools_call_name=["native_poke_current_sender"],
                        tools_call_args=[{}],
                        tools_call_ids=["call-poke"],
                    ),
                    LLMResponse(
                        role="assistant",
                        completion_text="",
                        tools_call_name=["native_poke_current_sender"],
                        tools_call_args=[{}],
                        tools_call_ids=["call-poke-duplicate"],
                    ),
                    LLMResponse(role="assistant", completion_text="fixture_reply"),
                ]
                with patch.object(provider, "text_chat", new=AsyncMock(side_effect=sequence)):
                    tool_flow = plugin.respond(query)
                    try:
                        await anext(tool_flow)
                        state = query.get_extra(module.OWNER)
                        assert context.send_message.await_count == 0  # Same session uses event transport.
                        assert bot.send_group_msg.call_args.kwargs["message"] == [
                            {"type": "text", "data": {"text": "正文"}}
                        ]
                        assert state.transport.successes == 1
                        assert query.get_extra("_send_message_to_user_current_session_plain_texts") == [
                            "正文"
                        ]
                        bot.api.call_action.assert_awaited_once_with(
                            "send_poke", group_id=int(query.get_group_id()), user_id=12345
                        )
                        public = await plugin.service.journal.recent(
                            state.selection.anchor["room"], 2**63 - 1, 0, 20
                        )
                        assert len([e for e in public if e["kind"] == "action"]) == (0 if failure else 1)
                        assert [e for e in public if e["kind"] == "self"][0]["parts"] == [
                            {"type": "text", "text": "正文"}
                        ]
                        raw = await plugin.service.journal.execution(state.gid, query.unified_msg_origin)
                        arguments = json.loads(raw[0]["tool_calls"][0]["function"]["arguments"])
                        assert arguments["messages"][0]["text"] == marker + "正文"
                        assert ("未确认成功" if failure else "已成功向当前发送者") in json.dumps(
                            raw, ensure_ascii=False
                        )
                        action_tool = state.request.func_tool.get_tool("native_poke_current_sender")
                        wrong_context = SimpleNamespace(context=SimpleNamespace(event=seed))
                        assert "未执行" in await action_tool.call(wrong_context)
                        assert "未执行" in await action_tool.call(
                            SimpleNamespace(context=SimpleNamespace(event=query)), target_id="99999"
                        )
                        await query.send(query.get_result())
                        await call_event_hook(query, EventType.OnAfterMessageSentEvent)
                        await call_event_hook(query, EventType.OnAfterMessageSentEvent)
                        assert bot.api.call_action.await_count == 1  # No random duplicate after reply.
                        try:
                            await anext(tool_flow)
                        except StopAsyncIteration:
                            pass
                    finally:
                        await tool_flow.aclose()
            assert "native_poke_current_sender" not in {t.name for t in toolset.tools}
            scoped = RequestContext(context, query)
            elsewhere = MessageChain([Plain("other session")])
            assert await scoped.send_message("other-session", elsewhere)
            context.send_message.assert_awaited_once_with("other-session", elsewhere)
            results["core_message_tool_guarded_and_public_ack_recorded"] = True
            results["native_poke_tool_success_failure_and_duplicate_calls_match_platform_actions"] = True
            results["poke_tool_bound_to_current_event_and_shared_tools_unchanged"] = True

            # Rename + real history tool loop, entirely in a fixture room.
            from datetime import datetime, timedelta, timezone

            from astrbot.core.tools.message_tools import GetGroupMessageHistoryTool

            original_history_tool = GetGroupMessageHistoryTool()
            toolset.add_tool(original_history_tool)
            old_card = event("alias-old", [Plain("假期酒店很贵")], sender="1234567890", group="alias-fixture")
            old_card.message_obj.sender.nickname = "Traveler（10.3～10.7 北城）"
            await record(plugin, old_card)
            ask = event(
                "alias-ask",
                [At(qq="fixture-bot"), Plain("我去哪了")],
                sender="1234567890",
                group="alias-fixture",
            )
            ask.message_obj.sender.nickname = "小旅"
            await record(plugin, ask)
            history_sequence = [
                LLMResponse(
                    role="assistant",
                    completion_text="",
                    tools_call_name=["get_group_message_history"],
                    tools_call_args=[{"keyword": "长假", "sender": "1234567890"}],
                    tools_call_ids=["call-history"],
                ),
                LLMResponse(role="assistant", completion_text="fixture_reply"),
            ]
            with patch.object(provider, "text_chat", new=AsyncMock(side_effect=history_sequence)) as offline:
                history_flow = plugin.respond(ask)
                try:
                    await anext(history_flow)
                    alias_state = ask.get_extra(module.OWNER)
                    encoded = json.dumps(offline.call_args_list[-1].kwargs, default=str, ensure_ascii=False)
                    assert "same_sender_recent_fallback" in encoded and "Traveler" in encoded
                    assert "1234567890" in encoded and "酒店很贵" in encoded
                    history_tool = alias_state.request.func_tool.get_tool("get_group_message_history")
                    assert history_tool is not original_history_tool
                    assert toolset.get_tool("get_group_message_history") is original_history_tool
                    assert "未执行" in await history_tool.call(
                        SimpleNamespace(context=SimpleNamespace(event=old_card))
                    )
                    assert "未执行" in await history_tool.call(
                        SimpleNamespace(context=SimpleNamespace(event=ask)), room="other"
                    )
                    frame = await plugin.service.journal.frame(alias_state.selection.anchor["seq"])
                    assert "本群成员身份关联" not in json.dumps(frame, ensure_ascii=False)
                    await ask.send(ask.get_result())
                    await call_event_hook(ask, EventType.OnAfterMessageSentEvent)
                    try:
                        await anext(history_flow)
                    except StopAsyncIteration:
                        pass
                finally:
                    await history_flow.aclose()
            renamed = event(
                "alias-renamed",
                [At(qq="fixture-bot"), Plain("还记得吗")],
                sender="1234567890",
                group="alias-fixture",
            )
            renamed.message_obj.sender.nickname = "小旅（常驻南城）"
            await record(plugin, renamed)
            rename_flow = plugin.respond(renamed)
            try:
                await anext(rename_flow)
                renamed_state = renamed.get_extra(module.OWNER)
                tail = json.dumps(renamed_state.request.extra_user_content_parts, ensure_ascii=False)
                assert "Traveler" in tail and "小旅（常驻南城）" in tail
                assert renamed_state.selection.scope == alias_state.selection.scope
                assert await plugin.service.journal.frame(alias_state.selection.anchor["seq"]) == frame
                assert renamed_state.request.extra_user_content_parts[-1]["text"].startswith(
                    "[native_turn_control]"
                )
                assert "本群成员身份关联" not in renamed_state.request.system_prompt.split("你正在群聊", 1)[0]
                query_tool = renamed_state.request.func_tool.get_tool("get_group_message_history")
                ctx = SimpleNamespace(context=SimpleNamespace(event=renamed))
                cfg["provider_ltm_settings"]["group_message_history_enable"] = True
                created = datetime.now(timezone.utc) - timedelta(minutes=1)
                records = [
                    SimpleNamespace(
                        id=10,
                        sender_id="1234567890",
                        sender_name="更早名片",
                        created_at=created,
                        content={"type": "user", "message": [{"type": "plain", "text": "更早旅行"}]},
                    ),
                    SimpleNamespace(
                        id=200,
                        sender_id="1234567890",
                        sender_name="未来名片",
                        created_at=created,
                        content={"type": "user", "message": [{"type": "plain", "text": "未来正文"}]},
                    ),
                ]
                context.message_history_manager = SimpleNamespace(get=AsyncMock(return_value=records))
                renamed.set_extra("_current_platform_message_history_id", 100)
                legacy = json.loads(await query_tool.call(ctx, source="session", sender="更早名片"))
                assert legacy["messages"][0]["sender_id"] == "1234567890"
                assert legacy["identities"][0]["current_name"] == "小旅（常驻南城）"
                assert "未来" not in json.dumps(legacy, ensure_ascii=False)
                context.message_history_manager.get.assert_awaited_once_with(
                    platform_id=renamed.get_platform_id(), user_id=renamed.unified_msg_origin, page_size=700
                )
                assert "error" in json.loads(await query_tool.call(ctx, source="foreign"))
                await plugin.service.journal.reset(renamed_state.selection.anchor["room"])
                assert "error" in json.loads(await query_tool.call(ctx, source="session"))
                cfg["provider_ltm_settings"]["group_message_history_enable"] = False
            finally:
                await rename_flow.aclose()
            results["rename_links_full_sender_id_and_history_tool_fallback_in_real_runner"] = True
            results["rename_tail_preserves_scope_frozen_frames_and_shared_history_tool"] = True
            results["legacy_history_keeps_full_ids_scope_watermark_and_reset_boundary"] = True

            # Real OpenAI SDK + core provider, with an in-memory HTTP transport.
            # Verify the actual serialized body after the core's conversions.
            import inspect

            from openai import _base_client

            http = getattr(_base_client, "httpx", None) or getattr(_base_client, "httpx2", None)
            if http is None:
                http = importlib.import_module("httpx")
            bodies, wire_records = [], []

            def wire_response(request):
                body = json.loads(request.content)
                bodies.append(body)
                step = len(bodies)
                choice = {"role": "assistant", "content": "fixture_wire_reply"}
                finish = "stop"
                if step == 1 and any(
                    t.get("function", {}).get("name") == "fixture_lookup" for t in body.get("tools", [])
                ):
                    choice = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "wire-call",
                                "type": "function",
                                "function": {"name": "fixture_lookup", "arguments": "{}"},
                            }
                        ],
                    }
                    finish = "tool_calls"
                return http.Response(
                    200,
                    json={
                        "id": "fixture-completion",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [{"index": 0, "message": choice, "finish_reason": finish}],
                        "usage": {
                            "prompt_tokens": 5000 + step * 1000,
                            "completion_tokens": 20,
                            "total_tokens": 5020 + step * 1000,
                            "prompt_tokens_details": {"cached_tokens": step * 2000},
                        },
                    },
                )

            mock_http = http.AsyncClient(transport=http.MockTransport(wire_response))
            wire_cfg = {
                "id": "fixture-wire-provider",
                "type": "openai_chat_completion",
                "key": ["fixture-key"],
                "api_base": "https://fixture.invalid/v1",
                "model": "fixture-wire-model",
                "modalities": ["text", "image", "tool_use"],
                "max_context_tokens": 0,
                "custom_extra_body": {"reasoning_effort": "low"},
            }
            with patch.object(
                runner_module.ProviderOpenAIOfficial, "_create_http_client", new=lambda self, conf: mock_http
            ):
                wire_provider = runner_module.ProviderOpenAIOfficial(wire_cfg, cfg["provider_settings"])
            try:
                wire_mod = importlib.import_module("native_mm_probe.multimodal.wire")

                async def observe_wire(payload):
                    wire_records.append(payload)

                observer = wire_mod.WireObserver(observe_wire)
                before_create = wire_provider.client.chat.completions.create
                owned = runner_module.strict_provider(wire_provider, observer)
                another = runner_module.strict_provider(wire_provider, wire_mod.WireObserver(observe_wire))
                assert owned.client is not wire_provider.client and another.client is not owned.client
                assert owned.client._client is wire_provider.client._client
                assert wire_provider.client.chat.completions.create == before_create
                assert inspect.signature(before_create) != inspect.signature(
                    owned.client.chat.completions.create
                )
                scoped = runner_module.RequestContext(context, observer=observer)
                with patch.object(context, "get_provider_by_id", return_value=wire_provider):
                    fallback = scoped.get_provider_by_id("fixture")
                    assert fallback.client is not wire_provider.client
                    await fallback.text_chat(prompt="fixture direct call", contexts=[])
                assert len(wire_records) == 1 and wire_records[0]["input_tokens"] == 6000
                bodies.clear()
                wire_records.clear()
                results["wire_clients_are_request_owned_and_fallbacks_observed"] = True

                context.get_using_provider_async.return_value = wire_provider
                context.get_provider_by_id.return_value = wire_provider

                async def lookup(ctx):
                    return "PRIVATE_WIRE_TOOL_RESULT"

                tool.handler = lookup
                context.get_llm_tool_manager.return_value = SimpleNamespace(
                    get_full_tool_set=lambda: ToolSet([tool])
                )
                wire_query = event(
                    "wire-question", [At(qq="fixture-bot"), Plain("检查原图与工具"), Image.fromBytes(PNG)]
                )
                await record(plugin, wire_query)
                with patch(
                    "astrbot.core.astr_main_agent.retrieve_knowledge_base", new=AsyncMock(return_value=None)
                ):
                    async for _ in plugin.respond(wire_query):
                        pass
                wire_state = wire_query.get_extra(module.OWNER)
                status = await plugin.service.journal.status(wire_state.selection.anchor["room"])
                rows = [
                    r["payload"]
                    for r in reversed(status["wire_diagnostics"])
                    if r["payload"]["generation"] == wire_state.gid
                ]
                assert len(bodies) == 2 and len(rows) == 2
                assert [r["stage"] for r in rows] == ["first", "tool"]
                assert [r["input_tokens"] for r in rows] == [6000, 7000]
                assert [r["cached_tokens"] for r in rows] == [2000, 4000]
                assert bodies[0]["reasoning_effort"] == "low" and bodies[0]["tools"]
                assert rows[0]["tools"] == wire_mod.hashed(bodies[0]["tools"])
                assert rows[1]["messages"] == [wire_mod.hashed(m) for m in bodies[1]["messages"]]
                assert rows[0]["images"] == 1 and rows[1]["images"] == 1
                assert "data:image" not in json.dumps(rows) and "PRIVATE_WIRE" not in json.dumps(rows)
                assert "tool" in [m["role"] for m in bodies[1]["messages"]]
                aggregate = stat_sink.await_args.kwargs["stats"]["token_usage"]
                assert aggregate == {"input_other": 7000, "input_cached": 6000, "output": 40}
                assert wire_provider.client.chat.completions.create == before_create
                results["final_sdk_body_image_tools_hashes_and_per_call_usage_match_http"] = True
                results["first_and_tool_cache_usage_do_not_double_count_core_stats"] = True
            finally:
                await wire_provider.client.close()
        finally:
            for name in ("flow", "first", "second", "pure"):
                if name in locals():
                    await locals()[name].aclose()
            await plugin.terminate()
    print("SDK_PROBE_RESULT=" + json.dumps(results, ensure_ascii=False))


asyncio.run(main())
