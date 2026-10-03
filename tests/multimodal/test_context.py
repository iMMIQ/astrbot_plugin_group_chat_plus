import asyncio
import base64
import io
import json
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from multimodal.bridge import (
    additions,
    payload_cost,
    protocol_messages,
    request_snapshot,
    restore_legacy_wrappers,
    rewrite,
)
from multimodal.context import ContextLimit, ContextSelector
from multimodal.storage.assets import MediaStore
from multimodal.storage.sqlite import Journal

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)


@pytest.fixture
async def env(tmp_path):
    journal = Journal(tmp_path)
    media = MediaStore(journal, tmp_path / "media", {})
    selector = ContextSelector(journal, media, {})
    yield (journal, media, selector)
    await journal.close()


async def add(journal, eid, sender="a", parts=None, received=1000, room="r"):
    return (
        await journal.add(
            room, eid, sender, sender, parts or [{"type": "text", "text": eid}], received=received
        )
    )[0]


async def image(journal, media, eid, sender="a", received=1000, room="r", color=None):
    data = PNG
    if color is not None:
        buffer = io.BytesIO()
        Image.new("RGB", (1, 1), color).save(buffer, format="PNG")
        data = buffer.getvalue()
    mid = await media.capture(room, "base64://" + base64.b64encode(data).decode())
    await media.tasks[mid]
    return await add(journal, eid, sender, [{"type": "image", "media_id": mid}], received, room)


def count_images(history, current):
    return sum(
        (
            p.get("type") == "image_url"
            for message in history
            for p in message.get("content", [])
            if isinstance(p, dict)
        )
    ) + sum((p.get("type") == "image_url" for p in current))


@pytest.mark.parametrize("gap", [3, 21, 119])
def test_previous_image_after_at(env, gap):

    async def scenario():
        (j, m, s) = env
        first = await image(j, m, "image")
        await add(j, "interruption", "b", received=1001)
        current = await add(j, "question", received=1000 + gap)
        selected = await s.candidates(current)
        (history, parts) = await s.assemble(selected)
        assert first["seq"] in selected.protected
        assert count_images(history, parts) == 1
        assert all((msg["role"] == "user" for msg in history))
        assert '"sender_id": "a"' in history[0]["content"][0]["text"]
        assert '"sender_id": "b"' in history[1]["content"][0]["text"]

    asyncio.run(scenario())


def test_same_message_single_image(env):

    async def scenario():
        (j, m, s) = env
        current = await image(j, m, "current")
        (history, parts) = await s.assemble(await s.candidates(current))
        assert history == [] and count_images(history, parts) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("noise_sender", ["a", "b"])
def test_fast_room_does_not_evict_same_sender_image(env, noise_sender):

    async def scenario():
        (j, m, s) = env
        first = await image(j, m, "image")
        for i in range(80):
            await add(j, str(i), noise_sender, received=1001 + i / 10)
        current = await add(j, "question", received=1021)
        selection = await s.candidates(current)
        (history, parts) = await s.assemble(selection)
        assert first["seq"] in selection.protected
        assert count_images(history, parts) == 1

    asyncio.run(scenario())


def test_explicit_cross_sender_reply_outranks_own_image(env):

    async def scenario():
        (j, m, s) = env
        own = await image(j, m, "own")
        other = await image(j, m, "other", "b", 1001)
        current = await add(
            j,
            "question",
            parts=[{"type": "reply", "event_id": "other"}, {"type": "text", "text": "看这个"}],
            received=2000,
        )
        selection = await s.candidates(current)
        assert other["seq"] in selection.protected and own["seq"] not in selection.protected
        (history, parts) = await s.assemble(selection)
        assert count_images(history, parts) == 1

    asyncio.run(scenario())


async def test_snapshot_and_room_isolation(env):
    (j, m, s) = env
    await add(j, "same-id", room="elsewhere")
    current = await add(j, "question")
    await add(j, "future", received=1001)
    selected = await s.candidates(current)
    assert [e["event_id"] for e in selected.events] == ["question"]


async def test_duplicate_and_reset_persist(env):
    (j, m, s) = env
    (_, first) = await j.add("r", "id", "a", "A", [])
    (_, second) = await j.add("r", "id", "a", "A", [{"type": "text", "text": "different"}])
    assert first and (not second)
    assert (await j.get("r", "id"))["parts"] == []
    await j.reset("r")
    current = await add(j, "new")
    assert [e["event_id"] for e in (await s.candidates(current)).events] == ["new"]


def test_budget_never_drops_protected_current(env):

    async def scenario():
        (j, m, s) = env
        await add(j, "old", parts=[{"type": "text", "text": "旧" * 2000}])
        current = await add(j, "question", received=1001)
        s.config = {"input_token_budget": 1000}
        (history, parts) = await s.assemble(await s.candidates(current))
        assert history == [] and "question" in json.dumps(parts)
        with pytest.raises(ContextLimit):
            await s.assemble(await s.candidates(current), fixed_tokens=999)

    asyncio.run(scenario())


def test_too_many_associated_images_is_explicit_error(env):

    async def scenario():
        (j, m, s) = env
        for i in range(7):
            await image(j, m, str(i), received=1000 + i, color=(i, 0, 0))
        current = await add(j, "question", received=1010)
        with pytest.raises(ContextLimit):
            await s.assemble(await s.candidates(current))

    asyncio.run(scenario())


def test_failed_attachment_is_not_silent(env):

    async def scenario():
        (j, m, s) = env
        mid = await m.capture("r", "/no/such/image")
        await m.tasks[mid]
        current = await add(j, "question", parts=[{"type": "image", "media_id": mid}])
        (history, parts) = await s.assemble(await s.candidates(current))
        assert count_images(history, parts) == 0
        assert "failed" in json.dumps(parts) and "尚未看到图片内容" in json.dumps(parts, ensure_ascii=False)

    asyncio.run(scenario())


def test_media_reboot_and_cleanup_lease(tmp_path):

    async def scenario():
        j = Journal(tmp_path)
        m = MediaStore(j, tmp_path / "media", {"media_retention_hours": 0})
        first = await image(j, m, "image")
        mid = first["parts"][0]["media_id"]
        m.lease([mid])
        await m.cleanup()
        assert (await j.media(mid))["status"] == "ready"
        m.release([mid])
        await j.close()
        j = Journal(tmp_path)
        m = MediaStore(j, tmp_path / "media", {"media_retention_hours": 0})
        assert await m.data_uri(mid, "r")
        await m.cleanup()
        assert (await j.media(mid))["status"] == "expired"
        await j.close()

    asyncio.run(scenario())


def test_protocol_is_private_and_keeps_tool_ids(env):

    async def scenario():
        (j, m, s) = env
        anchor = await add(j, "original", received=time.time())
        gid = await j.begin("r", anchor["seq"])
        protocol = [
            {
                "role": "assistant",
                "content": "调用工具",
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "t1"},
            {"role": "assistant", "content": "完成"},
        ]
        await j.finish(gid, "generated", protocol, "bot")
        current = await add(j, "next", received=time.time())
        (history, parts) = await s.assemble(await s.candidates(current))
        assert [msg["role"] for msg in history] == ["user"]
        execution = await j.execution(gid)
        assert execution == protocol
        assert execution[1]["tool_call_id"] == "t1"
        assert await j.execution(gid, "another-session") is None

    asyncio.run(scenario())


def test_bridge_preserves_persona_tools_extensions_and_current_once(env):

    async def scenario():
        (j, m, s) = env
        await image(j, m, "image")
        current = await add(j, "question", received=1003)
        old = [{"role": "user", "content": "legacy_history"}]
        tools = SimpleNamespace(openai_schema=lambda: [{"name": "lookup"}])
        req = SimpleNamespace(
            conversation=SimpleNamespace(history=json.dumps(old)),
            contexts=old.copy(),
            extra_user_content_parts=[],
            image_urls=[],
            system_prompt="persona",
            prompt="question",
            func_tool=tools,
        )
        snap = request_snapshot(req)
        req.contexts.append({"role": "user", "content": "third_party_memory"})
        req.system_prompt += "\nthird_party_persona_extension"
        req.prompt += "\nprompt_extension"
        external = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()},
        }
        req.extra_user_content_parts = [external, {"type": "text", "text": "knowledge"}]
        await rewrite(req, await s.candidates(current), s, snap, "question", 32768)
        assert req.func_tool is tools
        assert "third_party_persona_extension" in req.system_prompt
        assert "legacy_history" not in json.dumps(req.contexts)
        assert "third_party_memory" in json.dumps(req.contexts)
        assert "knowledge" in json.dumps(req.extra_user_content_parts)
        assert "prompt_extension" in json.dumps(req.extra_user_content_parts)
        assert req.prompt == "" and req.image_urls == []
        assert count_images(req.contexts, req.extra_user_content_parts) == 2

    asyncio.run(scenario())


def test_external_image_cost_does_not_count_base64_as_text():
    (cost, count) = payload_cost(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "x" * 100000}}], 1600
    )
    assert count == 1 and cost < 1700


def test_hot_reload_restores_only_exact_legacy_wrappers():

    async def original(*args):
        pass

    async def _make_tracking_wrapper(event, *args, __original=original):
        pass

    _make_tracking_wrapper.__module__ = "data.plugins.astrbot_plugin_group_chat_plus.main"
    ours = SimpleNamespace(handler=_make_tracking_wrapper, _gcp_tracking_wrapped=True)
    unrelated = SimpleNamespace(handler=original, _gcp_tracking_wrapped=True)
    assert restore_legacy_wrappers([ours, unrelated]) == 1
    assert ours.handler is original and unrelated.handler is original
    assert restore_legacy_wrappers([ours, unrelated]) == 0


async def test_generation_failure_and_restart_does_not_replay(env):
    (j, m, s) = env
    anchor = await add(j, "question")
    gid = await j.begin("r", anchor["seq"])
    await j.finish(gid, "failed")
    await j.settle(gid)
    assert (await j.status("r"))["last_generation"]["status"] == "failed"
    assert (await j.status("r"))["events"] == 1


def test_diff_handles_identical_messages_by_multiplicity():
    assert additions([{"role": "user", "content": "x"}] * 2, [{"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "x"}
    ]


def test_tool_image_context_kept_without_replaying_plain_users():
    image = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}],
    }
    assert protocol_messages(
        [{"role": "user", "content": "original"}, image, {"role": "assistant", "content": "answer"}]
    ) == [image, {"role": "assistant", "content": "answer"}]


async def test_anchor_claim_prevents_duplicate_batch_generation(env):
    (j, m, s) = env
    anchor = await add(j, "question")
    assert await j.begin("r", anchor["seq"])
    assert await j.begin("r", anchor["seq"]) is None
