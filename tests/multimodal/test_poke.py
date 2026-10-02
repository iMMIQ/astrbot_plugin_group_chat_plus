import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from multimodal.adapters.onebot import PlatformAdapter
from multimodal.adapters.poke import accepted, notice, send
from multimodal.context import ContextSelector
from multimodal.storage.sqlite import Journal


def event(**changes):
    raw = dict(
        post_type="notice",
        notice_type="notify",
        sub_type="poke",
        self_id=67890,
        user_id=12345,
        target_id=67890,
        group_id=24680,
    )
    raw.update(changes)
    return SimpleNamespace(
        message_obj=SimpleNamespace(raw_message=raw, message_id="poke-1"),
        get_platform_name=lambda: "aiocqhttp",
        get_platform_id=lambda: "fixture-platform",
        get_group_id=lambda: "24680",
        get_self_id=lambda: "67890",
        get_sender_id=lambda: "12345",
        get_sender_name=lambda: "Alice",
        get_messages=lambda: [],
        bot=SimpleNamespace(api=SimpleNamespace(call_action=AsyncMock())),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"post_type": "message"},
        {"notice_type": "group_increase"},
        {"sub_type": "other"},
        {"self_id": 111},
        {"group_id": 111},
        {"user_id": 111},
        {"target_id": ""},
    ],
)
def test_reject_forged_or_mismatched_notice(changes):
    assert notice(event(**changes)) is None


def test_poke_modes_and_group_scope():
    ev = event()
    poke = notice(ev)
    assert accepted({}, ev, poke)
    assert not accepted({"poke_message_mode": "ignore"}, ev, poke)
    assert not accepted({"poke_enabled_groups": ["another-room"]}, ev, poke)
    peer = notice(event(target_id=99999))
    assert not accepted({}, ev, peer)
    assert accepted({"poke_message_mode": "all"}, ev, peer)


def test_empty_notice_preserves_identity_and_history(tmp_path):

    async def scenario():
        journal = Journal(tmp_path)
        try:
            media = SimpleNamespace()
            adapter = PlatformAdapter(journal, media)
            ev = event()
            (anchor, inserted) = await adapter.ingest(ev)
            assert inserted and anchor["sender"] == "12345"
            assert anchor["parts"] == [{"type": "poke", "actor_id": "12345", "target_id": "67890"}]
            (history, current) = await ContextSelector(journal, media, {}).assemble(
                await ContextSelector(journal, media, {}).candidates(anchor)
            )
            assert not history and "戳一戳事件" in current[1]["text"]
            assert "12345" in current[1]["text"] and "67890" in current[1]["text"]
            assert (await adapter.ingest(ev))[1] is False
            assert ev.get_messages() == []
        finally:
            await journal.close()

    asyncio.run(scenario())


def test_send_uses_original_group_and_actor():

    async def scenario():
        ev = event()
        assert await send(ev, "12345")
        ev.bot.api.call_action.assert_awaited_once_with("send_poke", group_id=24680, user_id=12345)
        assert not await send(ev, "invalid")
        assert ev.bot.api.call_action.await_count == 1

    asyncio.run(scenario())
