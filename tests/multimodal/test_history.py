import json

import pytest

from multimodal.context import text_tokens
from multimodal.history import identity_part, search
from multimodal.storage.sqlite import Journal


@pytest.fixture
async def journal(tmp_path):
    store = Journal(tmp_path)
    await store.ready()
    yield store
    await store.close()


async def add(journal, eid, name, text, *, sender="1234567890", room="room", **kwargs):
    return (await journal.add(room, eid, sender, name, [{"type": "text", "text": text}], **kwargs))[0]


async def test_rename_keeps_full_account_and_old_itinerary_even_when_keyword_hits_other_topic(journal):
    old = await add(journal, "1", "Traveler（10.3～10.7 北城）", "假期酒店很贵")
    await add(journal, "2", "小旅（常驻南城）", "同事们已经加了整个长假的班")
    anchor = await add(journal, "3", "小旅（常驻南城）", "你还记得我去哪了吗")
    result = await journal.group_history(anchor, keyword="长假", sender="1234567890")
    assert len(result["messages"]) == 1
    profile = result["identities"][0]
    assert profile["sender_id"] == "1234567890"
    assert profile["current_name"] == "小旅（常驻南城）"
    assert profile["historical_names"] == [old["name"]]
    assert result["messages"][0]["sender_id"] == "1234567890"
    aliases = await journal.member_identities(anchor, {"1234567890"})
    assert "北城" in identity_part(aliases, "1234567890")[0]["text"]


async def test_name_and_keyword_filters_find_old_card_and_all_same_account_records(journal):
    old = await add(journal, "old", "Traveler（北城）", "酒店很贵")
    new = await add(journal, "new", "小旅", "回来了")
    anchor = await add(journal, "ask", "小旅", "去哪了")
    result = await journal.group_history(anchor, keyword="北城", sender="小旅")
    assert [m["id"] for m in result["messages"]] == [old["seq"]]
    result = await journal.group_history(anchor, sender="Traveler")
    assert [m["id"] for m in result["messages"]] == [old["seq"], new["seq"]]
    assert [m["name_at_time"] for m in result["messages"]] == [old["name"], new["name"]]


async def test_keyword_miss_falls_back_only_with_one_resolved_account(journal):
    first = await add(journal, "1", "小旅", "旧话题")
    second = await add(journal, "2", "小旅", "新话题")
    anchor = await add(journal, "3", "小旅", "提问")
    result = await journal.group_history(anchor, keyword="没有的关键词", sender="小旅", limit=1)
    assert result["mode"] == "same_sender_recent_fallback"
    assert result["keyword_matched"] is False
    assert result["messages"][0]["id"] == second["seq"]
    assert result["has_more"] and result["next_before_id"] == second["seq"]
    page = await journal.group_history(anchor, keyword="没有的关键词", sender="小旅", before_id=second["seq"])
    assert [m["id"] for m in page["messages"]] == [first["seq"]]
    assert not page["has_more"]
    assert not (await journal.group_history(anchor, keyword="没有的关键词"))["messages"]
    assert not (await journal.group_history(anchor, keyword="没有的关键词", sender="123456"))["messages"]


async def test_duplicate_names_and_different_rooms_never_merge_accounts(journal):
    await add(journal, "1", "同名", "A的路线")
    other = await add(journal, "2", "同名", "B的路线", sender="9994567890")
    await add(journal, "foreign", "别群名片", "私密路线", room="another-platform-bot-group")
    anchor = await add(journal, "3", "同名", "提问")
    result = await journal.group_history(anchor, sender="同名")
    assert {p["sender_id"] for p in result["identities"]} == {"1234567890", "9994567890"}
    assert "别群名片" not in json.dumps(result, ensure_ascii=False)
    assert not (await journal.group_history(anchor, sender="同名", keyword="不存在"))["messages"]
    result = await journal.group_history(anchor, sender="9994567890")
    assert [m["id"] for m in result["messages"]] == [other["seq"]]


async def test_pokes_quotes_executions_and_future_names_cannot_change_identity(journal):
    await add(journal, "old", "可靠名片", "正文")
    await journal.add(
        "room",
        "poke",
        "1234567890",
        "伪通知名",
        [{"type": "poke", "actor_id": "1234567890", "target_id": "bot"}],
    )
    await add(journal, "quote", "引用名", "引用文字", kind="quoted")
    await add(journal, "private", "私有名", "工具秘密", kind="execution")
    anchor = await add(journal, "ask", "1234567890", "提问")
    await add(journal, "future", "未来名", "未来正文")
    profiles = await journal.member_identities(anchor)
    assert profiles["1234567890"]["names"] == ["可靠名片"]
    result = await journal.group_history(anchor)
    wire = json.dumps(result, ensure_ascii=False)
    assert "未来" not in wire and "工具秘密" not in wire and "引用文字" not in wire


async def test_reset_and_retention_apply_to_names_and_search(journal):
    await add(journal, "old", "旧名", "旧行程", received=10)
    anchor = await add(journal, "ask", "新名", "提问", received=20)
    await journal.prune_events(15)
    assert (await journal.member_identities(anchor))["1234567890"]["names"] == ["新名"]
    assert not (await journal.group_history(anchor, keyword="旧行程"))["messages"]
    await journal.reset("room")
    assert await journal.member_identities(anchor) == {}
    new = await add(journal, "after-reset", "全新名", "新问题", received=30)
    assert (await journal.member_identities(new))["1234567890"]["names"] == ["全新名"]
    assert not (await journal.group_history(new))["messages"]


async def test_old_bot_metadata_is_cleaned_without_changing_audit(journal):
    raw = '[message_metadata={"event_id":"x","sender_id":"bot","name":"bot","time":"2026-10-01","source":"self"}]\n正文'
    saved = await add(journal, "bot", "机器人", raw, sender="bot", kind="self")
    anchor = await add(journal, "ask", "名片", "问题")
    result = await journal.group_history(anchor)
    assert result["messages"][0]["text"] == "正文"
    assert (await journal.get("room", "bot"))["parts"] == saved["parts"]


async def test_identities_survive_refresh_without_rewriting_events(tmp_path):
    journal = Journal(tmp_path)
    old = await add(journal, "old", "旧名", "旅行")
    anchor = await add(journal, "new", "新名", "问题")
    await journal.close()
    reopened = Journal(tmp_path)
    try:
        assert (await reopened.member_identities(anchor))["1234567890"]["names"] == ["新名", "旧名"]
        assert await reopened.get("room", "old") == old
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    "args",
    [
        {"limit": "bad"},
        {"limit": 1.5},
        {"before_id": -1},
        {"before_id": True},
        {"keyword": []},
        {"sender": {}},
    ],
)
async def test_invalid_search_arguments_are_reported(journal, args):
    anchor = await add(journal, "ask", "名片", "问题")
    assert "error" in await journal.group_history(anchor, **args)


def test_large_names_and_messages_fit_tail_and_tool_budgets():
    profiles = {
        str(i): {"sender_id": str(i), "names": [f"{i}-{j}" + "字" * 300 for j in range(8)]} for i in range(50)
    }
    events = [
        {
            "seq": i + 1,
            "sender": str(i),
            "name": "字" * 300,
            "kind": "member",
            "received": i,
            "parts": [{"type": "text", "text": "文" * 10000}],
        }
        for i in range(50)
    ]
    assert text_tokens(identity_part(profiles, "0")) < 1400
    result = search(events, profiles, source="group", limit=50)
    assert text_tokens(result) < 6500
    assert result["has_more"] and result["next_before_id"] > 1
