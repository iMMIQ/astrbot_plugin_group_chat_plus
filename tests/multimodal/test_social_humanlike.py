"""Unit tests for human-like group chat behavior (P0/P1/P2)."""

from __future__ import annotations

import json
import time
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from multimodal.adapters.onebot import PlatformAdapter
from multimodal.bridge import build_gate_rules
from multimodal.config import Settings, resolve_tz
from multimodal.context import ContextSelector
from multimodal.output import polish_reply_text, split_reply_bubbles
from multimodal.participation import Participation
from multimodal.qq_faces import face_label, resolve_react_emoji, send_emoji_like
from multimodal.routing import TurnRoute, control_part
from multimodal.segments import SegmentManager
from multimodal.storage.assets import MediaStore
from multimodal.storage.sqlite import Journal


def test_qq_faces_decoding_and_react_resolution():
    assert face_label(277) == "[QQ表情:汪汪]"
    assert face_label("344") == "[QQ表情:大怨种]"
    assert face_label("182") == "[QQ表情:笑哭]"
    assert face_label("99999") == "[QQ表情 id=99999]"
    assert resolve_react_emoji("狗头") == "277"
    assert resolve_react_emoji("笑死") == "182"
    assert resolve_react_emoji("裂开") == "312"
    assert resolve_react_emoji("76") == "76"
    assert resolve_react_emoji("not-a-face") is None


@pytest.mark.asyncio
async def test_send_emoji_like_calls_onebot_action():
    call_action = AsyncMock(return_value={"status": "ok"})
    ev = SimpleNamespace(
        get_platform_name=lambda: "aiocqhttp",
        bot=SimpleNamespace(api=SimpleNamespace(call_action=call_action)),
    )
    ok = await send_emoji_like(ev, "123456", "277")
    assert ok is True
    call_action.assert_awaited_once_with(
        "set_msg_emoji_like",
        message_id=123456,
        emoji_id="277",
        emoji_type="1",
        set=True,
    )
    assert await send_emoji_like(ev, "not-numeric-uuid", "277") is False


def test_output_polish_and_newline_bubble_splitting():
    assert polish_reply_text("今天吃火锅。", strip_trailing_period=True) == "今天吃火锅"
    assert polish_reply_text("第一句。第二句。", strip_trailing_period=True) == "第一句。第二句。"
    assert (
        polish_reply_text("```python\nprint(1)\n```", strip_trailing_period=True)
        == "```python\nprint(1)\n```"
    )

    bubbles = split_reply_bubbles(
        "猪皮都出来了。\n\n下一个是不是猪蹄。", max_bubbles=4, strip_trailing_period=True
    )
    assert bubbles == ["猪皮都出来了", "下一个是不是猪蹄"]

    # Structured lists & code blocks stay unsplit
    card = "📈 600900.SH\n- 现价 31.8\n- 跌幅 0.5%"
    assert split_reply_bubbles(card, max_bubbles=4, strip_trailing_period=True) == [card]
    code = "看看代码\n```py\nx = 1\n```"
    assert split_reply_bubbles(code, max_bubbles=4, strip_trailing_period=True) == [code]

    # Cap max_bubbles by merging trailing lines
    many = split_reply_bubbles("a\nb\nc\nd\ne", max_bubbles=3)
    assert many == ["a", "b", "c\nd\ne"]


def test_config_group_overrides_and_timezone():
    cfg = Settings(
        {
            "timezone_name": "Asia/Shanghai",
            "auto_candidate_probability": 0.02,
            "group_overrides": json.dumps(
                {"480234482": {"auto_candidate_probability": 0.05, "plus_one_enabled": True}}
            ),
        }
    )
    tz = resolve_tz(cfg)
    assert tz != timezone.utc
    assert cfg.for_group("480234482")["auto_candidate_probability"] == 0.05
    assert cfg.for_group("480234482")["plus_one_enabled"] is True
    assert cfg.for_group("999999")["auto_candidate_probability"] == 0.02

    with pytest.raises(ValueError):
        Settings({"group_overrides": '{"480234482": {"auto_candidate_probability": 2.0}}'})


def test_participation_thread_continuation_nicknames_anti_buttin_and_plus_one():
    cfg = Settings(
        {
            "auto_reply_enabled": True,
            "auto_candidate_probability": 0.0,
            "auto_reply_cooldown": 60,
            "thread_continuation_seconds": 50,
            "thread_max_turns": 2,
            "thread_min_interval_seconds": 0.0,
            "nicknames": ["大肥鱼", "肥鱼"],
            "nickname_candidate_probability": 1.0,
            "ignored_bot_ids": ["4015116734"],
            "plus_one_enabled": True,
            "plus_one_min_users": 3,
            "plus_one_probability": 1.0,
            "plus_one_break_probability": 0.0,
            "emoji_react_enabled": True,
            "emoji_react_cooldown": 120,
        }
    )
    p = Participation(cfg)
    room = "room-1"

    # Anti-butt-in: @third_party rejected even with nickname
    at_other = {
        "sender": "u1",
        "parts": [{"type": "mention", "target_id": "u2"}, {"type": "text", "text": "大肥鱼笑死"}],
    }
    assert p.classify_passive(room, at_other, "bot", cfg) is None

    # Ignored bot rejected
    bot_msg = {"sender": "4015116734", "parts": [{"type": "text", "text": "大肥鱼"}]}
    assert p.classify_passive(room, bot_msg, "bot", cfg) is None

    # Peer poke never auto-triggers
    peer_poke = {"sender": "u1", "parts": [{"type": "poke", "actor_id": "u1", "target_id": "u2"}]}
    assert p.classify_passive(room, peer_poke, "bot", cfg) is None

    # Active thread continuation triggers immediately for same partner
    p.observe_bot_reply(room, "u1", 10, auto=False)
    followup = {"sender": "u1", "parts": [{"type": "text", "text": "真的假的"}]}
    assert p.classify_passive(room, followup, "bot", cfg) == "thread_continue"

    # Streak cap enforced
    p.observe_bot_reply(room, "u1", 11, auto=True, is_thread=True)
    p.observe_bot_reply(room, "u1", 12, auto=True, is_thread=True)
    assert p.classify_passive(room, followup, "bot", cfg) is None

    # Mid-sentence nickname triggers nickname_mention
    nick_msg = {"sender": "u3", "parts": [{"type": "text", "text": "这大肥鱼是想被吃了"}]}
    assert p.classify_passive(room, nick_msg, "bot", cfg) == "nickname_mention"

    # +1 repeat detection needs 3 distinct senders
    now = time.monotonic()
    p.observe_member(room, "u1", "每天都能看到更便宜的股票", now)
    p.observe_member(room, "u1", "每天都能看到更便宜的股票", now + 1)
    assert p.detect_plus_one(room, "u1", "每天都能看到更便宜的股票", cfg) is None
    p.observe_member(room, "u2", "每天都能看到更便宜的股票", now + 2)
    p.observe_member(room, "u3", "每天都能看到更便宜的股票", now + 3)
    assert p.detect_plus_one(room, "u3", "每天都能看到更便宜的股票", cfg) == "repeat"
    p.mark_plus_one(room, "每天都能看到更便宜的股票")
    assert p.detect_plus_one(room, "u4", "每天都能看到更便宜的股票", cfg) is None

    # Gate parsing: new action schema + fenced JSON + legacy bool schema
    assert Participation.parse_gate('{"action":"react","emoji":"狗头"}') == {
        "action": "react",
        "emoji": "狗头",
    }
    assert Participation.parse_gate('```json\n{"action":"reply"}\n```') == {"action": "reply", "emoji": None}
    assert Participation.parse_gate('{"reply":true}') == {"action": "reply", "emoji": None}
    assert Participation.parse_gate('{"action":"skip"}') == {"action": "skip", "emoji": None}
    assert Participation.parse_gate('{"action":"react"}') == {"action": "skip", "emoji": None}


def test_build_gate_rules_and_routing_local_time():
    rules = build_gate_rules(persona_hint="你是大肥鱼", react_enabled=True, trigger_hint="续聊判断")
    assert "大肥鱼" in rules and "续聊判断" in rules and "狗头" in rules
    route = TurnRoute("bot", "e1", "thread_continue")
    assert route.explicit is False
    part = control_part(
        route,
        {"event_id": "e2", "sender": "u1", "received": 1764230400.0},
        tz=resolve_tz({"timezone_name": "Asia/Shanghai"}),
    )
    payload = json.loads(part["text"].removeprefix("[native_turn_control]"))
    assert payload["mode"] == "auto_approved"
    assert payload["reason"] == "thread_continue"
    assert "local_time" in payload


@pytest.mark.asyncio
async def test_sqlite_helpers_poke_name_backfill_and_room_memory(tmp_path):
    cfg = Settings({"timezone_name": "Asia/Shanghai", "room_memory_enabled": True})
    journal = Journal(tmp_path)
    media = MediaStore(journal, tmp_path / "media", cfg)
    adapter = PlatformAdapter(journal, media)
    selector = ContextSelector(journal, media, cfg)
    segments = SegmentManager(selector)
    await journal.ready()
    try:
        room = '["napcat","bot","group","123"]'
        e1, _ = await journal.add(
            room, "1001", "111", "爱吃鱼的群友", [{"type": "text", "text": "hello"}], received=1000.0
        )
        await journal.add(room, "1002", "222", "路人甲", [{"type": "text", "text": "world"}], received=1001.0)
        assert await journal.latest_sender_name(room, "111") == "爱吃鱼的群友"
        assert await journal.count_member_after(room, e1["seq"]) == 1

        # Poke notice with raw numeric sender_name gets backfilled with real group card
        poke_ev = SimpleNamespace(
            get_platform_name=lambda: "aiocqhttp",
            get_platform_id=lambda: "napcat",
            get_self_id=lambda: "bot",
            get_group_id=lambda: "123",
            get_sender_id=lambda: "111",
            get_sender_name=lambda: "111",
            get_messages=lambda: [],
            message_obj=SimpleNamespace(
                message_id="poke-uuid-1",
                raw_message={
                    "post_type": "notice",
                    "notice_type": "notify",
                    "sub_type": "poke",
                    "user_id": 111,
                    "target_id": "bot",
                    "self_id": "bot",
                    "group_id": "123",
                },
            ),
        )
        ingested, _ = await adapter.ingest(poke_ev)
        assert ingested["name"] == "爱吃鱼的群友"

        # Room memory persists, snapshots into segment prefix (KV-cache stable), and clears on reset
        mem = {"facts": ["群里常聊恒生科技"], "members": {"111": "爱叫大肥鱼红烧"}}
        await journal.save_room_memory(room, mem)
        assert await journal.room_memory(room, 86400) == mem

        sel1 = await selector.candidates(e1)
        stable1, _, _ = await segments.assemble(sel1, "scope-a", 100, 8000, 0, room_memory=mem)
        assert any("群里常聊恒生科技" in json.dumps(m, ensure_ascii=False) for m in stable1)

        # Updating DB room_memory does NOT mutate already-frozen segment prefix
        await journal.save_room_memory(room, {"facts": ["新事实"], "members": {}})
        e3, _ = await journal.add(
            room, "1003", "111", "爱吃鱼的群友", [{"type": "text", "text": "again"}], received=1002.0
        )
        sel2 = await selector.candidates(e3)
        stable2, _, _ = await segments.assemble(
            sel2, "scope-a", 100, 8000, 0, room_memory={"facts": ["新事实"], "members": {}}
        )
        assert stable2[0] == stable1[0]

        # /mmreset clears room_memory too
        await journal.reset(room)
        assert await journal.room_memory(room, 86400) is None
    finally:
        await media.close()
        await journal.close()


@pytest.mark.asyncio
async def test_transport_smart_decoration_and_bubble_splitting(tmp_path):
    pytest.importorskip("astrbot")
    from astrbot.core.message.components import At, Plain, Reply
    from astrbot.core.message.message_event_result import MessageEventResult, ResultContentType

    from multimodal.adapters.astrbot import AstrBotGateway, TransportTracker
    from multimodal.models import Selection, TurnState

    cfg = Settings(
        {
            "smart_reply_decoration": True,
            "split_newline_bubbles": True,
            "max_reply_bubbles": 4,
            "strip_trailing_period": True,
            "bubble_typing_delay_max": 0.0,
        }
    )

    # 1) decorate_bubbles splits multiline casual text and strips trailing period
    res = MessageEventResult([Plain("猪皮都出来了。\n\n下一个是不是猪蹄。")]).set_result_content_type(
        ResultContentType.LLM_RESULT
    )
    fake_ev = SimpleNamespace(
        _res=res,
        get_result=lambda: res,
        set_result=lambda r: None,
    )
    AstrBotGateway.decorate_bubbles(fake_ev, cfg)
    assert [p.text for p in res.chain] == ["猪皮都出来了", "下一个是不是猪蹄"]

    # 2) TransportTracker smart decoration scenarios
    journal = Journal(tmp_path)
    await journal.ready()
    try:
        sent_chains = []

        async def run_case(*, anchor_eid, is_poke, auto, intervening, body_text, bubbles=1):
            sent_chains.clear()
            room = '["napcat","bot","group","123"]'
            e, _ = await journal.add(room, anchor_eid, "111", "alice", [{"type": "text", "text": "hi"}])
            gid = await journal.begin(room, e["seq"], "umo")

            async def orig_send(chain):
                sent_chains.append(list(chain.chain))
                return {"message_id": "999"}

            ev = SimpleNamespace(
                send=orig_send,
                get_group_id=lambda: "123",
                message_obj=SimpleNamespace(raw_message={}),
            )
            sel = Selection(e, [e], {e["seq"]}, set(), {})
            state = TurnState(sel, gid, None, auto=auto, is_poke_trigger=is_poke, agent_finished=True)
            tracker = TransportTracker(
                ev,
                journal,
                gid,
                turn_state=state,
                config=cfg,
                intervening_fn=AsyncMock(return_value=intervening),
            )
            for idx in range(bubbles):
                parts = (
                    [Reply(id=anchor_eid), At(qq="111"), Plain("\n" + body_text)]
                    if idx == 0
                    else [Plain("第二句")]
                )
                mc = MessageEventResult(parts).set_result_content_type(ResultContentType.LLM_RESULT)
                await tracker.send(mc)
            tracker.restore()
            return list(sent_chains)

        # Case A: explicit short reply with 0 intervening -> bare message (no Reply, no At, no leading \n)
        out = await run_case(anchor_eid="10001", is_poke=False, auto=False, intervening=0, body_text="草")
        assert len(out[0]) == 1 and isinstance(out[0][0], Plain) and out[0][0].text == "草"

        # Case B: explicit reply with 2 intervening -> Reply ONLY (no At, no leading \n), bubble 2 bare
        out = await run_case(
            anchor_eid="10002", is_poke=False, auto=False, intervening=2, body_text="接住了", bubbles=2
        )
        assert isinstance(out[0][0], Reply) and str(out[0][0].id) == "10002"
        assert isinstance(out[0][1], Plain) and out[0][1].text == "接住了"
        assert not any(isinstance(c, At) for c in out[0])
        assert len(out[1]) == 1 and isinstance(out[1][0], Plain) and out[1][0].text == "第二句"

        # Case C: poke event with fake UUID message_id -> NEVER attach Reply; intervening=0 -> bare
        out = await run_case(
            anchor_eid="8bd24307a5ac407286fe792e58e1d69f",
            is_poke=True,
            auto=False,
            intervening=0,
            body_text="戳回去",
        )
        assert not any(isinstance(c, Reply) for c in out[0])
        assert not any(isinstance(c, At) for c in out[0])
        assert out[0][0].text == "戳回去"

        # Case D: poke event with intervening=3 -> At only (still NEVER fake UUID Reply)
        out = await run_case(
            anchor_eid="9bd24307a5ac407286fe792e58e1d69f",
            is_poke=True,
            auto=False,
            intervening=3,
            body_text="戳回去",
        )
        assert not any(isinstance(c, Reply) for c in out[0])
        assert isinstance(out[0][0], At) and str(out[0][0].qq) == "111"

        # Case E: auto chime-in with intervening=0 -> bare; intervening=1 -> Reply only
        out0 = await run_case(anchor_eid="10003", is_poke=False, auto=True, intervening=0, body_text="确实")
        assert len(out0[0]) == 1 and isinstance(out0[0][0], Plain)
        out1 = await run_case(anchor_eid="10004", is_poke=False, auto=True, intervening=1, body_text="确实")
        assert isinstance(out1[0][0], Reply) and not any(isinstance(c, At) for c in out1[0])
    finally:
        await journal.close()
