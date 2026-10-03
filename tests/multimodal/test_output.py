import json
import sqlite3

import pytest
from test_regressions import member, picture
from test_regressions import state as regression_state

from multimodal.config import Settings
from multimodal.context import ContextSelector
from multimodal.models import FRAME_VERSION
from multimodal.output import strip_headers
from multimodal.segments import SegmentManager

state = regression_state
META = {"event_id": "invented", "sender_id": "bot", "name": "bot", "time": "fake", "source": "self"}
HEADER = "[message_metadata=" + json.dumps(META) + "]"
CONTROL = '[native_turn_control]{"bot_id":"bot","anchor_event_id":"a","sender_id":"u","reason":"mention_self","participation":"approved"}'


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (HEADER + '[reply_to="question"][mention_id="user"] 少爷什么时候打款', "少爷什么时候打款"),
        (HEADER + "第一段\n\n" + HEADER + "第二段", "第一段\n\n第二段"),
        (" \n" + HEADER + "\n回答", " \n回答"),
        (HEADER, ""),
        (CONTROL + "\n回答", "回答"),
        ('[native_bot_identity={"bot_id":"bot"}]回答', "回答"),
        ("[message_metadata=" + json.dumps(META, indent=2) + "]正文", "正文"),
        ("[message_metadata= \n" + json.dumps(META, indent=2) + "\n]正文", "正文"),
        (CONTROL.replace("]{", "]\n {") + "\n正文", "正文"),
        ("[message_metadata=" + json.dumps({**META, "name": '名字有 ] 和 "'}) + "]正文", "正文"),
    ],
)
def test_generated_headers_are_removed(raw, expected):
    assert strip_headers(raw) == expected
    assert strip_headers(expected) == expected


@pytest.mark.parametrize(
    "literal",
    [
        "普通回答里有 log、ERROR、时间 13:46 和 ID 12345",
        '普通 JSON：{"event_id":"a","sender_id":"b"}',
        "字段示例 " + HEADER,
        "`" + HEADER + "`",
        "```json\n" + HEADER + "\n```\n这就是标记",
        "~~~text\n" + CONTROL + "\n~~~\n正文",
        "    " + HEADER,
        "\t" + HEADER,
        "> " + HEADER,
        '[message_metadata={"event_id":"a"}]正常数据',
        '[message_metadata={"event_id":"a"',
        '[reply_to="question"]是字段示例',
        '[mention_id="user"]是字段示例',
    ],
)
def test_literals_and_code_examples_are_preserved(literal):
    assert strip_headers(literal) == literal


async def sent(journal, anchor, text):
    gid = await journal.begin("room", anchor["seq"])
    await journal.finish(gid, "generated", [{"role": "assistant", "content": text}])
    await journal.sent(
        gid,
        "ack",
        "bot-answer",
        [
            {"type": "reply", "event_id": anchor["event_id"]},
            {"type": "mention", "target_id": "alice"},
            {"type": "text", "text": text},
        ],
        "bot",
    )
    return gid


async def test_old_leak_is_clean_in_assistant_history_but_raw_audit_survives(state):
    journal, _, selector = state
    first = await member(journal, "question")
    raw = HEADER + '[reply_to="question"][mention_id="alice"]实际回答'
    gid = await sent(journal, first, raw)
    current = await member(journal, "followup")
    history, _ = await selector.assemble(await selector.candidates(current))
    assert [m["role"] for m in history] == ["user", "user", "assistant"]
    assert history[-1]["content"] == [{"type": "text", "text": "实际回答"}]
    facts = json.dumps(history[-2], ensure_ascii=False)
    assert "曾夹带内部格式标记" in facts
    assert "message_metadata" in facts and "bot-answer" in facts and "question" in facts and "alice" in facts
    assert (await journal.get("room", "bot-answer"))["parts"][-1]["text"] == raw
    assert (await journal.execution(gid))[0]["content"] == raw


async def test_stale_frames_replaced_once_and_new_frames_remain_immutable(state, tmp_path):
    journal, _, selector = state
    anchor = await member(journal, "first")
    with sqlite3.connect(tmp_path / "events.sqlite3") as db:
        db.execute(
            "INSERT INTO frames VALUES(?,1,?)",
            (anchor["seq"], json.dumps({"messages": [{"role": "assistant", "content": "STALE"}]})),
        )
    assert await journal.frame(anchor["seq"]) is None
    await selector.assemble(await selector.candidates(anchor))
    first = await journal.frame(anchor["seq"])
    assert first and "STALE" not in json.dumps(first)
    assert await journal.freeze(anchor["seq"], {"messages": []}) == first
    assert (await journal.freeze_many({anchor["seq"]: {"messages": []}}))[anchor["seq"]] == first
    with sqlite3.connect(tmp_path / "events.sqlite3") as db:
        assert (
            db.execute("SELECT version FROM frames WHERE event_seq=?", (anchor["seq"],)).fetchone()[0]
            == FRAME_VERSION
        )


async def test_old_segments_and_summaries_do_not_reintroduce_headers(state):
    journal, media, _ = state
    selector = ContextSelector(journal, media, Settings({}))
    manager = SegmentManager(selector)
    first = await member(journal, "question")
    await sent(journal, first, HEADER + "回答")
    current = await member(journal, "current")
    selection = await selector.candidates(current)
    await manager.assemble(selection, "scope", 100, None, 0)
    previous = await journal.segment("room", "scope")
    previous.pop("frame_version")
    previous["summary"] = [{"text": "STALE SUMMARY", "sources": [first["seq"]]}]
    previous["frames"] = {str(first["seq"]): {"messages": [{"role": "assistant", "content": "STALE FRAME"}]}}
    await journal.save_segment("room", "scope", previous)
    next_turn = await member(journal, "next")
    selection = await selector.candidates(next_turn)
    history, tail, canonical = await manager.assemble(selection, "scope", 100, None, 0)
    assert selection.rollover == "context_format"
    assert "STALE" not in json.dumps([history, tail, canonical])
    assert all("message_metadata" not in json.dumps(m) for m in history if m["role"] == "assistant")
    raw = await journal.get("room", "bot-answer")
    summary = await manager._summary([raw], [], None)
    assert "message_metadata" not in json.dumps(summary)
    assert "回答" in json.dumps(summary, ensure_ascii=False)


async def test_bot_image_and_caption_keep_native_image_and_pure_assistant_body(state):
    journal, media, selector = state
    source = await picture(journal, media, "original-image")
    mid = source["parts"][0]["media_id"]
    gid = await journal.begin("room", source["seq"])
    await journal.sent(
        gid,
        "ack",
        "bot-picture",
        [{"type": "image", "media_id": mid}, {"type": "text", "text": "图片说明"}],
        "bot",
    )
    current = await member(journal, "followup")
    history, _ = await selector.assemble(await selector.candidates(current))
    assert json.dumps(history).count("data:image/png;base64,") == 1
    assert history[-1] == {"role": "assistant", "content": [{"type": "text", "text": "图片说明"}]}
    assert all(
        p.get("type") != "image_url" for m in history if m["role"] == "assistant" for p in m["content"]
    )


async def test_metadata_envelopes_append_without_changing_previous_prefix(state):
    journal, media, _ = state
    manager = SegmentManager(ContextSelector(journal, media, Settings({})))
    first = await member(journal, "question")
    selection = await manager.selector.candidates(first)
    previous, _, _ = await manager.assemble(selection, "scope", 100, None, 0)
    await sent(journal, first, "实际回答")
    next_turn = await member(journal, "next")
    history, _, _ = await manager.assemble(
        await manager.selector.candidates(next_turn), "scope", 100, None, 0
    )
    assert history[: len(previous)] == previous
    assert history[-2]["role"] == "assistant" and history[-2]["content"] == [
        {"type": "text", "text": "实际回答"}
    ]
    assert history[-3]["role"] == "user"
