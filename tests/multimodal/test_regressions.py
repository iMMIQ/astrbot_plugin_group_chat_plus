import asyncio
import base64
import io
import json
import sqlite3
import time

import pytest
from PIL import Image

from multimodal.config import Settings
from multimodal.context import ContextSelector
from multimodal.participation import Participation
from multimodal.storage.assets import MediaStore
from multimodal.storage.sqlite import Journal


@pytest.fixture
async def state(tmp_path):
    journal = Journal(tmp_path)
    await journal.ready()
    media = MediaStore(journal, tmp_path / "media", {})
    yield journal, media, ContextSelector(journal, media, {})
    await media.close()
    await journal.close()


async def member(journal, eid, parts=None):
    return (await journal.add("room", eid, "alice", "Alice", parts or [{"type": "text", "text": eid}]))[0]


async def picture(journal, media, eid, color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    mid = await media.capture("room", "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    await media.wait([mid])
    return await member(journal, eid, [{"type": "image", "media_id": mid}])


async def test_queued_turn_sees_completed_previous_reply_but_not_future_members(state):
    journal, media, selector = state
    first = await member(journal, "first")
    generation = await journal.begin("room", first["seq"])
    second = await member(journal, "second")
    await journal.finish(
        generation, "generated", [{"role": "assistant", "content": "answer to first"}], "bot"
    )
    await journal.sent(
        generation, "acked", parts=[{"type": "text", "text": "answer to first"}], self_id="bot"
    )
    await member(journal, "future member message")
    selection = await selector.candidates(second)
    history, current = await selector.assemble(selection)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[-1]["content"][-1] == {"type": "text", "text": "answer to first"}
    assert "future member message" not in json.dumps(history)
    assert "second" in json.dumps(current)
    assert selection.view_seq > second["seq"]


async def test_inline_source_is_not_persisted_and_same_asset_is_sent_once(state, tmp_path):
    journal, media, selector = state
    await picture(journal, media, "picture-1")
    await picture(journal, media, "picture-2")
    anchor = await member(journal, "question")
    history, current = await selector.assemble(await selector.candidates(anchor))
    wire = json.dumps([history, current])
    assert wire.count("data:image/png;base64,") == 1
    assert len(await journal.asset_paths()) == 1
    await journal.close()
    with sqlite3.connect(tmp_path / "events.sqlite3") as db:
        assert not any(
            "base64" in source or source.startswith("data:")
            for (source,) in db.execute("SELECT source FROM media")
        )
        assert not any("base64" in payload for (payload,) in db.execute("SELECT payload FROM frames"))


async def test_frozen_current_becomes_identical_history_prefix(state):
    journal, media, selector = state
    first = await picture(journal, media, "picture")
    history1, current1 = await selector.assemble(await selector.candidates(first))
    assert history1 == []
    second = await member(journal, "question")
    history2, current2 = await selector.assemble(await selector.candidates(second))
    assert history2[0] == {"role": "user", "content": current1}
    assert "question" in json.dumps(current2)
    await member(journal, "other")
    third = await member(journal, "next question")
    history3, _ = await selector.assemble(await selector.candidates(third))
    assert history3[: len(history2)] == history2


async def test_gate_budget_does_not_mutate_or_freeze_main_selection(state):
    journal, media, selector = state
    first = await member(journal, "older", [{"type": "text", "text": "旧" * 1200}])
    anchor = await member(journal, "question")
    selection = await selector.candidates(anchor)
    original = list(selection.events)
    await selector.assemble(selection, total_limit=1000, freeze=False)
    assert selection.events == original
    assert await journal.frame(first["seq"]) is None
    history, _ = await selector.assemble(selection, total_limit=10000)
    assert "旧" in json.dumps(history, ensure_ascii=False)


async def test_delivery_deduplicates_and_only_completed_pipeline_is_sent(state):
    journal, media, selector = state
    first = await member(journal, "first")
    gid = await journal.begin("room", first["seq"])
    await journal.finish(gid, "generated", [{"role": "assistant", "content": "answer"}], "bot")
    assert await journal.sent(gid, "receipt-1")
    assert not await journal.sent(gid, "receipt-1")
    await journal.settle(gid, completed=False)
    status = (await journal.status("room"))["last_generation"]
    assert status["delivery"] == "partial" and status["sent_parts"] == 1
    assert status["generation_status"] == "generated"
    assert await journal.sent(gid, "receipt-2")
    await journal.settle(gid, completed=True)
    assert (await journal.status("room"))["last_generation"]["delivery"] == "sent"


async def test_worker_io_does_not_block_event_loop(state):
    journal, media, selector = state
    pending = journal._executor.submit(time.sleep, 0.15)
    tick = asyncio.Event()
    asyncio.get_running_loop().call_later(0.02, tick.set)
    started = time.monotonic()
    await asyncio.wait_for(tick.wait(), 0.08)
    assert time.monotonic() - started < 0.08
    await asyncio.wrap_future(pending)


async def test_v1_migration_preserves_events_and_assets_scrubs_payload_and_is_idempotent(tmp_path):
    path = tmp_path / "events.sqlite3"
    original_file = tmp_path / "original.png"
    original_file.write_bytes(b"fixture asset")
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT,room TEXT NOT NULL,event_id TEXT NOT NULL,
              sender TEXT NOT NULL,name TEXT NOT NULL,kind TEXT NOT NULL,received REAL NOT NULL,parts TEXT NOT NULL,UNIQUE(room,event_id));
            CREATE TABLE media(id TEXT PRIMARY KEY,room TEXT NOT NULL,source TEXT NOT NULL,status TEXT NOT NULL,
              path TEXT,mime TEXT,width INTEGER,height INTEGER,bytes INTEGER,sha TEXT,created REAL NOT NULL,error TEXT);
            CREATE TABLE generations(id TEXT PRIMARY KEY,room TEXT NOT NULL,anchor INTEGER NOT NULL,status TEXT NOT NULL,
              created REAL NOT NULL,sent_parts INTEGER DEFAULT 0,UNIQUE(room,anchor));
            CREATE TABLE room_state(room TEXT PRIMARY KEY,floor INTEGER NOT NULL);
            PRAGMA user_version=1;
        """)
        db.execute(
            "INSERT INTO events VALUES(1,?,?,?,?,?,?,?)",
            ("room", "first", "alice", "Alice", "member", time.time(), "[]"),
        )
        db.execute(
            "INSERT INTO generations VALUES(?,?,?,?,?,?)", ("generation", "room", 1, "sent", time.time(), 1)
        )
        db.execute(
            "INSERT INTO events VALUES(2,?,?,?,?,?,?,?)",
            ("room", "generation:generation", "bot", "bot", "self", time.time(), "[]"),
        )
        db.execute(
            "INSERT INTO media VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "image",
                "room",
                "data:image/png;base64,fixture",
                "ready",
                str(original_file),
                "image/png",
                1,
                1,
                13,
                "sha",
                time.time(),
                None,
            ),
        )
    for _ in range(2):
        journal = Journal(tmp_path)
        try:
            await journal.ready()
            assert (await journal.get("room", "generation:generation"))["causal_anchor"] == 1
            assert (await journal.media("image"))["source"] == "inline:sha"
            assert (await journal.status("room"))["last_generation"]["delivery"] == "sent"
            assert len(await journal.asset_paths()) == 1
        finally:
            await journal.close()
    assert (tmp_path / "events.pre-v2.sqlite3").is_file()


def test_config_migration_preserves_participation_and_poke():
    config = Settings(
        {
            "auto_reply_enabled": True,
            "enable_poke_after_reply": False,
            "poke_reverse_on_poke_probability": 0.7,
        }
    )
    assert config["auto_reply_enabled"] is True
    assert config["enable_poke_after_reply"] is False
    assert config["poke_reverse_on_poke_probability"] == 0.7
    assert config["config_version"] == 2
    with pytest.raises(ValueError):
        config.update(auto_candidate_probability=1.2)
    assert config["auto_candidate_probability"] == 0.02


@pytest.mark.parametrize(
    "text", ["not JSON", "null", "[]", '{"reply":"true"}', '{"reply":1}', '{"reply":false}']
)
def test_participation_rejects_invalid_gate(text):
    assert not Participation.accepts(text)


async def test_pending_image_becomes_tail_attachment_without_rewriting_history(state):
    journal, media, selector = state
    original = await picture(journal, media, "picture")
    mid = original["parts"][0]["media_id"]
    await journal.media_update(mid, status="pending")
    _, first = await selector.assemble(await selector.candidates(original))
    assert "pending" in json.dumps(first)
    await journal.media_update(mid, status="ready")
    question = await member(journal, "look at my image")
    history, current = await selector.assemble(await selector.candidates(question))
    assert history[0] == {"role": "user", "content": first}
    assert json.dumps(current).count("data:image/png;base64,") == 1


async def test_room_queue_is_bounded_and_cancel_releases_slot(tmp_path):
    from types import SimpleNamespace

    from multimodal.context import ContextLimit
    from multimodal.conversation import ConversationService

    service = ConversationService(
        SimpleNamespace(data_root=lambda: tmp_path), Settings({"max_pending_turns": 2})
    )
    async with service.room("room"):
        entered = asyncio.Event()

        async def queued():
            async with service.room("room"):
                entered.set()

        pending = asyncio.create_task(queued())
        await asyncio.sleep(0.01)
        assert service.rooms["room"].pending == 2
        with pytest.raises(ContextLimit):
            async with service.room("room"):
                pass
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert service.rooms["room"].pending == 1
        assert not entered.is_set()
    assert service.rooms["room"].pending == 0
    await service.terminate()


async def test_tool_turn_is_not_split_by_budget(state):
    journal, media, selector = state
    first = await member(journal, "first", [{"type": "text", "text": "长" * 2000}])
    gid = await journal.begin("room", first["seq"])
    await journal.finish(
        gid,
        "generated",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "t", "content": "result"},
            {"role": "assistant", "content": "answer"},
        ],
        "bot",
    )
    second = await member(journal, "second")
    history, _ = await selector.assemble(await selector.candidates(second), total_limit=1000)
    assert history == []


async def test_arrival_tracking_waits_for_same_sender_ingest_only(tmp_path):
    from types import SimpleNamespace

    from multimodal.adapters.onebot import PlatformAdapter

    journal = Journal(tmp_path)
    adapter = PlatformAdapter(journal, SimpleNamespace())
    started = asyncio.Event()
    release = asyncio.Event()

    async def ingest(event, room, received):
        started.set()
        await release.wait()
        return received

    adapter._ingest = ingest
    event = SimpleNamespace(
        get_platform_id=lambda: "platform",
        get_self_id=lambda: "bot",
        get_group_id=lambda: "group",
        get_sender_id=lambda: "alice",
    )
    pending = asyncio.create_task(adapter.ingest(event))
    await started.wait()
    room = next(iter(adapter.inflight.values()))[0]
    await adapter.settle_arrivals(room, "bob", 0, time.time(), 0.1)
    waited = asyncio.create_task(adapter.settle_arrivals(room, "alice", 0, time.time(), 1))
    await asyncio.sleep(0.01)
    assert not waited.done()
    release.set()
    await waited
    await pending
    assert not adapter.inflight
    await journal.close()


async def test_send_failure_cannot_be_reported_as_sent_on_normal_pipeline_end(state):
    journal, media, selector = state
    anchor = await member(journal, "question")
    gid = await journal.begin("room", anchor["seq"])
    await journal.finish(gid, "generated", [{"role": "assistant", "content": "answer"}], "bot")
    await journal.delivery_attempt(gid, "first")
    await journal.sent(gid, "first")
    await journal.delivery_attempt(gid, "second")
    await journal.delivery_failed(gid, "second")
    await journal.settle(gid, completed=True)
    status = (await journal.status("room"))["last_generation"]
    assert status["delivery"] == "partial"
    assert status["sent_parts"] == 1


@pytest.mark.parametrize("finish,complete,delivery", [("failed", True, "sent"), (None, False, "partial")])
async def test_generation_failure_does_not_erase_known_delivery(state, finish, complete, delivery):
    journal, media, selector = state
    anchor = await member(journal, "question")
    gid = await journal.begin("room", anchor["seq"])
    if finish:
        await journal.finish(gid, finish)
    await journal.delivery_attempt(gid, "receipt")
    await journal.sent(gid, "receipt")
    await journal.settle(gid, completed=complete)
    last = (await journal.status("room"))["last_generation"]
    assert last["generation_status"] == "failed"
    assert last["delivery"] == delivery


async def test_cancelled_asset_io_finishes_before_refresh_cleanup():
    import threading

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def write():
        started.set()
        release.wait(2)
        finished.set()

    operation = asyncio.create_task(MediaStore._io(write))
    await asyncio.to_thread(started.wait)
    operation.cancel()
    await asyncio.sleep(0.01)
    assert not operation.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert finished.is_set()
