"""Public delivery, causal images and stable request prefix regressions."""

import json
import time

from test_regressions import member, picture
from test_regressions import state as regression_state

from multimodal.budget import BudgetEstimator
from multimodal.config import Settings
from multimodal.context import ContextSelector
from multimodal.segments import SegmentManager

state = regression_state


async def test_calibration_extends_segment_without_changing_frozen_prefix(state):
    journal, _, _ = state
    mm = await manager(state, input_token_budget=10000)
    first = await member(journal, "before-calibration", [{"type": "text", "text": "旅行计划" * 200}])
    initial, prefix, _, _ = await request(mm, first)
    frame = await journal.frame(first["seq"])
    for i in range(3):
        anchor = await member(journal, "after-" + str(i), [{"type": "text", "text": "旅行计划" * 200}])
    selection = await mm.selector.candidates(anchor)
    selection.estimator = BudgetEstimator(text_scale=0.6)
    stable, _, _ = await mm.assemble(selection, "policy", 100, None, 0)
    assert selection.segment_id == initial.segment_id
    assert stable[: len(prefix)] == prefix
    assert await journal.frame(first["seq"]) == frame
    assert len(selection.chosen_events) == 4
    assert selection.tokens <= 10000


async def manager(state, **options):
    journal, media, _ = state
    return SegmentManager(ContextSelector(journal, media, Settings(options)))


async def request(manager, anchor, scope="policy", summarize=None):
    selection = await manager.selector.candidates(anchor)
    wire, tail, canonical = await manager.assemble(selection, scope, 100, None, 0, summarize)
    return selection, wire, tail, canonical


async def test_unsent_protocol_and_failed_segment_never_enter_public_context(state):
    journal, _, selector = state
    anchor = await member(journal, "question")
    gid = await journal.begin("room", anchor["seq"], "private-session")
    protocol = [
        {"role": "tool", "content": "PRIVATE_SECRET", "tool_call_id": "call"},
        {"role": "assistant", "content": "unsent"},
    ]
    await journal.finish(gid, "generated", protocol)
    next_turn = await member(journal, "next")
    assert "PRIVATE_SECRET" not in json.dumps(await selector.assemble(await selector.candidates(next_turn)))
    assert "unsent" not in json.dumps(await selector.assemble(await selector.candidates(next_turn)))
    await journal.sent(gid, "one", parts=[{"type": "text", "text": "actually visible"}], self_id="bot")
    await journal.delivery_attempt(gid, "two")
    await journal.delivery_failed(gid, "two")
    await journal.settle(gid, completed=True)
    wire = json.dumps(await selector.assemble(await selector.candidates(next_turn)))
    assert "actually visible" in wire and "PRIVATE_SECRET" not in wire and "unsent" not in wire
    assert await journal.execution(gid, "private-session") == protocol
    assert await journal.execution(gid, "another-session") is None


async def test_quote_bot_answer_restores_original_image_beyond_initial_window(state):
    journal, media, selector = state
    photo = await picture(journal, media, "photo")
    question = await member(journal, "look")
    await journal.link("room", question["seq"], [photo["seq"]])
    gid = await journal.begin("room", question["seq"])
    await journal.finish(gid, "generated", [{"role": "assistant", "content": "photo answer"}])
    await journal.sent(gid, "ack", "platform-answer", [{"type": "text", "text": "photo answer"}], "bot")
    mm = await manager(state)
    fresh = (
        await journal.add(
            "room", "fresh", "alice", "Alice", [{"type": "text", "text": "fresh"}], received=time.time() + 900
        )
    )[0]
    _, first, _, _ = await request(mm, fresh)
    followup = (
        await journal.add(
            "room",
            "late",
            "bob",
            "Bob",
            [{"type": "reply", "event_id": "platform-answer"}, {"type": "text", "text": "那这张呢"}],
            received=time.time() + 901,
        )
    )[0]
    selection = await selector.candidates(followup)
    assert photo["seq"] in selection.protected
    assert photo["parts"][0]["media_id"] in selection.primary_media
    assert json.dumps(await selector.assemble(selection)).count("data:image/png;base64,") == 1
    # The old quote is supplemental; it does not rewrite the active prefix.
    _, second, tail, _ = await request(mm, followup)
    assert second[: len(first)] == first
    assert "本轮引用补充" in json.dumps(tail, ensure_ascii=False)


async def test_segment_appends_beyond_seed_limit_and_survives_reload(state):
    mm = await manager(state, max_context_messages=3)
    previous = []
    identifier = None
    for index in range(12):
        anchor = await member(state[0], "turn-" + str(index))
        selection, stable, _, _ = await request(mm, anchor)
        assert stable[: len(previous)] == previous
        identifier = identifier or selection.segment_id
        assert selection.segment_id == identifier
        previous = stable
        mm = await manager(state, max_context_messages=3)
    assert len(previous) == 12


async def test_bulk_rollover_summarizes_once_and_summary_sources_are_checked(state):
    journal, _, _ = state
    mm = await manager(state, input_token_budget=2600, summary_token_budget=250)
    calls = []

    async def summarize(rules, data):
        calls.append(data)
        first = data["messages"][0]
        return json.dumps(
            {"items": [{"text": first["sender"] + ": 决定出游，路线未定", "sources": [first["seq"]]}]}
        )

    identifiers = []
    for index in range(8):
        anchor = await member(journal, "long-" + str(index), [{"type": "text", "text": "旅行计划" * 40}])
        selection, _, _, _ = await request(mm, anchor, summarize=summarize)
        identifiers.append(selection.segment_id)
    assert 0 < len(calls) < 8
    assert len(set(identifiers)) == len(calls) + 1
    saved = await journal.segment("room", "policy")
    assert saved["summary"]
    source = saved["summary"][0]["sources"][0]
    await journal.call("prune_events", (await journal.by_seq("room", source))["received"] + 0.000001)
    new = await member(journal, "after-source-deletion")
    selection, stable, _, _ = await request(mm, new, summarize=summarize)
    assert selection.rollover == "source_removed"
    assert "决定出游" not in json.dumps(stable, ensure_ascii=False)


async def test_reset_and_policy_isolation_invalidate_stale_segment(state):
    journal, _, _ = state
    mm = await manager(state)
    initial = await member(journal, "old")
    first, _, _, _ = await request(mm, initial, "persona-a")
    another, _, _, _ = await request(mm, initial, "persona-b")
    assert first.segment_id != another.segment_id
    assert await journal.segment("room", "persona-a")
    await journal.reset("room")
    assert await journal.segment("room", "persona-a") is None
    new = await member(journal, "new")
    fresh, wire, _, _ = await request(mm, new, "persona-a")
    assert fresh.segment_id != first.segment_id
    assert "old" not in json.dumps(wire)
    assert not await journal.dependencies("room", initial["seq"])


async def test_summary_failure_uses_attributed_bounded_excerpts(state):
    mm = await manager(state, summary_token_budget=200)
    anchor = await member(state[0], "decision", [{"type": "text", "text": "周六出发，酒店还没定"}])

    async def fail(rules, data):
        assert data["messages"][0]["sender"] == "alice"
        assert data["messages"][0]["name"] == "Alice"
        raise TimeoutError

    output = await mm._summary([anchor], [], fail)
    assert output == [{"text": "alice (Alice): 周六出发，酒店还没定", "sources": [anchor["seq"]]}]


async def test_diagnostics_are_hash_only_and_distinguish_system_changes(state):
    journal = state[0]
    base = {
        "system": "hash-a",
        "tools": "hash-tools",
        "messages": ["hash-1", "tail-1"],
        "stable": ["hash-1"],
        "segment": "one",
    }
    first = await journal.diagnose("room", "policy", "reply", base)
    assert first["prefix_change"] == "first_request"
    second = await journal.diagnose(
        "room",
        "policy",
        "reply",
        base | {"messages": ["hash-1", "hash-2", "tail-2"], "stable": ["hash-1", "hash-2"]},
    )
    assert second["common_prefix_messages"] == 1 and second["prefix_change"] == "append"
    third = await journal.diagnose("room", "policy", "reply", base | {"system": "hash-b"})
    assert third["prefix_change"] == "system" and third["common_prefix_messages"] == 0


async def test_v2_migration_quarantines_unsent_and_tool_trace(tmp_path):
    import sqlite3

    from multimodal.storage.sqlite import Journal

    now = time.time()
    with sqlite3.connect(tmp_path / "events.sqlite3") as db:
        db.executescript("""
            CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT NOT NULL,event_id TEXT NOT NULL,
                sender TEXT NOT NULL,name TEXT NOT NULL,kind TEXT NOT NULL,received REAL NOT NULL,parts TEXT NOT NULL,
                causal_anchor INTEGER,UNIQUE(room,event_id));
            CREATE TABLE generations(id TEXT PRIMARY KEY,room TEXT NOT NULL,anchor INTEGER NOT NULL,status TEXT NOT NULL,
                created REAL NOT NULL,sent_parts INTEGER DEFAULT 0,delivery TEXT NOT NULL DEFAULT 'pending',
                pipeline_complete INTEGER NOT NULL DEFAULT 0,UNIQUE(room,anchor));
            PRAGMA user_version=2;
        """)
        protocol = [
            {"role": "tool", "content": "PRIVATE_TOOL_SECRET", "tool_call_id": "call"},
            {"role": "assistant", "content": "answer"},
        ]
        for index, delivery in enumerate(("sent", "partial", "uncertain")):
            anchor, reply = index * 2 + 1, index * 2 + 2
            db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)",
                (anchor, "room", "question-" + delivery, "alice", "Alice", "member", now, "[]", None),
            )
            db.execute(
                "INSERT INTO generations VALUES(?,?,?,?,?,?,?,?)",
                (delivery, "room", anchor, "generated", now, int(delivery != "uncertain"), delivery, 1),
            )
            db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    reply,
                    "room",
                    "generation:" + delivery,
                    "bot",
                    "bot",
                    "self",
                    now,
                    json.dumps([{"type": "protocol", "messages": protocol}]),
                    anchor,
                ),
            )
    journal = Journal(tmp_path)
    try:
        await journal.ready()
        public = await journal.recent("room", 99, 0, 99)
        assert len([e for e in public if e["kind"] == "self"]) == 1
        assert "PRIVATE_TOOL_SECRET" not in json.dumps(public)
        assert (await journal.get("room", "generation:sent"))["parts"] == [{"type": "text", "text": "answer"}]
        assert (await journal.get("room", "generation:uncertain"))["kind"] == "execution"
        assert await journal.execution("sent", "legacy") == protocol
        assert await journal.execution("sent", "another-session") is None
        assert (tmp_path / "events.pre-v3.sqlite3").is_file()
    finally:
        await journal.close()
