"""Versioned room journal. SQL never uses sender-only or naked group scopes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..models import FRAME_VERSION, Event, GenerationStatus, Part


class SQLiteRepository:
    def __init__(self, root: Path):
        self.closed = False
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "events.sqlite3", timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 3:
            raise RuntimeError("Unsupported journal schema; use the matching plugin version")
        if version == 1:
            backup = root / "events.pre-v2.sqlite3"
            if not backup.exists():
                with sqlite3.connect(backup) as target:
                    self.db.backup(target)
        if version in (1, 2):
            backup = root / "events.pre-v3.sqlite3"
            if not backup.exists():
                with sqlite3.connect(backup) as target:
                    self.db.backup(target)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                room TEXT NOT NULL, event_id TEXT NOT NULL,
                sender TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL,
                received REAL NOT NULL, parts TEXT NOT NULL,
                UNIQUE(room, event_id)
            );
            CREATE INDEX IF NOT EXISTS event_room_seq ON events(room, seq);
            CREATE INDEX IF NOT EXISTS event_sender_time ON events(room, sender, received);
            CREATE INDEX IF NOT EXISTS event_retention ON events(received);
            CREATE TABLE IF NOT EXISTS media (
                id TEXT PRIMARY KEY, room TEXT NOT NULL, source TEXT NOT NULL,
                status TEXT NOT NULL, path TEXT, mime TEXT, width INTEGER,
                height INTEGER, bytes INTEGER, sha TEXT, created REAL NOT NULL,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS media_room ON media(room, created);
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY, room TEXT NOT NULL, anchor INTEGER NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, sent_parts INTEGER DEFAULT 0,
                UNIQUE(room,anchor)
            );
            CREATE TABLE IF NOT EXISTS room_state (room TEXT PRIMARY KEY, floor INTEGER NOT NULL);
        """)
        self._migrate(version)
        # A refresh cannot replay requests or downloads whose outcome is unknown.
        with self.db:
            self.db.execute("UPDATE media SET status='failed',error='interrupted' WHERE status='pending'")
            self.db.execute(
                "UPDATE generations SET status='uncertain',delivery=CASE WHEN sent_parts>0 THEN 'partial' ELSE 'uncertain' END WHERE status='running' OR (status='generated' AND pipeline_complete=0)"
            )

    def _migrate(self, version):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if version < 2:
                self.db.execute("ALTER TABLE events ADD COLUMN causal_anchor INTEGER")
                self.db.execute("ALTER TABLE generations ADD COLUMN delivery TEXT NOT NULL DEFAULT 'pending'")
                self.db.execute(
                    "ALTER TABLE generations ADD COLUMN pipeline_complete INTEGER NOT NULL DEFAULT 0"
                )
                self.db.execute(
                    "UPDATE events SET causal_anchor=(SELECT anchor FROM generations WHERE events.event_id='generation:'||generations.id AND events.room=generations.room) WHERE kind='self'"
                )
                self.db.execute(
                    "UPDATE generations SET status='generated',delivery='sent',pipeline_complete=1 WHERE status='sent'"
                )
            schema = """
                CREATE TABLE IF NOT EXISTS assets (
                    room TEXT NOT NULL, sha TEXT NOT NULL, path TEXT NOT NULL,
                    mime TEXT NOT NULL, width INTEGER, height INTEGER, bytes INTEGER NOT NULL,
                    created REAL NOT NULL, last_used REAL NOT NULL, PRIMARY KEY(room,sha)
                );
                CREATE TABLE IF NOT EXISTS deliveries (
                    generation TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
                    receipt TEXT NOT NULL, platform_id TEXT, created REAL NOT NULL, status TEXT NOT NULL DEFAULT 'sent',
                    PRIMARY KEY(generation,receipt)
                );
                CREATE TABLE IF NOT EXISTS frames (
                    event_seq INTEGER PRIMARY KEY REFERENCES events(seq) ON DELETE CASCADE,
                    version INTEGER NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY, room TEXT NOT NULL, anchor INTEGER NOT NULL,
                    phase TEXT NOT NULL, provider TEXT, input_tokens INTEGER,
                    cached_tokens INTEGER, output_tokens INTEGER, elapsed REAL,
                    images INTEGER, selected INTEGER, dropped INTEGER, rebases INTEGER,
                    status TEXT NOT NULL, created REAL NOT NULL
                );
            """
            for statement in schema.split(";"):
                if statement.strip():
                    self.db.execute(statement)
            if version < 2:
                for row in self.db.execute("SELECT * FROM media WHERE status='ready'").fetchall():
                    if row["path"] and Path(row["path"]).is_file():
                        self._asset(row)
                    else:
                        self.db.execute(
                            "UPDATE media SET status='failed',error='migration_file_missing' WHERE id=?",
                            (row["id"],),
                        )
            self.db.execute(
                "UPDATE media SET source='inline:'||COALESCE(sha,id) WHERE source LIKE 'data:%' OR source LIKE 'base64://%'"
            )
            if version < 3:
                self.db.execute("ALTER TABLE generations ADD COLUMN scope TEXT NOT NULL DEFAULT ''")
            for statement in """
                CREATE TABLE IF NOT EXISTS executions (
                    generation TEXT PRIMARY KEY REFERENCES generations(id) ON DELETE CASCADE,
                    scope TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turn_links (
                    room TEXT NOT NULL, anchor INTEGER NOT NULL REFERENCES events(seq) ON DELETE CASCADE,
                    source INTEGER NOT NULL REFERENCES events(seq) ON DELETE CASCADE,
                    PRIMARY KEY(room,anchor,source)
                );
                CREATE TABLE IF NOT EXISTS segments (
                    room TEXT NOT NULL, scope TEXT NOT NULL, payload TEXT NOT NULL,
                    updated REAL NOT NULL, PRIMARY KEY(room,scope)
                );
                CREATE TABLE IF NOT EXISTS diagnostics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT NOT NULL,
                    scope TEXT NOT NULL, phase TEXT NOT NULL, payload TEXT NOT NULL,
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS diagnostic_scope ON diagnostics(room,scope,phase,id);
                CREATE INDEX IF NOT EXISTS diagnostic_provider ON diagnostics(scope,phase,id);
            """.split(";"):
                if statement.strip():
                    self.db.execute(statement)
            if version < 3:
                # Earlier versions published generated protocol before transport ACK.
                # Keep it privately; only complete legacy sends have a recoverable reply.
                for row in self.db.execute(
                    "SELECT e.*,g.id AS gid,g.delivery FROM events e JOIN generations g ON e.event_id='generation:'||g.id AND e.room=g.room WHERE e.kind='self'"
                ).fetchall():
                    parts = json.loads(row["parts"])
                    protocol = [m for p in parts if p["type"] == "protocol" for m in p["messages"]]
                    self.db.execute(
                        "INSERT OR IGNORE INTO executions VALUES(?,?,?)",
                        (row["gid"], "legacy", json.dumps(protocol, ensure_ascii=False)),
                    )
                    public = []
                    if row["delivery"] == "sent":
                        # Only the final assistant text is knowable, never tool results.
                        final = next(
                            (
                                m
                                for m in reversed(protocol)
                                if m.get("role") == "assistant" and not m.get("tool_calls")
                            ),
                            {},
                        )
                        content = final.get("content", "")
                        public = (
                            [{"type": "text", "text": content}]
                            if isinstance(content, str) and content
                            else [p for p in content if p.get("type") == "text"]
                            if isinstance(content, list)
                            else []
                        )
                    self.db.execute(
                        "UPDATE events SET kind=?,parts=? WHERE seq=?",
                        (
                            "self" if public else "execution",
                            json.dumps(public, ensure_ascii=False),
                            row["seq"],
                        ),
                    )
                    self.db.execute("DELETE FROM frames WHERE event_seq=?", (row["seq"],))
            self.db.execute("PRAGMA user_version=3")

            self.db.commit()
            if version == 1:
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.db.execute("VACUUM")
        except BaseException:
            self.db.rollback()
            raise

    def _asset(self, row):
        self.db.execute(
            "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(room,sha) DO UPDATE SET last_used=MAX(last_used,excluded.last_used)",
            (
                row["room"],
                row["sha"],
                row["path"],
                row["mime"],
                row["width"],
                row["height"],
                row["bytes"],
                row["created"],
                row["created"],
            ),
        )

    def close(self):
        self.closed = True
        self.db.close()

    @staticmethod
    def _event(row) -> Event | None:
        if row is None:
            return None
        value = dict(row)
        value["parts"] = json.loads(value["parts"])
        return value

    def get(self, room: str, event_id: str) -> Event | None:
        return self._event(
            self.db.execute("SELECT * FROM events WHERE room=? AND event_id=?", (room, event_id)).fetchone()
        )

    def add(
        self,
        room: str,
        event_id: str,
        sender: str,
        name: str,
        parts: list[Part],
        *,
        kind="member",
        received=None,
    ):
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO events(room,event_id,sender,name,kind,received,parts) VALUES(?,?,?,?,?,?,?)",
                (
                    room,
                    str(event_id),
                    str(sender),
                    str(name),
                    kind,
                    time.time() if received is None else received,
                    json.dumps(parts, ensure_ascii=False),
                ),
            )
        return self.get(room, str(event_id)), bool(cur.rowcount)

    def floor(self, room):
        row = self.db.execute("SELECT floor FROM room_state WHERE room=?", (room,)).fetchone()
        return row[0] if row else 0

    def reset(self, room):
        seq = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM events WHERE room=?", (room,)).fetchone()[0]
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO room_state VALUES(?,?)", (room, seq))
            self.db.execute("DELETE FROM segments WHERE room=?", (room,))
            self.db.execute("DELETE FROM turn_links WHERE room=?", (room,))
        return seq

    def recent(self, room, snapshot, since, limit):
        rows = self.db.execute(
            "SELECT * FROM events WHERE room=? AND kind!='execution' AND seq>? AND seq<=? AND received>=? ORDER BY seq DESC LIMIT ?",
            (room, self.floor(room), snapshot, since, limit),
        ).fetchall()
        return [self._event(r) for r in reversed(rows)]

    def same_sender(self, anchor, seconds, limit=32, images_only=False):
        image_filter = (
            " AND EXISTS(SELECT 1 FROM json_tree(events.parts) WHERE key='media_id')" if images_only else ""
        )
        return [
            self._event(r)
            for r in self.db.execute(
                "SELECT * FROM events WHERE room=? AND sender=? AND seq>? AND seq<=? AND received>=?"
                + image_filter
                + " ORDER BY seq DESC LIMIT ?",
                (
                    anchor["room"],
                    anchor["sender"],
                    self.floor(anchor["room"]),
                    anchor["seq"],
                    anchor["received"] - seconds,
                    limit,
                ),
            ).fetchall()
        ]

    def member_identities(self, anchor, senders=None):
        from ..history import identities

        if senders is not None and not senders:
            return {}
        sender_clause = " AND sender IN (" + ",".join("?" for _ in senders) + ")" if senders else ""
        rows = self.db.execute(
            "SELECT sender,name,MAX(seq) AS latest FROM events "
            "WHERE room=? AND kind='member' AND seq>? AND seq<=? "
            "AND EXISTS(SELECT 1 FROM json_each(events.parts) WHERE json_extract(value,'$.type')!='poke') "
            + sender_clause
            + " GROUP BY sender,name ORDER BY latest DESC",
            (anchor["room"], self.floor(anchor["room"]), anchor["seq"], *(senders or [])),
        ).fetchall()
        return identities([dict(r) for r in rows])

    def group_history(self, anchor, **kwargs):
        from ..history import search

        rows = self.db.execute(
            "SELECT * FROM events WHERE room=? AND kind IN ('member','self') "
            "AND seq>? AND seq<? AND received<=? ORDER BY seq",
            (anchor["room"], self.floor(anchor["room"]), anchor["seq"], anchor["received"]),
        ).fetchall()
        return search(
            [self._event(r) for r in rows], self.member_identities(anchor), source="group", **kwargs
        )

    def new_media(self, room, source):
        mid = uuid.uuid4().hex
        with self.db:
            self.db.execute(
                "INSERT INTO media(id,room,source,status,created) VALUES(?,?,?,'pending',?)",
                (
                    mid,
                    room,
                    "inline:" + hashlib.sha256(source.encode()).hexdigest()
                    if source.startswith(("data:", "base64://"))
                    else "remote:"
                    if source.startswith(("http://", "https://"))
                    else "local",
                    time.time(),
                ),
            )
        return mid

    def media(self, mid, room=None):
        row = self.db.execute("SELECT * FROM media WHERE id=?", (mid,)).fetchone()
        if row is None or (room is not None and row["room"] != room):
            return None
        return dict(row)

    def media_update(self, mid, **fields):
        allowed = {"status", "path", "mime", "width", "height", "bytes", "sha", "error"}
        if not fields or not set(fields) <= allowed:
            raise ValueError("invalid media update")
        with self.db:
            self.db.execute(
                "UPDATE media SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?",
                (*fields.values(), mid),
            )
            if fields.get("status") == "ready":
                self._asset(self.media(mid))

    def view(self, anchor, since, limit):
        room = anchor["room"]
        watermark = self.db.execute(
            "SELECT COALESCE(MAX(seq),0) FROM events WHERE room=?", (room,)
        ).fetchone()[0]
        rows = self.db.execute(
            """SELECT * FROM events WHERE room=? AND kind!='execution' AND seq>? AND seq<=? AND received>=?
            AND (seq<=? OR (kind='self' AND causal_anchor<?))
            ORDER BY COALESCE(causal_anchor,seq) DESC, (causal_anchor IS NOT NULL) DESC,seq DESC LIMIT ?""",
            (room, self.floor(room), watermark, since, anchor["seq"], anchor["seq"], limit),
        ).fetchall()
        return [self._event(r) for r in reversed(rows)], watermark

    def frame(self, seq):
        row = self.db.execute(
            "SELECT payload FROM frames WHERE event_seq=? AND version=?", (seq, FRAME_VERSION)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def freeze(self, seq, payload):
        wire = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if "data:image/" in wire or "base64://" in wire:
            raise ValueError("Wire image payloads cannot be persisted")
        with self.db:
            self.db.execute(
                "INSERT INTO frames VALUES(?,?,?) ON CONFLICT(event_seq) DO UPDATE SET version=excluded.version,payload=excluded.payload WHERE frames.version!=excluded.version",
                (seq, FRAME_VERSION, wire),
            )
        return self.frame(seq)

    def asset_paths(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM assets ORDER BY created")]

    def asset_refs(self, path):
        return [
            r[0] for r in self.db.execute("SELECT id FROM media WHERE path=? AND status='ready'", (path,))
        ]

    def expire_asset(self, path):
        with self.db:
            self.db.execute("UPDATE media SET status='expired' WHERE path=? AND status='ready'", (path,))
            self.db.execute("DELETE FROM assets WHERE path=?", (path,))

    def trace(self, **fields):
        fields.setdefault("id", uuid.uuid4().hex)
        fields.setdefault("created", time.time())
        names = [
            "id",
            "room",
            "anchor",
            "phase",
            "provider",
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "elapsed",
            "images",
            "selected",
            "dropped",
            "rebases",
            "status",
            "created",
        ]
        with self.db:
            self.db.execute(
                "INSERT INTO traces VALUES(" + ",".join("?" for _ in names) + ")",
                [fields.get(n) for n in names],
            )

    def by_seq(self, room, seq):
        return self._event(
            self.db.execute(
                "SELECT * FROM events WHERE room=? AND kind!='execution' AND seq=? AND seq>?",
                (room, seq, self.floor(room)),
            ).fetchone()
        )

    def context_data(self, room, seqs, mids):
        return ({mid: self.media(mid, room) for mid in mids}, {seq: self.frame(seq) for seq in seqs})

    def freeze_many(self, frames):
        with self.db:
            for seq, payload in frames.items():
                wire = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                if "data:image/" in wire or "base64://" in wire:
                    raise ValueError("Wire image payloads cannot be persisted")
                self.db.execute(
                    "INSERT INTO frames VALUES(?,?,?) ON CONFLICT(event_seq) DO UPDATE SET version=excluded.version,payload=excluded.payload WHERE frames.version!=excluded.version",
                    (seq, FRAME_VERSION, wire),
                )
        return {seq: self.frame(seq) for seq in frames}

    def begin(self, room, anchor, scope=""):
        gid = uuid.uuid4().hex
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO generations(id,room,anchor,status,created,scope) VALUES(?,?,?,'running',?,?)",
                (gid, room, anchor, time.time(), scope),
            )
        return gid if cur.rowcount else None

    def finish(self, gid, status, messages=None, self_id=""):
        status = GenerationStatus(status)
        row = self.db.execute("SELECT * FROM generations WHERE id=?", (gid,)).fetchone()
        if not row or row["status"] != "running":
            return
        with self.db:
            self.db.execute("UPDATE generations SET status=? WHERE id=?", (status, gid))
            if messages is not None:
                wire = json.dumps(messages, ensure_ascii=False)
                if "data:image/" in wire or "base64://" in wire:
                    raise ValueError("Execution images must be symbolic")
                self.db.execute("INSERT OR REPLACE INTO executions VALUES(?,?,?)", (gid, row["scope"], wire))

    def execution(self, gid, scope):
        row = self.db.execute(
            "SELECT payload FROM executions WHERE generation=? AND scope=?", (gid, scope)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def link(self, room, anchor, sources):
        with self.db:
            for source in sources:
                if source != anchor:
                    self.db.execute("INSERT OR IGNORE INTO turn_links VALUES(?,?,?)", (room, anchor, source))

    def dependencies(self, room, anchor):
        return [
            r[0]
            for r in self.db.execute(
                "SELECT source FROM turn_links WHERE room=? AND anchor=? ORDER BY source", (room, anchor)
            )
        ]

    def segment(self, room, scope):
        row = self.db.execute(
            "SELECT payload FROM segments WHERE room=? AND scope=?", (room, scope)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_segment(self, room, scope, payload):
        wire = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if "data:image/" in wire or "base64://" in wire:
            raise ValueError("Segment images must be symbolic")
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO segments VALUES(?,?,?,?)", (room, scope, wire, time.time())
            )

    def diagnose(self, room, scope, phase, payload):
        previous = self.db.execute(
            "SELECT payload FROM diagnostics WHERE room=? AND scope=? AND phase=? ORDER BY id DESC LIMIT 1",
            (room, scope, phase),
        ).fetchone()
        old = json.loads(previous[0]) if previous else {}
        common = 0
        for before, after in zip(old.get("messages", []), payload.get("messages", [])):
            if before != after:
                break
            common += 1
        payload["common_prefix_messages"] = (
            common
            if old.get("system") == payload.get("system") and old.get("tools") == payload.get("tools")
            else 0
        )
        payload["first_changed_message"] = payload["common_prefix_messages"] if old else None
        payload["prefix_change"] = (
            "first_request"
            if not old
            else "system"
            if old.get("system") != payload.get("system")
            else "tools"
            if old.get("tools") != payload.get("tools")
            else "segment_rollover"
            if old.get("segment") != payload.get("segment")
            else "append"
            if common >= len(old.get("stable", []))
            else "history_or_attachment"
        )
        with self.db:
            self.db.execute(
                "INSERT INTO diagnostics(room,scope,phase,payload,created) VALUES(?,?,?,?,?)",
                (room, scope, phase, json.dumps(payload, sort_keys=True), time.time()),
            )
        return payload

    def budget_samples(self, scope):
        # Calibration is provider/model scoped, across rooms, and automatically
        # expires with diagnostics. Ignore retries/errors and tool continuations.
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT json_extract(payload,'$.estimate_version') AS estimate_version, "
                "json_extract(payload,'$.status') AS status, json_extract(payload,'$.stage') AS stage, "
                "json_extract(payload,'$.input_tokens') AS input_tokens, "
                "json_extract(payload,'$.raw_text_tokens') AS raw_text_tokens, "
                "json_extract(payload,'$.images') AS images, "
                "json_extract(payload,'$.raw_image_tokens') AS raw_image_tokens "
                "FROM diagnostics WHERE scope=? AND phase='wire_reply_first' "
                "AND json_extract(payload,'$.status')='completed' "
                "AND json_extract(payload,'$.input_tokens')>=512 "
                "ORDER BY id DESC LIMIT 96",
                (scope,),
            )
        ]

    def delivery_attempt(self, gid, receipt):
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO deliveries(generation,receipt,created,status) VALUES(?,?,?,'pending')",
                (gid, receipt, time.time()),
            )

    def delivery_failed(self, gid, receipt):
        with self.db:
            self.db.execute(
                "UPDATE deliveries SET status='failed' WHERE generation=? AND receipt=? AND status='pending'",
                (gid, receipt),
            )

    def sent(self, gid, receipt, platform_id=None, parts=None, self_id=""):
        existing = self.db.execute(
            "SELECT status FROM deliveries WHERE generation=? AND receipt=?", (gid, receipt)
        ).fetchone()
        if existing and existing[0] == "sent":
            return False
        with self.db:
            self.db.execute(
                "INSERT INTO deliveries(generation,receipt,platform_id,created,status) VALUES(?,?,?,?,'sent') ON CONFLICT(generation,receipt) DO UPDATE SET status='sent',platform_id=excluded.platform_id",
                (gid, receipt, platform_id, time.time()),
            )
            self.db.execute("UPDATE generations SET sent_parts=sent_parts+1 WHERE id=?", (gid,))
        if parts:
            self.publish(gid, receipt, parts, self_id, platform_id)
        return True

    def publish(self, gid, receipt, parts, self_id, platform_id=None):
        row = self.db.execute(
            "SELECT g.room,g.anchor FROM generations g JOIN deliveries d ON g.id=d.generation WHERE g.id=? AND d.receipt=? AND d.status='sent'",
            (gid, receipt),
        ).fetchone()
        if not row or not parts:
            return None
        eid = platform_id or "delivery:" + receipt
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO events(room,event_id,sender,name,kind,received,parts,causal_anchor) VALUES(?,?,?,?,?,?,?,?)",
                (
                    row["room"],
                    eid,
                    self_id,
                    "bot",
                    "self",
                    time.time(),
                    json.dumps(parts, ensure_ascii=False),
                    row["anchor"],
                ),
            )
        return self.get(row["room"], eid)

    def settle(self, gid, completed=False):
        with self.db:
            self.db.execute(
                """UPDATE generations SET delivery=CASE
                WHEN ? AND sent_parts>0 AND NOT EXISTS(SELECT 1 FROM deliveries WHERE generation=generations.id AND status!='sent') THEN 'sent'
                WHEN sent_parts>0 THEN 'partial'
                WHEN EXISTS(SELECT 1 FROM deliveries WHERE generation=generations.id AND status!='sent') THEN 'uncertain'
                WHEN ? THEN 'none' ELSE 'uncertain' END,
                pipeline_complete=? WHERE id=? AND status IN ('running','generated','failed')""",
                (completed, completed, completed, gid),
            )
            self.db.execute(
                "UPDATE generations SET status='failed' WHERE id=? AND status='running'",
                (gid,),
            )

    def status(self, room):
        row = self.db.execute(
            "SELECT status,delivery,sent_parts FROM generations WHERE room=? ORDER BY created DESC LIMIT 1",
            (room,),
        ).fetchone()
        last = dict(row) if row else None
        if last:
            last["generation_status"] = last["status"]
            if last["status"] == "generated" and last["delivery"] in {"sent", "partial"}:
                last["status"] = last["delivery"]
        return {
            "events": self.db.execute(
                "SELECT count(*) FROM events WHERE room=? AND kind!='execution' AND seq>?",
                (room, self.floor(room)),
            ).fetchone()[0],
            "media": {
                r[0]: r[1]
                for r in self.db.execute(
                    "SELECT status,count(*) FROM media WHERE room=? GROUP BY status", (room,)
                )
            },
            "last_generation": last,
            "diagnostics": [
                dict(r) | {"payload": json.loads(r["payload"])}
                for r in self.db.execute(
                    "SELECT phase,payload FROM diagnostics WHERE room=? AND phase NOT LIKE 'wire_%' ORDER BY id DESC LIMIT 3",
                    (room,),
                )
            ],
            "wire_diagnostics": [
                dict(r) | {"payload": json.loads(r["payload"])}
                for r in self.db.execute(
                    "SELECT phase,payload FROM diagnostics WHERE room=? AND phase LIKE 'wire_%' ORDER BY id DESC LIMIT 6",
                    (room,),
                )
            ],
            "segments": self.db.execute("SELECT count(*) FROM segments WHERE room=?", (room,)).fetchone()[0],
            "trace": [
                dict(r)
                for r in self.db.execute(
                    "SELECT phase,count(*) AS calls,COALESCE(SUM(input_tokens),0) AS input_tokens,COALESCE(SUM(cached_tokens),0) AS cached_tokens, COALESCE(SUM(input_tokens-cached_tokens),0) AS uncached_tokens, AVG(elapsed) AS avg_elapsed FROM traces WHERE room=? GROUP BY phase",
                    (room,),
                )
            ],
        }

    def prune_events(self, cutoff):
        with self.db:
            self.db.execute("DELETE FROM events WHERE received<?", (cutoff,))
            self.db.execute("DELETE FROM traces WHERE created<?", (cutoff,))
            self.db.execute("DELETE FROM diagnostics WHERE created<?", (cutoff,))
            self.db.execute("DELETE FROM segments WHERE updated<?", (cutoff,))
            self.db.execute("DELETE FROM generations WHERE created<?", (cutoff,))
            self.db.execute("DELETE FROM media WHERE status IN ('expired','failed') AND created<?", (cutoff,))


class Journal:
    """One SQLite connection owned by one worker; no SQL on the event loop.

    Submitted operations finish even if their awaiting task is cancelled. Close
    drains that worker before closing the connection, including during refresh.
    """

    def __init__(self, root: Path):
        self.closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-mm-sqlite")
        self._repository = self._executor.submit(SQLiteRepository, root)

    async def call(self, method, *args, **kwargs):
        if self.closed:
            raise RuntimeError("Journal is closed")
        future = self._executor.submit(lambda: getattr(self._repository.result(), method)(*args, **kwargs))
        return await asyncio.shield(asyncio.wrap_future(future))

    async def ready(self):
        await asyncio.shield(asyncio.wrap_future(self._repository))

    async def close(self):
        if self.closed:
            return
        self.closed = True
        future = self._executor.submit(lambda: self._repository.result().close())
        try:
            await asyncio.shield(asyncio.wrap_future(future))
        finally:
            await asyncio.to_thread(self._executor.shutdown, wait=True)

    async def get(self, room, event_id):
        return await self.call("get", room, event_id)

    async def add(self, room, event_id, sender, name, parts, **kwargs):
        return await self.call("add", room, event_id, sender, name, parts, **kwargs)

    async def floor(self, room):
        return await self.call("floor", room)

    async def reset(self, room):
        return await self.call("reset", room)

    async def recent(self, room, snapshot, since, limit):
        return await self.call("recent", room, snapshot, since, limit)

    async def view(self, anchor, since, limit):
        return await self.call("view", anchor, since, limit)

    async def same_sender(self, anchor, seconds, **kwargs):
        return await self.call("same_sender", anchor, seconds, **kwargs)

    async def member_identities(self, anchor, senders=None):
        return await self.call("member_identities", anchor, senders)

    async def group_history(self, anchor, **kwargs):
        return await self.call("group_history", anchor, **kwargs)

    async def new_media(self, room, source):
        return await self.call("new_media", room, source)

    async def media(self, mid, room=None):
        return await self.call("media", mid, room)

    async def media_update(self, mid, **fields):
        return await self.call("media_update", mid, **fields)

    async def begin(self, room, anchor, scope=""):
        return await self.call("begin", room, anchor, scope)

    async def finish(self, gid, status, messages=None, self_id=""):
        return await self.call("finish", gid, status, messages, self_id)

    async def sent(self, gid, receipt, platform_id=None, parts=None, self_id=""):
        return await self.call("sent", gid, receipt, platform_id, parts, self_id)

    async def settle(self, gid, completed=False):
        return await self.call("settle", gid, completed)

    async def status(self, room):
        return await self.call("status", room)

    async def prune_events(self, cutoff):
        return await self.call("prune_events", cutoff)

    async def frame(self, seq):
        return await self.call("frame", seq)

    async def freeze(self, seq, payload):
        return await self.call("freeze", seq, payload)

    async def asset_paths(self):
        return await self.call("asset_paths")

    async def asset_refs(self, path):
        return await self.call("asset_refs", path)

    async def expire_asset(self, path):
        return await self.call("expire_asset", path)

    async def trace(self, **fields):
        return await self.call("trace", **fields)

    async def by_seq(self, room, seq):
        return await self.call("by_seq", room, seq)

    async def context_data(self, room, seqs, mids):
        return await self.call("context_data", room, seqs, mids)

    async def freeze_many(self, frames):
        return await self.call("freeze_many", frames)

    async def delivery_attempt(self, gid, receipt):
        return await self.call("delivery_attempt", gid, receipt)

    async def delivery_failed(self, gid, receipt):
        return await self.call("delivery_failed", gid, receipt)

    async def execution(self, gid, scope=""):
        return await self.call("execution", gid, scope)

    async def link(self, room, anchor, sources):
        return await self.call("link", room, anchor, sources)

    async def dependencies(self, room, anchor):
        return await self.call("dependencies", room, anchor)

    async def segment(self, room, scope):
        return await self.call("segment", room, scope)

    async def save_segment(self, room, scope, payload):
        return await self.call("save_segment", room, scope, payload)

    async def diagnose(self, room, scope, phase, payload):
        return await self.call("diagnose", room, scope, phase, payload)

    async def budget_samples(self, scope):
        return await self.call("budget_samples", scope)

    async def publish(self, gid, receipt, parts, self_id, platform_id=None):
        return await self.call("publish", gid, receipt, parts, self_id, platform_id)
