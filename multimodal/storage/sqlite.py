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

from ..models import Event, GenerationStatus, Part


class SQLiteRepository:
    def __init__(self, root: Path):
        self.closed = False
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "events.sqlite3", timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 2:
            raise RuntimeError("Unsupported journal schema; use the matching plugin version")
        if version == 1:
            backup = root / "events.pre-v2.sqlite3"
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
            self.db.execute("PRAGMA user_version=2")

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
        return seq

    def recent(self, room, snapshot, since, limit):
        rows = self.db.execute(
            "SELECT * FROM events WHERE room=? AND seq>? AND seq<=? AND received>=? ORDER BY seq DESC LIMIT ?",
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
            """SELECT * FROM events WHERE room=? AND seq>? AND seq<=? AND received>=?
            AND (seq<=? OR (kind='self' AND causal_anchor<?))
            ORDER BY COALESCE(causal_anchor,seq) DESC, (causal_anchor IS NOT NULL) DESC,seq DESC LIMIT ?""",
            (room, self.floor(room), watermark, since, anchor["seq"], anchor["seq"], limit),
        ).fetchall()
        return [self._event(r) for r in reversed(rows)], watermark

    def frame(self, seq):
        row = self.db.execute("SELECT payload FROM frames WHERE event_seq=? AND version=1", (seq,)).fetchone()
        return json.loads(row[0]) if row else None

    def freeze(self, seq, payload):
        wire = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if "data:image/" in wire or "base64://" in wire:
            raise ValueError("Wire image payloads cannot be persisted")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO frames VALUES(?,1,?)", (seq, wire))
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
                "SELECT * FROM events WHERE room=? AND seq=? AND seq>?", (room, seq, self.floor(room))
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
                self.db.execute("INSERT OR IGNORE INTO frames VALUES(?,1,?)", (seq, wire))
        return {seq: self.frame(seq) for seq in frames}

    def begin(self, room, anchor):
        gid = uuid.uuid4().hex
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO generations(id,room,anchor,status,created) VALUES(?,?,?,'running',?)",
                (gid, room, anchor, time.time()),
            )
        return gid if cur.rowcount else None

    def finish(self, gid, status, messages=None, self_id=""):
        status = GenerationStatus(status)
        row = self.db.execute("SELECT * FROM generations WHERE id=?", (gid,)).fetchone()
        if not row or row["status"] != "running":
            return
        with self.db:
            self.db.execute("UPDATE generations SET status=? WHERE id=?", (status, gid))
            if messages and status in {"generated", "failed"}:
                self.db.execute(
                    "INSERT OR IGNORE INTO events(room,event_id,sender,name,kind,received,parts,causal_anchor) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        row["room"],
                        "generation:" + gid,
                        self_id,
                        "bot",
                        "self",
                        time.time(),
                        json.dumps([{"type": "protocol", "messages": messages}], ensure_ascii=False),
                        row["anchor"],
                    ),
                )

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

    def sent(self, gid, receipt, platform_id=None):
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
        return True

    def settle(self, gid, completed=False):
        with self.db:
            self.db.execute(
                """UPDATE generations SET delivery=CASE
                WHEN ? AND sent_parts>0 AND NOT EXISTS(SELECT 1 FROM deliveries WHERE generation=generations.id AND status!='sent') THEN 'sent'
                WHEN sent_parts>0 THEN 'partial'
                WHEN EXISTS(SELECT 1 FROM deliveries WHERE generation=generations.id AND status!='sent') THEN 'uncertain'
                WHEN ? THEN 'none' ELSE 'uncertain' END,
                pipeline_complete=? WHERE id=? AND status='generated'""",
                (completed, completed, completed, gid),
            )
            self.db.execute(
                "UPDATE generations SET status='failed',delivery='uncertain' WHERE id=? AND status='running'",
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
                "SELECT count(*) FROM events WHERE room=? AND seq>?", (room, self.floor(room))
            ).fetchone()[0],
            "media": {
                r[0]: r[1]
                for r in self.db.execute(
                    "SELECT status,count(*) FROM media WHERE room=? GROUP BY status", (room,)
                )
            },
            "last_generation": last,
            "trace": [
                dict(r)
                for r in self.db.execute(
                    "SELECT phase,count(*) AS calls,COALESCE(SUM(input_tokens),0) AS input_tokens,COALESCE(SUM(cached_tokens),0) AS cached_tokens FROM traces WHERE room=? GROUP BY phase",
                    (room,),
                )
            ],
        }

    def prune_events(self, cutoff):
        with self.db:
            self.db.execute("DELETE FROM events WHERE received<?", (cutoff,))
            self.db.execute("DELETE FROM traces WHERE created<?", (cutoff,))
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

    async def new_media(self, room, source):
        return await self.call("new_media", room, source)

    async def media(self, mid, room=None):
        return await self.call("media", mid, room)

    async def media_update(self, mid, **fields):
        return await self.call("media_update", mid, **fields)

    async def begin(self, room, anchor):
        return await self.call("begin", room, anchor)

    async def finish(self, gid, status, messages=None, self_id=""):
        return await self.call("finish", gid, status, messages, self_id)

    async def sent(self, gid, receipt, platform_id=None):
        return await self.call("sent", gid, receipt, platform_id)

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
