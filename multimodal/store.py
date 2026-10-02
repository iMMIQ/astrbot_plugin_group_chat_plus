"""Versioned room journal. SQL never uses sender-only or naked group scopes."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


class Journal:
    def __init__(self, root: Path):
        self.closed = False
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "events.sqlite3", timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
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
            PRAGMA user_version=1;
        """)
        # A refresh cannot replay requests or downloads whose outcome is unknown.
        with self.db:
            self.db.execute("UPDATE media SET status='failed',error='interrupted' WHERE status='pending'")
            self.db.execute("UPDATE generations SET status='uncertain' WHERE status IN ('running','generated')")

    def close(self):
        self.closed = True
        self.db.close()

    @staticmethod
    def _event(row):
        if row is None:
            return None
        value = dict(row)
        value["parts"] = json.loads(value["parts"])
        return value

    def get(self, room: str, event_id: str):
        return self._event(self.db.execute(
            "SELECT * FROM events WHERE room=? AND event_id=?", (room, event_id)
        ).fetchone())

    def add(self, room, event_id, sender, name, parts, *, kind="member", received=None):
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO events(room,event_id,sender,name,kind,received,parts) VALUES(?,?,?,?,?,?,?)",
                (room, str(event_id), str(sender), str(name), kind,
                 time.time() if received is None else received, json.dumps(parts, ensure_ascii=False)),
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
        image_filter = " AND EXISTS(SELECT 1 FROM json_tree(events.parts) WHERE key='media_id')" if images_only else ""
        return [self._event(r) for r in self.db.execute(
            "SELECT * FROM events WHERE room=? AND sender=? AND seq>? AND seq<=? AND received>=?" + image_filter + " ORDER BY seq DESC LIMIT ?",
            (anchor["room"], anchor["sender"], self.floor(anchor["room"]), anchor["seq"],
             anchor["received"] - seconds, limit),
        ).fetchall()]

    def new_media(self, room, source):
        mid = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO media(id,room,source,status,created) VALUES(?,?,?,'pending',?)",
                            (mid, room, source, time.time()))
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
            self.db.execute("UPDATE media SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?",
                            (*fields.values(), mid))

    def begin(self, room, anchor):
        gid = uuid.uuid4().hex
        with self.db:
            cur = self.db.execute("INSERT OR IGNORE INTO generations(id,room,anchor,status,created) VALUES(?,?,?,'running',?)",
                                  (gid, room, anchor, time.time()))
        return gid if cur.rowcount else None

    def finish(self, gid, status, messages=None, self_id=""):
        row = self.db.execute("SELECT * FROM generations WHERE id=?", (gid,)).fetchone()
        if not row or row["status"] != "running":
            return
        with self.db:
            self.db.execute("UPDATE generations SET status=? WHERE id=?", (status, gid))
            if messages and status in {"generated", "failed"}:
                self.db.execute(
                    "INSERT OR IGNORE INTO events(room,event_id,sender,name,kind,received,parts) VALUES(?,?,?,?,?,?,?)",
                    (row["room"], "generation:" + gid, self_id, "bot", "self", time.time(),
                     json.dumps([{"type": "protocol", "messages": messages}], ensure_ascii=False)),
                )

    def sent(self, gid):
        with self.db:
            self.db.execute("UPDATE generations SET sent_parts=sent_parts+1 WHERE id=?", (gid,))

    def settle(self, gid):
        if self.closed:
            return
        with self.db:
            self.db.execute("UPDATE generations SET status=CASE WHEN sent_parts>0 THEN 'sent' ELSE 'uncertain' END WHERE id=? AND status='generated'", (gid,))
            self.db.execute("UPDATE generations SET status='failed' WHERE id=? AND status='running'", (gid,))

    def status(self, room):
        return {
            "events": self.db.execute("SELECT count(*) FROM events WHERE room=? AND seq>?", (room, self.floor(room))).fetchone()[0],
            "media": {r[0]: r[1] for r in self.db.execute("SELECT status,count(*) FROM media WHERE room=? GROUP BY status", (room,))},
            "last_generation": dict(r) if (r := self.db.execute("SELECT status,sent_parts FROM generations WHERE room=? ORDER BY created DESC LIMIT 1", (room,)).fetchone()) else None,
        }

    def prune_events(self, cutoff):
        with self.db:
            self.db.execute("DELETE FROM events WHERE received<?", (cutoff,))
            self.db.execute("DELETE FROM generations WHERE created<?", (cutoff,))
            self.db.execute("DELETE FROM media WHERE status IN ('expired','failed') AND created<?", (cutoff,))
