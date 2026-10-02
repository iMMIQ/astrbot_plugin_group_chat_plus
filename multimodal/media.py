"""Capture original image bytes; leases keep cleanup out of active requests."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp
from PIL import Image


class MediaStore:
    def __init__(self, journal, root, config):
        self.journal, self.root, self.config = journal, Path(root), config
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks = {}
        self.leases = Counter()
        self.semaphore = asyncio.Semaphore(4)
        self.write_lock = asyncio.Lock()
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))

    def capture(self, room, source):
        mid = self.journal.new_media(room, source)
        task = asyncio.create_task(self._capture(mid, source))
        self.tasks[mid] = task
        task.add_done_callback(lambda _: self.tasks.pop(mid, None))
        return mid

    async def _read(self, source):
        limit = int(self.config.get("max_image_mib", 10)) * 1024 * 1024
        if source.startswith(("http://", "https://")):
            async with self.session.get(source) as response:
                response.raise_for_status()
                if response.content_length and response.content_length > limit:
                    raise ValueError("image_too_large")
                data = bytearray()
                async for block in response.content.iter_chunked(65536):
                    data.extend(block)
                    if len(data) > limit:
                        raise ValueError("image_too_large")
                return bytes(data)
        if source.startswith(("data:image/", "base64://")):
            encoded = source.split(",", 1)[1] if source.startswith("data:") else source[9:]
            if len(encoded) > (limit + 2) // 3 * 4:
                raise ValueError("image_too_large")
            data = base64.b64decode(encoded, validate=True)
        else:
            path = Path(unquote(urlparse(source).path)) if source.startswith("file://") else Path(source)
            if path.stat().st_size > limit:
                raise ValueError("image_too_large")
            data = await asyncio.to_thread(path.read_bytes)
        if len(data) > limit:
            raise ValueError("image_too_large")
        return data

    async def _capture(self, mid, source):
        try:
            async with self.semaphore:
                data = await self._read(source)
                def inspect():
                    with Image.open(io.BytesIO(data)) as image:
                        size, fmt = image.size, image.format
                        image.verify()
                    if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
                        raise ValueError("unsupported_image")
                    return size, Image.MIME[fmt]
                (width, height), mime = await asyncio.to_thread(inspect)
                sha = hashlib.sha256(data).hexdigest()
                room = self.journal.media(mid)["room"]
                bucket = self.root / hashlib.sha256(room.encode()).hexdigest()[:24]
                bucket.mkdir(exist_ok=True)
                path = bucket / sha
                async with self.write_lock:
                    if not path.exists():
                        if not self._make_space(len(data), room):
                            raise ValueError("media_quota_exceeded")
                        tmp = bucket / (mid + ".tmp")
                        await asyncio.to_thread(tmp.write_bytes, data)
                        tmp.replace(path)
                    self.journal.media_update(mid, status="ready", path=str(path), mime=mime,
                                              width=width, height=height, bytes=len(data), sha=sha, error=None)
        except asyncio.CancelledError:
            self.journal.media_update(mid, status="failed", error="interrupted")
            raise
        except Exception as exc:
            # Never put a remote URL, token or local exception detail in diagnostics.
            reason = str(exc) if isinstance(exc, ValueError) and str(exc) in {
                "image_too_large", "unsupported_image", "media_quota_exceeded"
            } else type(exc).__name__
            self.journal.media_update(mid, status="failed", error=reason)

    def _expire_path(self, path):
        rows = self.journal.db.execute("SELECT id FROM media WHERE path=? AND status='ready'", (path,)).fetchall()
        if any(self.leases[r[0]] for r in rows):
            return False
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            return False
        with self.journal.db:
            self.journal.db.execute("UPDATE media SET status='expired' WHERE path=? AND status='ready'", (path,))
        return True

    def _make_space(self, incoming, room):
        rows = self.journal.db.execute(
            "SELECT path,room,MAX(bytes) AS bytes,MIN(created) AS created FROM media WHERE status='ready' GROUP BY path,room ORDER BY created"
        ).fetchall()
        total = sum(r["bytes"] for r in rows)
        room_total = sum(r["bytes"] for r in rows if r["room"] == room)
        hard = int(self.config.get("total_media_mib", 1024)) * 1024 * 1024
        soft = int(self.config.get("room_media_mib", 256)) * 1024 * 1024
        for row in rows:
            if total + incoming <= hard and (room_total + incoming <= soft or row["room"] != room):
                continue
            if self._expire_path(row["path"]):
                total -= row["bytes"]
                if row["room"] == room:
                    room_total -= row["bytes"]
        return total + incoming <= hard

    async def wait(self, mids):
        tasks = [self.tasks[mid] for mid in mids if mid in self.tasks]
        if tasks:
            await asyncio.wait(tasks, timeout=float(self.config.get("media_wait_seconds", 3)))

    def lease(self, mids):
        self.leases.update(mids)

    def release(self, mids):
        for mid in mids:
            self.leases[mid] -= 1
            if self.leases[mid] <= 0:
                self.leases.pop(mid, None)

    async def data_uri(self, mid, room):
        media = self.journal.media(mid, room)
        if not media or media["status"] != "ready":
            return None
        try:
            data = await asyncio.to_thread(Path(media["path"]).read_bytes)
        except OSError:
            self.journal.media_update(mid, status="expired", error="file_missing")
            return None
        return f"data:{media['mime']};base64," + base64.b64encode(data).decode()

    def cleanup(self):
        cutoff = time.time() - float(self.config.get("media_retention_hours", 24)) * 3600
        rows = self.journal.db.execute("SELECT path FROM media WHERE status='ready' GROUP BY path HAVING MAX(created)<?", (cutoff,)).fetchall()
        for row in rows:
            self._expire_path(row[0])
        self.journal.prune_events(time.time() - float(self.config.get("journal_retention_days", 7)) * 86400)

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.session:
            await self.session.close()
