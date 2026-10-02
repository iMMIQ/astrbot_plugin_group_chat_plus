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
        self.tasks = {}
        self.leases = Counter()
        self.semaphore = asyncio.Semaphore(4)
        self.write_lock = asyncio.Lock()
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))

    async def capture(self, room, source):
        mid = await self.journal.new_media(room, source)
        if len(self.tasks) >= int(self.config.get("max_pending_media", 128)):
            await self.journal.media_update(mid, status="failed", error="download_queue_full")
            return mid
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
            data = await asyncio.to_thread(base64.b64decode, encoded, validate=True)
        else:
            path = Path(unquote(urlparse(source).path)) if source.startswith("file://") else Path(source)
            if await asyncio.to_thread(lambda: path.stat().st_size) > limit:
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
                sha = await asyncio.to_thread(lambda: hashlib.sha256(data).hexdigest())
                room = (await self.journal.media(mid))["room"]
                bucket = self.root / hashlib.sha256(room.encode()).hexdigest()[:24]
                await asyncio.to_thread(bucket.mkdir, parents=True, exist_ok=True)
                path = bucket / sha
                async with self.write_lock:
                    if not await asyncio.to_thread(path.exists):
                        if not await self._make_space(len(data), room):
                            raise ValueError("media_quota_exceeded")
                        tmp = bucket / (mid + ".tmp")
                        await asyncio.to_thread(tmp.write_bytes, data)
                        await asyncio.to_thread(tmp.replace, path)
                    await self.journal.media_update(
                        mid,
                        status="ready",
                        path=str(path),
                        mime=mime,
                        width=width,
                        height=height,
                        bytes=len(data),
                        sha=sha,
                        error=None,
                    )
        except asyncio.CancelledError:
            await self.journal.media_update(mid, status="failed", error="interrupted")
            raise
        except Exception as exc:
            # Never put a remote URL, token or local exception detail in diagnostics.
            reason = (
                str(exc)
                if isinstance(exc, ValueError)
                and str(exc) in {"image_too_large", "unsupported_image", "media_quota_exceeded"}
                else type(exc).__name__
            )
            await self.journal.media_update(mid, status="failed", error=reason)

    async def _expire_path(self, path):
        mids = await self.journal.asset_refs(path)
        if any(self.leases[mid] for mid in mids):
            return False
        try:
            await asyncio.to_thread(Path(path).unlink, missing_ok=True)
        except OSError:
            return False
        await self.journal.expire_asset(path)
        return True

    async def _make_space(self, incoming, room):
        rows = await self.journal.asset_paths()
        total = sum(r["bytes"] for r in rows)
        room_total = sum(r["bytes"] for r in rows if r["room"] == room)
        hard = int(self.config.get("total_media_mib", 1024)) * 1024 * 1024
        soft = int(self.config.get("room_media_mib", 256)) * 1024 * 1024
        for row in rows:
            if total + incoming <= hard and (room_total + incoming <= soft or row["room"] != room):
                continue
            if await self._expire_path(row["path"]):
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
        media = await self.journal.media(mid, room)
        if not media or media["status"] != "ready":
            return None
        try:
            data = await asyncio.to_thread(Path(media["path"]).read_bytes)
        except OSError:
            await self.journal.expire_asset(media["path"])
            return None
        encoded = await asyncio.to_thread(lambda: base64.b64encode(data).decode())
        return f"data:{media['mime']};base64," + encoded

    async def cleanup(self):
        cutoff = time.time() - float(self.config.get("media_retention_hours", 24)) * 3600
        async with self.write_lock:
            for asset in await self.journal.asset_paths():
                if asset["last_used"] < cutoff:
                    await self._expire_path(asset["path"])
        await self.journal.prune_events(
            time.time() - float(self.config.get("journal_retention_days", 7)) * 86400
        )

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.session:
            await self.session.close()
