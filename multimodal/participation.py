"""Participation policy, independent of framework hooks and persistence."""

import json
import random
import time


class Participation:
    def __init__(self, config):
        self.config = config
        self.last_check = {}

    def candidate(self, room):
        return (
            self.config["auto_reply_enabled"]
            and self.ready(room)
            and random.random() < self.config["auto_candidate_probability"]
        )

    def ready(self, room):
        return time.monotonic() - self.last_check.get(room, -1e9) >= self.config["auto_reply_cooldown"]

    def claim(self, room):
        if not self.ready(room):
            return False
        self.last_check[room] = time.monotonic()
        return True

    @staticmethod
    def accepts(text):
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return False
        return isinstance(value, dict) and type(value.get("reply")) is bool and value["reply"]

    def cleanup(self):
        cutoff = time.monotonic() - max(3600, self.config["auto_reply_cooldown"])
        self.last_check = {room: t for room, t in self.last_check.items() if t >= cutoff}
