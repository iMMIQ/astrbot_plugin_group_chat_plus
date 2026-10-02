"""AstrBot entry point: only event routing lives here."""

from __future__ import annotations

import contextlib
import sys

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .multimodal.adapters.astrbot import AstrBotGateway
from .multimodal.config import Settings
from .multimodal.conversation import OWNER, ConversationService

__all__ = ["ChatPlus", "OWNER"]


class ChatPlus(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = Settings(config)
        self.service = ConversationService(AstrBotGateway(context, self.config), self.config)

    async def initialize(self):
        await self.service.initialize()

    async def terminate(self):
        await self.service.terminate()

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=sys.maxsize + 10)
    async def capture(self, event: AstrMessageEvent):
        async with contextlib.aclosing(self.service.capture(event)) as flow:
            async for result in flow:
                yield result

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-sys.maxsize)
    async def respond(self, event: AstrMessageEvent):
        async with contextlib.aclosing(self.service.respond(event)) as flow:
            async for result in flow:
                yield result

    @filter.on_llm_request(priority=sys.maxsize)
    async def snapshot(self, event, req):
        await self.service.snapshot(event, req)

    @filter.on_llm_request(priority=-sys.maxsize)
    async def inject(self, event, req):
        await self.service.inject(event, req)

    @filter.on_agent_begin(priority=-sys.maxsize)
    async def agent_begin(self, event, run_context):
        await self.service.agent_begin(event, run_context)

    @filter.on_agent_done(priority=-sys.maxsize)
    async def agent_done(self, event, run_context, response):
        await self.service.agent_done(event, run_context, response)

    @filter.after_message_sent(priority=-sys.maxsize)
    async def delivered(self, event):
        await self.service.delivered(event)

    @filter.command("mmstatus")
    async def status(self, event):
        async for result in self.service.status(event):
            yield result

    @filter.command("mmreset")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def reset(self, event):
        async for result in self.service.reset(event):
            yield result
