"""Run inside AstrBot v4.28.2: real builder/runner/hooks, offline provider.

Usage: python tests/multimodal/sdk_probe.py /path/to/plugin
Does not load production configuration, contact providers or send platform messages.
"""
import asyncio
import base64
import copy
import importlib
import json
import pathlib
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from mcp.types import CallToolResult, TextContent, ImageContent

from astrbot.api.event import filter
from astrbot.api.star import Context
from astrbot.core.agent.tool import ToolSet, FunctionTool
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.config.agent_runner import normalize_agent_runner
from astrbot.core.astr_main_agent import MainAgentBuildConfig, build_main_agent, collect_initial_request
from astrbot.core.message.components import At, Image, Plain
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.provider.entities import LLMResponse, TokenUsage
from astrbot.core.provider.provider import Provider
from astrbot.core.star.star import StarMetadata, star_map
from astrbot.core.star.star_handler import EventType, star_handlers_registry

root = pathlib.Path(sys.argv[1]).resolve()
package = types.ModuleType('native_mm_probe')
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
module = importlib.import_module('native_mm_probe.main')
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC')


def event(eid, parts, sender='a', group='fixture-group'):
    msg = AstrBotMessage()
    msg.type = MessageType.GROUP_MESSAGE
    msg.self_id, msg.group_id, msg.message_id = 'fixture-bot', group, eid
    msg.sender = MessageMember(user_id=sender, nickname=sender)
    msg.message = parts
    msg.message_str = ''.join(p.text for p in parts if isinstance(p, Plain))
    meta = SimpleNamespace(id='fixture-platform', name='aiocqhttp', support_proactive_message=False,
                           support_streaming_message=False)
    ev = AstrMessageEvent(msg.message_str, msg, meta, sender+'_'+group)
    ev.send = AsyncMock()
    ev.is_at_or_wake_command = any(isinstance(p, At) for p in parts)
    return ev


async def record(plugin, ev):
    assert [result async for result in plugin.capture(ev)] == []


async def main():
    captures=[]
    tool_mode={'enabled':False,'step':0}
    provider=Mock(spec=Provider)
    provider.provider_config={'id':'fixture-provider','modalities':['text','image','tool_use'],
                              'max_context_tokens':128000}
    provider.get_model.return_value='fixture-model'
    provider.meta.return_value=SimpleNamespace(type='openai_chat_completion', id='fixture-provider')
    async def reply(**kwargs):
        captures.append(kwargs)
        if tool_mode['enabled'] and tool_mode['step']==0:
            tool_mode['step']+=1
            return LLMResponse(role='assistant',completion_text='',tools_call_name=['fixture_lookup'],
                               tools_call_args=[{}],tools_call_ids=['call-fixture'],
                               usage=TokenUsage(input_other=64,input_cached=128,output=8))
        return LLMResponse(role='assistant', completion_text='fixture_reply',
                           usage=TokenUsage(input_other=64,input_cached=128,output=8))
    provider.text_chat=AsyncMock(side_effect=reply)
    cfg=copy.deepcopy(DEFAULT_CONFIG)
    cfg['agent_runner']=normalize_agent_runner(cfg.get('agent_runner'))
    cfg['provider_ltm_settings']['group_message_history_enable']=False
    cfg['provider_settings']['proactive_capability']={'add_cron_tools':False}
    cfg['subagent_orchestrator']={}
    cfg['timezone']='UTC'
    cfg['provider_settings']['enable']=True
    context=Mock(spec=Context)
    context.get_config.return_value=cfg
    context.get_using_provider_async=AsyncMock(return_value=provider)
    context.get_provider_by_id.return_value=provider
    context.subagent_orchestrator=None
    context.kb_manager=SimpleNamespace()
    context.persona_manager=SimpleNamespace(resolve_selected_persona=AsyncMock(return_value=(
        'fixture-persona', {'prompt':'PERSONA_MUST_SURVIVE','tools':[], 'skills':[]}, None, False)))
    context.get_llm_tool_manager.return_value=SimpleNamespace(get_full_tool_set=lambda:ToolSet())
    conversation=SimpleNamespace(history=json.dumps([{'role':'user','content':'OLD_CORE_HISTORY'}]),
                                 cid='fixture-conversation',persona_id='fixture-persona',token_usage=0)
    context.conversation_manager=SimpleNamespace(get_curr_conversation_id=AsyncMock(return_value=conversation.cid),
                                                 get_conversation=AsyncMock(return_value=conversation),update_conversation=AsyncMock())
    results={}
    with (tempfile.TemporaryDirectory() as td,
          patch.object(module.StarTools,'get_data_dir',return_value=pathlib.Path(td)),
          patch('astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal.db_helper.insert_provider_stat',
                new=AsyncMock()) as stat_sink):
        plugin=module.ChatPlus(context,{'input_token_budget':32768,'mention_wait_seconds':0.05})
        star_map[module.__name__]=StarMetadata(name='native-mm-probe',activated=True)
        for handler in star_handlers_registry:
            if handler.handler_module_path==module.__name__:
                handler.handler=getattr(plugin,handler.handler.__name__)
        await plugin.initialize()
        try:
            source=event('image',[Image.fromBytes(PNG)])
            await record(plugin,source)
            initial=source.get_extra(module.OWNER+'_anchor')
            with plugin.journal.db:
                plugin.journal.db.execute('UPDATE events SET received=received-21 WHERE seq=?',(initial['seq'],))
            interrupt=event('interruption',[Plain('B 插话')],sender='b')
            await record(plugin,interrupt)
            target=event('question',[At(qq='fixture-bot'),Plain('看前一张图')])
            await record(plugin,target)
            flow=plugin.respond(target)
            with patch('astrbot.core.astr_main_agent.retrieve_knowledge_base',new=AsyncMock(return_value='KB_MUST_SURVIVE')) as kb:
                await anext(flow)
                assert kb.call_args.kwargs['query']=='看前一张图'
            state=target.get_extra(module.OWNER)
            assert 'OLD_CORE_HISTORY' in json.dumps(state['snapshot']['contexts'])
            actual=state['request']
            assert actual.prompt=='' and 'PERSONA_MUST_SURVIVE' in actual.system_prompt
            assert 'KB_MUST_SURVIVE' in json.dumps(actual.extra_user_content_parts)
            assert 'OLD_CORE_HISTORY' not in json.dumps(actual.contexts)
            assert len(captures)==1
            body=json.dumps(captures[0],default=str,ensure_ascii=False)
            assert body.count('data:image/png;base64,')==1
            assert 'PERSONA_MUST_SURVIVE' in body and 'KB_MUST_SURVIVE' in body
            assert '"sender_id": "a"' in body or '\\"sender_id\\": \\"a\\"' in body
            assert target.send.await_count==0
            # on_agent_done used real SDK Message objects and generated our journal turn.
            assert plugin.journal.status(initial['room'])['last_generation']['status']=='generated'
            target.set_result(target.plain_result('fixture_reply').set_result_content_type(ResultContentType.LLM_RESULT))
            await call_event_hook(target,EventType.OnAfterMessageSentEvent)
            try: await anext(flow)
            except StopAsyncIteration: pass
            assert 'data:image' not in json.dumps(context.conversation_manager.update_conversation.call_args.kwargs['history'])
            assert plugin.journal.status(initial['room'])['last_generation']['status']=='sent'
            results['actual_builder_history_reload_and_final_image_count']=1
            results['persona_and_kb_preserved']=True
            results['agent_completion_persisted_and_mirror_has_no_base64']=True

            # Per-room lock survives the yield boundary until the framework finishes.
            one=event('one',[At(qq='fixture-bot'),Plain('one')]); two=event('two',[At(qq='fixture-bot'),Plain('two')],sender='b')
            await record(plugin,one);await record(plugin,two)
            first=plugin.respond(one);second=plugin.respond(two)
            await anext(first)
            pending=asyncio.create_task(anext(second))
            await asyncio.sleep(0.02);assert not pending.done()
            await first.aclose();await asyncio.wait_for(pending,1);await second.aclose()
            results['room_lock_serializes_different_senders']=True

            # A pure @ may collect A's next line, but never takes B's line as A's.
            mention=event('mention',[At(qq='fixture-bot')]); pure=plugin.capture(mention)
            pending=asyncio.create_task(anext(pure));await asyncio.sleep(0.01)
            other=event('other',[Plain('B unrelated')],sender='b');await record(plugin,other)
            own=event('own-followup',[Plain('A followup')]);await record(plugin,own)
            await asyncio.wait_for(pending,1)
            state=mention.get_extra(module.OWNER)
            assert state['selection'].anchor['event_id']=='own-followup'
            assert state['selection'].anchor['sender']=='a'
            await pure.aclose()
            results['pure_mention_keeps_sender_identity']=True

            # Exercise a real SDK tool loop and the plugin's completion collector.
            async def lookup(event: AstrMessageEvent):
                return CallToolResult(content=[TextContent(type='text',text='fixture_tool_result'),
                                               ImageContent(type='image',data=base64.b64encode(PNG).decode(),mimeType='image/png')])
            tool=FunctionTool(name='fixture_lookup',description='fixture only',
                              parameters={'type':'object','properties':{}},handler=lookup)
            context.persona_manager.resolve_selected_persona.return_value=(
                'fixture-persona',{'prompt':'PERSONA_MUST_SURVIVE','tools':None,'skills':[]},None,False)
            toolset=ToolSet();toolset.add_tool(tool)
            context.get_llm_tool_manager.return_value=SimpleNamespace(get_full_tool_set=lambda:toolset)
            tool_mode['enabled']=True
            query=event('tool-probe',[At(qq='fixture-bot'),Plain('tool-probe')])
            await record(plugin,query)
            with patch('astrbot.core.astr_main_agent.retrieve_knowledge_base',new=AsyncMock(return_value=None)):
                async for _ in plugin.respond(query):
                    pass
            protocol=plugin.journal.get(initial['room'],'generation:'+query.get_extra(module.OWNER)['gid'])['parts'][0]['messages']
            assert [p['role'] for p in protocol]==['assistant','tool','user','assistant']
            assert protocol[0]['tool_calls'][0]['id']==protocol[1]['tool_call_id']=='call-fixture'
            assert 'fixture_tool_result' in json.dumps(protocol)
            assert 'journal_image' in json.dumps(protocol)
            assert 'data:image' not in json.dumps(protocol)
            results['real_tool_loop_ids_and_complete_reply_saved']=True

            # SDK also supports terminal tools with no final assistant response.
            async def terminal(event: AstrMessageEvent):
                return None
            tool.handler=terminal;tool_mode['step']=0
            query=event('tool-terminal',[At(qq='fixture-bot'),Plain('tool-terminal')])
            await record(plugin,query)
            with patch('astrbot.core.astr_main_agent.retrieve_knowledge_base',new=AsyncMock(return_value=None)):
                async for _ in plugin.respond(query):
                    pass
            protocol=plugin.journal.get(initial['room'],'generation:'+query.get_extra(module.OWNER)['gid'])['parts'][0]['messages']
            assert [p['role'] for p in protocol]==['assistant','tool']
            assert protocol[0]['tool_calls'][0]['id']==protocol[1]['tool_call_id']
            results['terminal_tool_result_persisted_without_fake_reply']=True

            # Per-request strict adapter refuses the core's image-removal retry.
            runner_module=importlib.import_module('native_mm_probe.multimodal.runner')
            raw=object.__new__(runner_module.ProviderOpenAIOfficial)
            raw.provider_config={'id':'fixture'};raw.api_keys=['fixture'];raw.client=SimpleNamespace()
            strict=runner_module.strict_provider(raw)
            assert strict is not raw and strict.client is raw.client
            try:
                await strict._fallback_to_text_only_and_retry()
            except ValueError:
                pass
            else:
                raise AssertionError('image removal allowed')
            results['image_removal_retry_blocked_without_global_mutation']=True
            # Real core stats helper, mocked storage: no probe rows in production.
            assert stat_sink.await_count==3
            recorded=[call.kwargs for call in stat_sink.await_args_list]
            assert all(row['provider_id']=='fixture-provider' and row['status']=='completed' for row in recorded)
            assert [row['stats']['token_usage'] for row in recorded]==[
                {'input_other':64,'input_cached':128,'output':8},
                {'input_other':128,'input_cached':256,'output':16},
                {'input_other':64,'input_cached':128,'output':8},
            ]
            results['core_cache_stats_preserved_including_tool_rounds']=True
        finally:
            for name in ('flow','first','second','pure'):
                if name in locals():
                    await locals()[name].aclose()
            await plugin.terminate()
    print('SDK_PROBE_RESULT='+json.dumps(results,ensure_ascii=False))


asyncio.run(main())
