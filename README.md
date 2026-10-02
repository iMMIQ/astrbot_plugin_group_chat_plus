# NativeMM 群聊

面向原生图片模型的 AstrBot 群聊插件，基于 Him666233 的 [Chat Plus](https://github.com/Him666233/astrbot_plugin_group_chat_plus) 重写运行入口。保留 AGPL-3.0 许可证和上游署名。

**v2.0.0：消息先保存，图片直接进模型上下文。** A 发图、B 插话、A 再 @ 提问时，图片留在 A 的原消息里；不先转述，不依赖平台 caption，不把所有群历史拼进一个大 prompt。

## 已实现

- 按平台实例、bot 账号、群隔离的 SQLite 时间线；纯图片、不触发回复的消息也持久化。
- 原始图片托管、内容哈希去重、重启恢复、TTL、磁盘配额和请求租约。
- 当前消息、明确引用、同人近期图片、最近群聊的分层选择；真实 user/assistant/tool 角色。
- 明确 @ / 回复 bot / 框架唤醒直接生成；纯 @ 最多等待 1 秒收同人的后续消息。
- 每房间生成串行、平台消息与生成去重；完整工具轮次记录，分段发送与生成全文分开。
- 人格、KB、工具、常规请求扩展保留；基于 AstrBot 原生 Agent 构建/执行/发送钩子。
- OpenAI 兼容提供商采用独立请求对象阻止“图片失败后去图重试”，不修改全局提供商。
- 可选主动插话：概率候选 + 硬冷却 + 一个 JSON 门控，再走完整 Agent；默认关闭。
- 热刷新时一次性恢复旧版 Chat Plus 留下的、可准确识别的请求钩子包装。

运行入口不加载旧 `utils/`、`private_chat/` 或 `web/`。旧情绪/质量/疲劳评分、图片转述、错字模拟、独立 Web 面板不参与新流程；保留的上游文件仅供对照。

## 支持范围

当前适配 AstrBot `>=4.28.2,<4.29` 的本地 Agent、NapCat/aiocqhttp 群聊、文字和图片。当前 GLM OpenAI 兼容路径已实测历史图片和多图输入。其他提供商的模态与错误处理需单独验证，配置中的“多模态”标签不能替代实际测试。

群聊回复采用缓冲输出，正常装饰和分段发送继续由框架处理。原生 Live/TTS、音频/视频/文档理解、私聊和长期摘要不在此版本范围内。引用/转发有界展开（引用最多 3 层、转发最多 20 节点）；未支持的附件明确标记不可用。

## 配置

WebUI 中仍使用唯一插件名 `astrbot_plugin_group_chat_plus`，显示为 **NativeMM 群聊**，可原地刷新替换旧 Chat Plus。仓库更新来源已改为本 fork。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `provider_id` | 空 | 沿用当前会话模型；图片能力需明确配置 |
| `enabled_groups` | `[]` | 空表示所有群 |
| `input_token_budget` | 32768 | 总输入预算，覆盖人格/工具/文字/图片估算 |
| `max_context_messages` / `context_minutes` | 40 / 10 | 背景候选上限与分钟窗口 |
| `association_seconds` | 120 | 同人前序图片关联窗口 |
| `max_images` / `background_images` | 6 / 2 | 每请求总图片上限 / 背景图片上限 |
| `image_token_reserve` | 1600 | 单图最低预算估算，不代表服务商计费 |
| `mention_wait_seconds` / `media_wait_seconds` | 1 / 3 | 纯 @ 补发窗口 / 附件等待秒数 |
| `max_image_mib` | 10 | 单图字节上限 |
| `media_retention_hours` / `journal_retention_days` | 24 / 7 | 原图小时 / 事件日志天数 |
| `room_media_mib` / `total_media_mib` | 256 / 1024 | 单群软配额 / 实例硬配额 |
| `auto_reply_enabled` | false | 允许主动插话 |
| `auto_candidate_probability` / `auto_reply_cooldown` | 0.02 / 60 | 主动候选概率 / 硬冷却秒数 |

预算估算保守，并给模型输出留空间；人格和工具本身超过预算时会明确提示。当前问题与关联附件超限时要求分批提问，不悄悄丢掉问题对象。

内置群 ICL/caption 应关闭，由本插件负责上下文。`astrbot_plugin_group_context_flow` 的已知 opt-out 标记在本插件管理的群生效，避免重复注入；无需关闭其他群的整个插件。其他插件若同样整体覆盖 contexts，需要明确上下文所有者，不能承诺任意覆盖器自动兼容。

## 数据与命令

数据目录：`data/plugin_data/astrbot_plugin_group_chat_plus/multimodal_v1/`。框架原 UMO 和每个人的 persona 选择保持原状；群转录以插件日志为准，框架会话只保存不含 base64 的最近一轮镜像。

- `/mmstatus`：本群事件、附件和最近生成状态。
- 管理员 `/mmreset`：移动本群上下文边界；原始文件按保留期清理。

首次切换后从新接收的消息开始。旧版 `[Image]` / `[多模态消息]` 不能恢复图片，不自动将旧历史注入新请求。框架 `/reset` 管理个人框架会话；共享房间的上下文边界使用 `/mmreset`。

失败请求保留用户事件。发送结果不确定时不自动重发；平台没有幂等发送接口时不保证严格 exactly-once。默认诊断包含事件/附件 ID、实际图片数和预算，不打印群原文或完整图片 data URI。

## 验证

组件测试：

```bash
uv run --with pytest --with aiohttp --with pillow python -m pytest -q tests/multimodal
```

在已安装 AstrBot v4.28.2 的独立 Python 进程中运行 SDK 探针：

```bash
python tests/multimodal/sdk_probe.py "$PWD"
```

SDK 探针使用临时日志和离线模型，调用真实框架构建、钩子和工具循环，不加载生产插件配置、不发送群消息。已通过组件测试、SDK 探针，以及当前 GLM 的红/蓝测试图多图识别。热刷新后的真实 QQ 收发由使用者在测试群确认。

[架构设计](docs/MULTIMODAL_DESIGN.md) · [实施验收](docs/MULTIMODAL_PLAN.md) · [协议示例](docs/examples/multimodal-context.json) · [原始上游说明](docs/UPSTREAM_README.md)
