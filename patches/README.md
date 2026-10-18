# 表情异步补发

补丁对应 207 当前安装的 [anka-afk/astrbot_plugin_meme_manager](https://github.com/anka-afk/astrbot_plugin_meme_manager) 5.0.7 代码快照，保留其 MIT 许可证及署名（见 `meme-manager.LICENSE`）。这里只保存本次补丁，不覆盖其他生产定制。

NativeMM v2.2.6 在自己的群回复事件中提供协作标记、真实正文 ACK 和补发回调。表情插件在响应钩子中仅清理可见标记并保存独立快照；正文送达后才启动后台选图。最长15秒、最多32个任务；刷新取消并清理临时文件。后台只补图片，不重复发送正文。纯表情回复及其他插件的调用方继续使用原流程。

后台发送等待 NativeMM 的群回复锁和原轮次发送包装释放，再复用独立的单轮 ACK 代理，保留平台消息 ID 与公开图片历史。原轮次重置或群聊插件关闭后不再发送，失败不重试。任一插件刷新时，已经在运行的旧任务都受对应生命周期约束。

应用前对照 `meme-manager-5.0.7-deferred.json` 的 before SHA-256，备份目标文件，再在表情插件目录执行：

```bash
git apply --check /path/to/meme-manager-5.0.7-deferred.patch
git apply /path/to/meme-manager-5.0.7-deferred.patch
```

这不要求目标目录本身有 `.git`。若校验不符，先合并现有改动，不直接覆盖。部署后分别在 WebUI 刷新 NativeMM 群聊和表情包管理器，无需重启 AstrBot。表情插件升级后需重新核对补丁。

隔离环境的回归覆盖：正文发送前没有辅助模型调用、真实 ACK 后只调度一次、快照隔离、标记/代码示例清理、工具中间响应不抢占最终答复、15秒超时、刷新取消、实际旧版表情选图路径，以及已有纯表情和事件循环回归。运行测试时必须将 `get_astrbot_data_path` 与 `get_astrbot_plugin_data_path` 指向临时目录，不加载生产配置或发送真实消息。

发送间隔通过 AstrBot 的默认配置设为 `platform_settings.segmented_reply.interval_method="random"`、`interval="0,0"`，保留分段并取消各段（含首段）的人为等待。配置更新通过 Dashboard 刷新流水线，不修改核心源码或重启进程。需要恢复旧节奏时，可改回 `interval_method="log"`、`interval="1.5,3.5"`，旧对数基数仍保留。
