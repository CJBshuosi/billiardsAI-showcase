# Phase 0 影像留存与删除手册

本流程对云端 session 影像实施一个权威清单和服务端删除状态机。它不代替法律审查，也不声称单元测试就能证明某个实际素材已经删除。

## 留存基准

只有复核和交付都完成时才产生云端到期时间：

```text
retention_expires_at = max(reviewed_at, delivered_at) + 30 days
```

所有时间是 UTC。首次里程碑也不能超过服务端当前时间 5 分钟；期限开始后，重复或较晚的里程碑事件不会静默延长它。边缘端 48 小时环形缓冲不属于本 Phase 0 手册。

## 范围与安全边界

删除覆盖视频、抽帧、关键帧、截图和所有可回溯派生影像。每个生产者必须登记 source lineage；执行前会再次计算派生闭包，完成前还会全表反查。未登记的对象不能被宣称已删除。

可保留分数、计数等非影像聚合。聚合注册时必须用 `media_ref_paths` 声明媒体引用的 JSON Pointer，并同时登记存在且同 owner 的 `media_asset_ids`；asset ID、hash、locator、URL 的检测结果必须与索引和 JSON Pointer 精确一致，否则注册直接 fail closed。删除时清除这些已验证路径并再次扫描剩余结构；损坏或事后不一致会使请求进入 `FAILED(REFERENCE_SCRUB_FAILED)`，不会假报完成。

真实影像注册只接受不含生命周期状态或里程碑的创建 DTO，并仍经过 Story 1.1 的 `authorize_real_video_use(...)`。服务端把 Gate 上下文与实际 asset ID、hash、捕获时间及所需 source provenance 逐项绑定，再生成 `ACTIVE`、审计时间和空里程碑。非删除资产的 locator 在整个清单中全局唯一，不支持跨 owner 共享物理对象。删除完成后 locator 被清空，权威解析会 fail closed；保留的 tombstone 防止迟到上传使素材“复活”。

Phase 0 采用两个明确的保守权衡：

- 一个 session 内即使只删除某个 asset，该 session 后续注册也会被 tombstone 阻止；新 session 不受影响。这避免迟到上传造成复活，代价是同 session 不能再补传。
- 某用户已删除影像的 content hash 会永久阻止该用户重新注册相同内容，包括新 session。这是超过 AC 最小措辞的保守隐私策略，若产品后续要允许用户重新上传，必须先建立显式政策迁移。

## 可重复操作

部署时由运维预先创建私有 media root；应用程序不会创建它，也不会修改其权限：

```bash
mkdir -m 700 /private/media-root
printf 'billiardsAI-media-root-v1\n' > /private/media-root/.billiards-media-root
chmod 600 /private/media-root/.billiards-media-root
export BILLIARDS_MEDIA_ROOT=/private/media-root
```

root 必须已存在、当前账户所有、权限精确为 `0700`，哨兵文件必须是 `0600` 且内容精确匹配。`/`、根目录的一级子目录、HOME/CWD 及其一级子目录都会被拒绝。CLI 不接受 `--media-root`，因此操作员不能用参数改变删除树。

运行命令时始终显式指定仓库外、权限为 `0700` 的私有 retention store 目录：

```bash
python -m billiards_ai.cloud.privacy.deletion_cli --store /private/path/retention.json submit --user USER --session-id UUID --idempotency-key KEY
python -m billiards_ai.cloud.privacy.deletion_cli --store /private/path/retention.json status --user USER --request-id UUID
python -m billiards_ai.cloud.privacy.deletion_cli --store /private/path/retention.json list-due --owner USER
python -m billiards_ai.cloud.privacy.deletion_cli --store /private/path/retention.json execute --request-id UUID --executor Boyu
python -m billiards_ai.cloud.privacy.deletion_cli --store /private/path/retention.json retry --request-id UUID --executor Boyu
python -m billiards_ai.cloud.privacy.deletion_cli --store /private/path/retention.json verify --user USER --request-id UUID
python -m billiards_ai.cloud.privacy.deletion_cli --store /private/path/retention.json reclaim --request-id UUID --executor Boyu --reason 'previous process confirmed stopped'
```

CLI 不接受任意待删文件路径。物理目标只来自权威清单，并限制在显式 media root 下。路径穿越、绝对路径、symlink 和越界目标会被拒绝；filesystem adapter 每次从已验证 root fd 出发，逐级使用 `openat/O_NOFOLLOW` 打开 parent fd，再相对该 fd 执行 `unlink`，中间目录被并发替换也不能把删除导向 root 外。

**重要身份边界**：Phase 0 CLI 是受信任的本机运维工具，`--user`/`--owner` 只是声明，不是认证凭据。它不能暴露给终端用户或直接照搬到 FastAPI。未来 HTTP 入口必须从服务端已验证身份注入 `current_user_id`，不能信任 payload 声明。`list-due` 是运维视角，所以必须显式指定 owner；`verify` 仅输出与公共状态相同的脱敏投影。

## 失败、重试与验证

状态只有 `RECEIVED | IN_PROGRESS | COMPLETED | FAILED`。领取执行时会与 `IN_PROGRESS` 原子写入 STARTED 事件；每次结束都留下 attempt 证据。任一删除、不存在验证或反向引用擦除失败，请求都是 `FAILED`，不得显示“已删除”。`KeyboardInterrupt` 等中断会先写入失败证据再重新抛出；若上次进程被强杀导致孤立 `IN_PROGRESS`，操作员必须确认旧进程已停止后使用 `reclaim`，记录原因并转为可重试 `FAILED`。不允许 `IN_PROGRESS→IN_PROGRESS`双执行。

`retry` 在同一 request ID 上增加新 attempt；已不存在对象作为幂等验证成功，完成证据会分别记录实际验证数和执行前已不存在数。

`COMPLETED` 只能在清单内每个对象都验证不存在、反向引用已擦除、locator 已失效且完成证据已落盘后出现。最终范围复核与 `COMPLETED` 写入共享 repository 的同一 flock 事务；进入 `IN_PROGRESS` 后的范围重算、扩展或 CAS 失败都会追加带实际已验证数量的类型化 `FAILED` attempt。公共状态不输出存储路径或内部错误证据。
