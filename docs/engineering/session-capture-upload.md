# Phase 0 店员录像时间与上传运行手册

本流程只处理已通过可信店员身份、当前球台 Gate 和所有参与者当前服务同意的活动会话。真实 Gate 当前仍未获业务批准；开发与演练只能使用合成素材。

## 部署边界

- 必须配置受控 table-binding authority、retention inventory、Gate repository、asset-context evidence repository 和私有 media root。各 authority 独立持久化，不能宣称跨存储事务。
- media root 必须由运行账户拥有、权限 `0700`，包含权限 `0600` 且内容为 `billiardsAI-media-root-v1\n` 的 `.billiards-media-root` sentinel。上传入口只能注入 deployment-owned、单次只读 `BinarySource`；不得把 caller path、文件名、URL 或凭据传入领域模型。
- `SessionCaptureLimits` 六项均必填：容器、视频 codec、最大字节、最大时长、probe timeout 和区间容差。缺失配置必须停止启动。
- 证据操作员通过 `TrustedAssetContextEvidenceService` 从部署身份边界取得封闭 capture permission，再写入受控 `JsonAssetContextEvidenceRepository`。店员上传请求不能声明 operator、旁人分类、匿名化、hash、size、MIME 或 capture facts。

## 操作顺序

1. 店员只查看本 venue 的 live/pending-upload session；响应只显示安全球厅/球台标签、人数、区间、状态和下一动作。
2. `start` 使用服务器 UTC 并冻结 1–2 人 roster；`end` 使用服务器 UTC，结束 session、完成并释放 binding 容量。
3. 异常修正只允许在结束后 48 小时内，携带 expected interval revision 和受控理由；区间标记 `STAFF_ATTESTED`，与同桌其他非取消区间不得重叠。
4. `prepare` 先在 table/session authority 持久化可枚举 attempt，再在私有 root 内以 attempt-specific locator、no-follow/exclusive create 流式写入并计算 hash/bytes；随后复用 FFmpeg probe 的 stream/format 或 frame duration 核验规范化容器、codec、音轨和区间容差。
5. `finalize` 必须消费 prepare 时冻结的 session/interval revision。saga 固定为：attempt authority → object `STAGED` → retention `PENDING_BINDING` → session binding CAS → retention `ACTIVE` → session `MEDIA_BOUND`。只有 retention `ACTIVE` 且 session `MEDIA_BOUND` 才可解析 locator。
6. 请求超时或进程崩溃后，不把客户端“完成”当事实。运行有界 `reconcile_pending_uploads()`，从各 authority 的确定性状态继续 activation/final CAS，或隔离并删除无权威 binding 的 object。无法证明 commit 状态时返回 `UPLOAD_COMMIT_STATE_UNKNOWN`。

## 故障与恢复

- Gate、服务同意或受控证据在 prepare/finalize 间变化：fail closed，保留不可访问 pending record，随后补偿清理。
- FAILED attempt 为终态；重试创建新 attempt 并记录 `supersedes`。不得把旧 attempt 倒退为 STAGED。
- 同 session 同 hash 返回既有 binding；同 session 不同 hash 返回 `SESSION_MEDIA_ALREADY_BOUND`；同 hash 已绑定另一 session 返回 `DUPLICATE_UPLOAD`。
- cancel 只允许未绑定 retention 的 OPEN/ENDED session；与 staging/finalize 使用同一 session/table authority 锁，并释放所有 live binding。
- 审计只保存可信 operator、authority ID、稳定错误码、digest 与生成的 capture/upload/asset ID。严禁记录 raw key/token/openid/JWT、consent/Gate hash、原请求、文件名、路径/locator、URL、凭据或 traceback。

## 共享删除

retention v2 以 `MediaOwnerScope.subject_user_ids` 表达 1–2 位 subject。legacy `owner_user_id` 首次成功 mutation 时无损迁移为单 subject。任一 subject 的本人影像删除请求覆盖共享原片和派生闭包；无媒体反向引用的非影像统计可以保留。
