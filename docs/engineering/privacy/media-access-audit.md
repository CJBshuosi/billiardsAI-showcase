# Phase 0 媒体访问与审计运行手册

## 信任边界

人工查看只允许通过 `MediaAccessService.authorize_and_record_access()` 和受控 viewer。身份及
`REVIEWER`/`ACCESS_ADMIN` 角色来自当前 OS 账户固定路径下、权限为 `0600` 的
`.config/billiards-ai/reviewer.json`；CLI 参数不能声明人员、角色、凭据路径或事件时间。
访问 store、retention store 与 Gate store 的父目录必须私有，且不得位于受版本控制的凭据位置。

Phase 0 信任本机 OS 账户和文件系统权限。它不提供 reviewer dashboard、组织级 RBAC/ABAC、
多复核员租约、MDM、DRM、DLP、屏幕录制检测、个人设备证明、SIEM、自动告警或外部不可抵赖审计。

## 会话授权申请、查询与撤销

`grant` 由具备 `ACCESS_ADMIN` 的受信身份为自身 reviewer principal 创建单一 `session_id`
授权；不支持通配符。`check` 只回答该授权记录是否有效，不表示媒体可查看，也不返回 locator。
工作结束后用 `revoke` 撤销。原因必须非空且不得包含控制字符。

授权记录与历史 `ALLOW` 都不是 capability。每次查看仍会重新读取 Story 1.2 retention inventory，
并重新验证 Story 1.1 当前 Gate、素材绑定与固定 `NECESSARY_REVIEW` 用途。

## 受控查看与默认拒绝

`controlled-view` 仅执行一次完整授权判定并向受控 consumer 交付内部 locator；CLI 本身只输出
结构化状态，绝不打印路径、URL、hash、凭据或证据位置。生产 viewer 必须作为该 consumer 集成，
不得直接调用 retention 的内部 locator primitive。

无身份、错角色、无精确 grant、空原因、导出/下载、Gate 失效、素材绑定不符、非 `ACTIVE`
状态或 repository 异常一律拒绝。下载、export、输出路径和长期链接没有 CLI 命令。禁止用个人设备
物化原始影像；当前阶段不能声称能技术识别截图、录屏或所有个人设备行为。

## 审计与安全事件

每次具体访问判定追加事件：repository 分配唯一顺序，服务端生成 event ID、受信 actor/role、UTC
时间、权威 session、动作、原因、结果与稳定 reason code。所有 `DENY` 都是安全事件。`ALLOW`
必须先成功落盘，之后 consumer 才会收到内部 locator；审计写入失败时 viewer 不会收到媒体结果。

使用 `audit` 可按 session 查看脱敏事件。输出和 store 事件均不得包含 locator、裸 URL、内容 hash、
凭据、原始内容或证据路径。发现 store 权限、owner、schema、大小、深度或 JSON 校验异常时，停止访问，
保留文件供受控调查，修复受信配置后再操作；不要手工编辑 JSON。

## 第三方关键帧前置门

`provider-approve` 只记录完成正式评估后的审批边界：供应商、`KEYFRAME`、用途、服务区域、境内/
跨境结论、成本上限和证据摘要。`provider-revoke` 撤销后立即失效。真正调用前必须使用
`authorize_third_party_keyframe_transfer()`，并由 retention 证明资产确为同 session 的 `KEYFRAME`，
再校验 Gate、用途和所有审批维度。该 guard 不选择供应商、不持有 API key、不调用网络、不发送素材，
也不构成法律或跨境合规结论；未选供应商或未完成正式评估时保持拒绝。

## 删除后的重检与恢复边界

删除终态优先。素材一旦不再为 `ACTIVE`，旧 grant、旧 asset ID、旧 Gate 或历史 `ALLOW` 都不能
复活媒体；每次尝试都会重新读取 tombstone 并记录 `DENY` 安全事件。

access store 使用版本化 JSON、进程锁、同目录临时文件、flush/fsync 和原子替换，可发现损坏、非法
事件前缀与序号重排。但同文件 append-only 校验无法证明抵抗同一 OS 账户把整个文件替换为较早且
本身合法的前缀。需要该保证时应迁移到受独立信任域保护的远端不可变日志；不要把 Phase 0 描述为
外部不可抵赖审计。
