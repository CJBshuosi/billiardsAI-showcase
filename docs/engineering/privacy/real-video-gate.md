# REAL_VIDEO_GATE Phase 0 操作手册

真实录像在进入开发、benchmark、算法、标注、训练或产品链路前，必须由本门禁统一判断。代码和测试只能验证门禁能力，不能证明某张真实球台已经合规；未完成现场申请和批准时，实际状态必须保持 `BLOCKED`。

## 1. 准备申请

申请 JSON 必须包含 `scope`、`inspector_id`、`inspected_at` 和非空 `evidence`。`scope` 要逐项记录：仅拍台面、关闭音频、现场标识与联系方式、服务和训练的独立同意文本版本与 SHA-256、批准用途，以及访问、留存、删除策略。现场证据文件保存在受控位置，仓库只记录引用、SHA-256 与采集时间。

```bash
uv run python -m billiards_ai.cloud.privacy.cli create \
  --store /受控目录/real-video-gates.json \
  --application /受控目录/table-3-application.json
```

`--store` 没有默认值，避免意外把真实门禁数据写入代码仓库。新申请始终创建为 `BLOCKED`，命令返回 gate UUID。

## 2. 外部确认和批准

在批准前，由合规专业人员确认实际运营法域、正式告知/同意文本、未成年人安排及可能的跨境供应商处理；最终 Gate 批准人固定为 Boyu。应用不能把这些确认替换成单元测试结果。

批准身份只从操作系统账户数据库中登记的真实 home 下固定路径 `~/.config/billiards-ai/approver.json` 读取；不信任进程 `HOME` 环境变量，CLI 和程序调用方都不能另选路径。凭据必须位于代码仓库外，目录由当前系统账户所有且权限为 `0700`；文件必须由当前账户所有、不是符号链接，并精确设置为 `0600`。凭据使用 v2 keyring：

```json
{"schema_version":"2.0","approver_id":"Boyu","credential_id":"phase0-local-approval","active_key_id":"2026-q3","keys":{"2026-q3":"64位小写十六进制密钥"}}
```

读取层仍可验证历史 v1 单 key 凭据，但它只映射为固定 `legacy-v1` key ID。Approval MAC 算法严格按持久化版本分派：无 `mac_version` 的 schema 1 记录按 `0a154bf` 的原始字段/raw-prefix 规则验证；显式 v2 按 `0e615b1` 的 normalized-prefix 规则验证；新批准使用 raw-prefix v3，并将 `mac_version=3.0` 本身纳入 MAC 载荷。不使用“任一算法通过即接受”的 fallback；错版本和跨算法重放默认拒绝。v1 凭据只是迁移兼容入口，不应作为新安装格式。

完成现场检查并取得外部签批证据后执行：

```bash
mkdir -p ~/.config/billiards-ai
chmod 700 ~/.config/billiards-ai
chmod 600 ~/.config/billiards-ai/approver.json

uv run python -m billiards_ai.cloud.privacy.cli approve \
  --store /受控目录/real-video-gates.json \
  --gate-id 00000000-0000-4000-8000-000000000000 \
  --approval-evidence-sha256 64位小写十六进制摘要
```

CLI 故意不提供 `--approver`、`--credential` 或 `--approved-at`。批准时间由服务内部受信 UTC clock 生成，不允许调用方回填。批准会追加不可变的 `APPROVED` revision；不会直接改写原 revision。`approve` 只输出 `门禁已批准`，不代表任何具体素材或用途已经获准。

### 凭据安全轮换

1. 先在 `keys` 中加入新 key，保留所有尚有历史批准引用的旧 key。
2. 将 `active_key_id` 切换为新 key；之后的新批准会记录新 `key_id`。
3. 读取历史时依每条批准的 `key_id` 选择对应 key。未知或被删除的 key 会使相关 store 默认拒绝；在历史批准仍需保留审计时，不得删除对应旧 key。

### v1 store 显式迁移

非空且尚无可信 registry anchor 的 store 永远不会自动建立信任。迁移不会改写 append-only 历史，必须由操作者先核对 store，再以文件原始字节的预期 SHA-256 显式锚定：

1. 停止该 store 的 writer，备份 store、`approver.json`、`gate-head.key` 与 `gate-heads.json`；后两者若尚不存在则记录为“不存在”。
2. 用受控工具计算原始文件摘要，例如 macOS `shasum -a 256 /受控目录/real-video-gates.json`。
3. 先 dry-run；它会完整解析 revision 链、按 v1 的精确 canonical MAC 验证旧批准并核对预期摘要，但不写 registry：

```bash
uv run python -m billiards_ai.cloud.privacy.cli migrate \
  --store /受控目录/real-video-gates.json \
  --expected-store-sha256 预先核对的64位摘要 \
  --dry-run
```

4. dry-run 成功后去掉 `--dry-run` 执行锚定，再运行 `show`/`evaluate` 验证读取。错误摘要、损坏历史或无法验证的旧批准都不会修改 store 或 registry。
   - dry-run 只允许创建/使用并发 lock，不创建或改写 `gate-head.key`、`gate-heads.json`、namespace 或 transaction。仅当 registry 与 key 两者都确实不存在时，正式 execute 才能初始化第一份 key 与可信 namespace。registry 缺失但 key 存在（包括 key 损坏）是 orphaned trust state，dry-run 与 execute 都必须 fail closed，不得用它重新锚定旧 store。反之，只要任何 schema 1/2 registry 已存在，key 缺失或损坏也表示全局信任状态不完整，针对已有或新 store 的 dry-run 与 execute 都必须失败；不得生成随机 key 修补或分裂任何 namespace。
   - 如果账户中已有 baseline `0a154bf` 生成的 schema 1.0 `gate-heads.json`（顶层 `gates`），同一个显式 `migrate` 流程会按原始持久化字段逐 revision 计算 prefix chain digest，并验证每一次旧 approval canonical MAC、最终 chain digest 与旧 head HMAC，再将它升级为只包含该 store identity 的 schema 2.0 namespace。多轮批准、revise/revoke 历史不会被改写；显式或混合的未知 `mac_version`、任一旧 prefix/approval 篡改均拒绝。普通读取不会自动升级或信任 schema 1 registry，未知结构也始终拒绝。
   - 升级成功后可以继续迁移或创建其他 store；它们各自使用独立 namespace。若 registry 原子替换已完成但父目录 fsync 报错，以相同 store 和 expected SHA-256 重试：仅当已提交 namespace 的 HMAC、store digest 与完整 heads 都完全一致时，重试才幂等成功，否则拒绝。
5. `gate-heads.json` 与 `gate-head.key` 是同一 OS 账户下所有 store 共享的全局信任状态，不能为单个 store 直接恢复整组旧备份。整组回滚前必须全局停止所有共享该 registry/key 的 store writer，确认所有 namespace 自备份后均无新 revision，并把所有 store、registry 与 key 恢复到同一一致快照。只停止一个 store、其他 store 仍运行，或任一 namespace 已有新 append 时，禁止恢复共享 registry/key。
6. 单个 store 的恢复只能在保留共享 key 与其他 namespace 最新状态的前提下做逐 namespace 恢复，并由受控工具重新验证该 store 的预期摘要、完整 revision 链、目标 namespace HMAC 与事务状态；当前 Phase 0 CLI 不提供 namespace 覆写命令，因此应停止目标 writer、保留现场并按完整性故障升级处理，不得手工编辑 registry。仅完成迁移且确认全局一致快照满足上一步条件时，才可采用整组回滚。

Store identity 是 canonical absolute store path 的 SHA-256，只在 registry 中保存摘要、不泄露路径。移动或重命名 store 会产生新的 identity，必须停止 writer、重新核对摘要并走显式迁移；不能复制旧 namespace 冒充新路径。

## 3. 查询与用途判断

```bash
uv run python -m billiards_ai.cloud.privacy.cli show \
  --store /受控目录/real-video-gates.json \
  --gate-id 00000000-0000-4000-8000-000000000000

uv run python -m billiards_ai.cloud.privacy.cli evaluate \
  --store /受控目录/real-video-gates.json \
  --gate-revision-id 00000000-0000-4000-8000-000000000000 \
  --asset-context /受控目录/asset-context.json \
  --purpose SERVICE_ANALYSIS
```

`show` 只是审计状态查询，输出 `门禁已批准` 或 `门禁未批准`；它不表示任何素材可以使用。所有业务链路都必须调用 `authorize_real_video_use(repository, ...)`，同时提交当前 Gate revision、采集时间 `captured_at`、采集所属 `captured_under_revision_id`、与 Gate scope 绑定的 fingerprint、素材 SHA-256、逐主体分类、请求用途和匿名化/同意证据。`repository` 首参是 Phase 0 的显式依赖注入，用于在同一次判断中解析并核对当前 revision；这是对 AC5 简写签名的有意扩展。不得先单独读取 `APPROVED` 再绕过素材用途判断。

素材必须在当前 revision 获批之后采集，并且 `captured_under_revision_id` 必须等于当前批准 revision。批准前、撤销期间、旧 revision 或另一个相同配置 Gate 下采集的素材都不会被后来批准追认。`captured_at` 不得超出当前受信 UTC 时间五分钟。已同意但当前用途不匹配的主体返回 `需匿名化或补授权`，而不是不可补救的硬拒绝；同意版本与当前 Gate 不符仍然默认拒绝。

只有 `evaluate` 命令可以输出用途授权令牌 `可继续`。授权判断的公开输出只有：

- `可继续`，退出码 `0`
- `不可处理`，退出码 `3`
- `需匿名化或补授权`，退出码 `4`
- 记录损坏、权限错误或输入无效时输出到 stderr 的 `不可处理`，退出码 `5`

## 4. 默认拒绝与变更

撤销 Gate 会清除当前 revision 的现场证据；旧检查和旧证据不能直接用于重新批准。必须通过 `revise(...)` 提交新的现场检查和证据后，才能再次执行批准。create、revise、approve 与 revoke revision 的 `created_at` 都来自服务内部受信 UTC clock；调用方只能提交 create/revise 的检查时间，且该时间受有限未来偏差约束。

缺少检查项或证据、批准身份不匹配、文件权限不安全、JSON 损坏、未知 schema、旧 revision 或未批准用途都默认拒绝。服务内部 revision 创建时钟与领域模型共同使用 `PRIVACY_MAX_FUTURE_CLOCK_SKEW`（当前五分钟）作为唯一允许偏差：设备检查时间可以有限领先 revision 创建时间，超过该界限则拒绝。批准仍不得早于检查，证据仍不得晚于检查，素材采集与批准/revision 的既有因果约束不变。设备、球台、摄像机视野、音频、告知/同意文本、用途或访问/留存/删除策略变化后，要创建新的 `BLOCKED` revision 并重新检查批准。旧批准只保留审计价值，不能继续放行。

出现未授权可识别旁人时，普通分析、标注和训练链路都不得使用源素材。只能先在受控环境生成裁剪、遮挡或匿名化后的新派生素材，并同时记录画面处理和可重新识别元数据清除证据，再重新分类和判断。`ANONYMIZED` 的 `AssetContext` 还必须携带源素材 `asset_id / sha256 / captured_at / gate_revision_id / gate_scope_fingerprint` provenance，与 `AnonymizationRecord` 一致，并绑定当前批准 Gate；源素材不得早于批准，且满足 `source captured_at <= processed_at <= derived captured_at`。source/output SHA-256 必须不同，仅更换 asset ID 不构成匿名化。Story 1.2 可在之后从 retention registry 构造该契约，本 Story 不依赖其实现。

## 5. 测试 fixture 与真实记录

自动化测试只使用合成 UUID、引用、摘要和临时目录。测试中出现的 `APPROVED` 仅证明状态机、持久化和用途矩阵工作正常，不代表任何真实人员、场地或素材获得批准。真实证据、凭据、视频和 gate store 均不得提交到仓库。

## 6. Phase 0 信任边界

- 批准身份的安全边界是运行进程的操作系统账户和该账户下的固定凭据文件。HMAC 能发现缺少凭据的伪造和离线篡改，但任何能以该账户读取或改写凭据的人仍处于信任边界内；它不是独立硬件身份或远程 IAM 证明。
- Gate store 的当前链头另外记录在真实 home 下的 `~/.config/billiards-ai/gate-heads.json`，并用独立 `gate-head.key` 做 HMAC 认证。每次 revision append 都同步更新链头；store 被截断、回滚或同步改写多条 revision 时，读侧与受信链头不匹配并 fail closed。可以修改该受保护 home 目录及密钥的同一 OS 账户仍是 Phase 0 信任边界。
- Registry 按 store identity 隔离 namespace；同一账户下多个合法 `--store` 不共享 Gate head 集合。已存在 namespace 而对应 store 文件缺失时是完整性故障，只有从未锚定的新空路径可以初始化。
- Append 使用固定的 store lock → registry lock 顺序和 HMAC 认证的 store-specific pending transaction：先 fsync pending，再原子替换并 fsync store，最后 finalize namespace。重启读取时，store 匹配 old 状态会幂等清理/回退 pending，匹配 new 状态会幂等 finalize；两者都不匹配则默认拒绝。因此三阶段故障可能让当次命令失败，但不会把已提交 store 永久留在不可读取状态。
- 凭据和 Gate store 的直接目录必须仅限当前账户访问；其全部祖先目录必须是实际目录、由当前账户或 root 所有，并且不可由组或其他账户写入。
- `AssetContext` 的旁人状态、素材摘要、采集时间、所属 revision 和匿名化证据均由调用方声明。本门禁验证声明之间的一致性，不会在授权函数中重新读取视频字节、核验文件摘要或执行视觉识别。
- `consent_evidence_sha256` 是外部同意记录的完整性索引。Phase 0 只校验其格式并将同意版本绑定到 Gate scope，不读取或验证该摘要对应的外部记录；运营流程必须保证索引真实可追溯。
- repository 的 `get_current(...).status` 只供审计与状态展示，不构成素材使用授权。业务准入必须调用 `authorize_real_video_use(...)` 并提交素材和用途上下文。
- Phase 0 JSON store 在读取时校验完整 revision 集合；任一 gate 记录损坏都会让整个 store fail closed。该选择优先避免在损坏数据上部分放行，但随着多球台扩展，需要迁移到具备逐 gate 隔离和全局 revision ID 唯一索引的存储后端。
- 代码、测试和 HMAC 记录都不能证明某张真实球台已经完成告知、同意或现场检查；真实业务实例在取得正式证据和 Boyu 批准前必须保持 `BLOCKED`。
