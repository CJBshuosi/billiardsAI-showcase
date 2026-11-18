雷神中式八球 development-only 全链路验收包

本包不写生产状态；每次 Start 都在 Downloads 下创建唯一 result root。为修复旧开发场次
占用同一球桌的问题，control-only 启动时会先停止残留的旧测试包进程，并仅将已安装开发库中
唯一未结束场次按“先云端、后本地”的顺序安全结束。除此以外不改已安装服务。采集桥由包内
C# 源码现场编译，必须保持 unsigned；因此本包永远不是 production ready。

准备：先确认旧 `billiardsAI-field-service` 已停止（脚本还会核对计划任务、旧服务进程和
8787 监听端口，无法证明已停就 fail closed），关闭其他占用摄像头的软件，并确认固定 Python
环境及已批准 detector、identity、DINO 资产仍位于既定 LOCALAPPDATA 路径。包内 candidate
必须包含 capture-setup 和 active calibration。

本轮真机默认使用显式 `-ExistingDevelopmentControlOnly -ObservationOnly`。它从
LOCALAPPDATA\billiardsAI\field-runtime\current.json 读取 sync_base_url、device_id 和
credential_target；若包内带有受控二维码则优先使用包内二维码，否则才复用 current.json。
URL 必须精确等于
https://cloud1-public-example-1461180277.ap-shanghai.app.tcloudbase.com，环境固定为
cloud1-public-example，device/table 固定为 raytheon-01，凭据固定为
billiardsAI/development/raytheon-01；HTTP redirect 禁止。显式覆盖任一值、二维码缺失、
旧服务冲突或 lease 返回的 table/status 不精确匹配都会 fail closed。

该模式直接复用旧开发环境已经部署的六类接口：session open/close、join query/confirm、
command claim/complete，因此两台微信手机仍可扫码认领 RED/BLUE 并选择 breaker，大屏沿用
原有页面。它没有 404 降级，也绝不调用尚未证明部署的 runtime-attestation、session-analysis
PUT 或 field sync；每场 worker 只生成本地产物，development_cloud_publication/小程序分析门
明确 NOT_RUN，accepted=false。它会写入上述既有 development join/control 状态，但不部署、
不写 production。`-LocalOnly` 仍保留作离线诊断，不能与本模式同时使用。

从任意目录启动（命令会持续运行并监督现场服务）：

  $Run=(Get-ChildItem -LiteralPath "$env:USERPROFILE\Downloads" -Recurse -File -Filter "run.ps1" | Where-Object {$_.FullName -like "*billiardsAI-raytheon-eight-ball-field-20260907-v14*"} | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).FullName; if([string]::IsNullOrWhiteSpace($Run)){throw "请先把v14测试包完整解压到下载目录"}; powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run -ObservationOnly -ExistingDevelopmentControlOnly

当前随包冻结的 identity 模型门为 FAIL，因此上述一行命令只能做 ObservationOnly 非验收诊断：
摄像头、采集、跟踪、扫码认领、breaker、场次和本地异步产物可测，但自动规则权威及新分析
云闭环不会启用，summary 必定 accepted=false，development_cloud_publication 门为 NOT_RUN。
去掉 -ObservationOnly、给 control-only 模式显式覆盖云/QR 参数或同时加 -LocalOnly 都会直接
fail closed；只有换入 gate_status=PASS 且
哈希冻结的新 identity 资产后，才允许正式自动轮换/胜负验收。

看到 USB_FIELD_READY 后，脚本会打印 BIG SCREEN、唯一 result root，以及可复制到第二个
PowerShell 窗口执行的 END COMMAND；仅在实际配置二维码时才打印 JOIN QR IMAGE。固定规则
profile 始终显式传为：

  cbsa-chinese-pool-2017-observable-v1

本包绝不会自动创建参赛者、指定 breaker 或伪造比赛事实。看到 JOIN QR IMAGE 后，两台手机
扫码认领 RED/BLUE 并按旧接口选择 breaker；大屏应从 WAITING_FOR_PLAYERS 进入
WAITING_FOR_SHOT。actor、真实 shot start/settle 和本地 job 会记录，但 ObservationOnly 下
自动规则/路线仍为 NOT_RUN。大屏只复用 settled schematic，不新增 raw USB preview；小程序
赛后四态须等新分析端点部署证明后另测，不得用本轮本地 AVAILABLE 补写云端 PASS。
不得用本轮本地结果补写 PASS。

现场顺序（缺少任何一步都不得补写 PASS）：

1. USB_FIELD_READY 后保持同一服务进程和摄像头连续运行至少 30 分钟。期间大屏不得进入
   DEGRADED；确认后在 work\operator-gate-evidence.json 的 capture_30m 中填写
   status=PASS、continuous_healthy_capture=true、带时区的 observed_at_utc 和至少一条
   evidence。脚本同时核对 launch/server-exit 时长、READY 和正常退出；缺操作员收据就是
   NOT_RUN，不能仅凭进程存活推断健康采集。
2. 完成真实击球，验证主线一能记录 shot start、settle 和本地媒体工作项。当前 ObservationOnly
   不具备规则权威，若出现规则结果只能是明确 abstain，不能期待或接受自动 continue/switch/end。
   路线约 2 秒和自动轮换门在本轮保持 NOT_RUN；以后启用正式开发规则权威时，脚本才从 SQLite
   rule receipt 判断真实规则耗时是否 <= 2 秒。
3. 人为制造一次可安全复现的缺失事实或歧义事实，确认只会显示 RULES_BLOCKED/abstain，且没有
   猜测 continue/switch。脚本只在 SQLite 中看到 abstention_reasons 且 decision=abstained
   且 reason_code=fact_adapter_abstained 时判 PASS；没有该证据就是 NOT_RUN。杆中断线产生的
   capture_interrupted 或其他 disconnect tombstone 会明确排除，不能同时充当本门证据。
4. 按顺序各做一次空闲拔线和杆中拔线。杆中断线不得合成正常 shot_end；重新插入后必须
   重新 READY，且使用新 source epoch。若当前 runtime 无法恢复，立即记录 FAIL，不得重启后
   当作恢复 PASS。
5. 只有上一步四项均真实观察到后，才编辑 result root 下
   work\operator-gate-evidence.json：status=PASS、idle_disconnect_abstained=true、
   in_shot_disconnect_abstained=true、reconnect_ready=true、synthesized_shot_end=false，
   同时填写带时区的 observed_at_utc 和至少一条 evidence（截图名、日志时间点或现场记录）。
   未执行保持 NOT_RUN；任一观察失败写 FAIL、reason 和 evidence。脚本不会从空字段推断 PASS。
6. 两场合计至少完成 10 杆 settled shots。第一场结束后执行打印的 END COMMAND 并追加
   -KeepRunning；确认同一 server PID、不同 session ID，再完成第二场，并确认 current.json
   提供的固定二维码 SHA256 不变；缺失或变化均不能通过 QR 复用门。
   最终 gate 还会从持久化规则 receipts 核对两场 source epoch 集合互不重叠；缺 receipt 或复用
   epoch 都不会 PASS。
7. 第一场的 rollover durable 写入 run-state 后，脚本立即以 BelowNormal 优先级启动该场后台
   worker；run-state 的 session receipt 冻结 launcher PID、随机 worker identity、命令身份哈希、
   stdout/stderr、live-resource audit、artifact root 和 manifest deadline。启动失败只记录诊断，
   不破坏已建立的第二场。worker 每个重分析阶段都查询本机 /api/screen：只有目标场明确 ENDED、
   新场身份一致、USB capture 正常、球静止、无路线显示且显示状态空闲时才工作；移动、路线显示、
   DEGRADED、探针失败或状态缺失均保持 PROCESSING/重试，把每次决定写入逐场 JSONL。
8. 第二场结束后执行未带 -KeepRunning 的 END COMMAND。脚本先核对首场 worker PID 与冻结的
   session/worker/runner 命令身份，并等待其 durable result 或严格 deadline；然后停止服务，且只
   为从未启动过 worker 的 ended session 启动一次 offline worker。已启动但失败、超时或重启后
   PID 被复用的场次只生成安全诊断，绝不重复启动或补记 PASS。本轮 control-only worker 不接收
   development-analysis-sync 或任何 sync/cloud 参数；云发布/小程序状态门保持 NOT_RUN，不能用
   本地 AVAILABLE 冒充。

若需在同一服务进程继续第二场，第一场结束后在打印的 END COMMAND 后追加 -KeepRunning。
脚本会确认第一场已经 durable ended（已经由规则终结时不会再次 POST end_session），等待
rollover 后冻结不同的新 session ID；若实际配置二维码才验证 SHA256 不变。只有 rollover 状态
先原子写入 run-state 后才登记并启动第一场 worker；worker 启动失败不回滚或中断新 session。

最后一场结束后，在第二个 PowerShell 窗口执行未带 -KeepRunning 的 END COMMAND。
EndSession 只会结束 run-state 当前冻结且仍为 open 的 session；若该 session 已 ended，
即使服务已经 rollover，也绝不会误结束新 session。最终命令先等待/收割已登记的首场 worker，
再让现场服务正常 STOP；对未登记的末场，以同一 state DB、jobs DB 启动一次 offline worker。
每场使用独立产物子目录，进程重启后仍以 durable launch/result receipts 去重；仅 PID 存活绝非
身份依据。已验证的 development server-exit receipt 只用于服务停止后启动的末场 worker。
worker 按照持久化的 ended_at + 120 秒 deadline 等待 finalization；worker 的空列表不会被当成
成功，runner 也不会因本地 finalization 提前杀掉仍在工作的 worker。每场 worker 必须自行正常
exit=0 且 durable result 的 completed_at 不晚于 manifest deadline。本轮 control-only 会把每场
worker-publication-<session_id>.json 明确写成
NOT_RUN，且只用本地 artifacts/finalization 判断 postgame_120s。保留的云模式才会只读核对该
session 唯一 field_sync_jobs 为 clean complete、session_analysis_publications 没有 pending、
持久化 payload 哈希正确，且最新已发布云状态明确为 AVAILABLE；任一 pending、partial、
blocked、损坏或非 AVAILABLE 都 fail closed。EndSession 始终生成
evidence\field-gates.json，每个必需门都含
PASS/FAIL/NOT_RUN、determination、evidence 和 reason。最终 accepted=true 还要求十个必需门
全部 PASS、恰好两场、非 ObservationOnly 且 identity gate=PASS；否则返回 2，并仍生成
SEND_BACK ZIP 供诊断。不要把 NOT_RUN 改写成 PASS。

首场后台 worker 不会因存在 open 的第二场而直接绕过 live gate：它依赖每次即时读取的真实
capture/route 状态，仅在可审计空闲窗口执行，并在解码、推理及发布边界重复检查。若第二场持续
繁忙、服务降级、进程身份不符或没有足够空闲窗口，首场可保持 PROCESSING 并最终如实错过
deadline；脚本不会用“已启动”或本地 AVAILABLE 伪造 120 秒 PASS。

重要限制：这只是 development best-effort live-first，不是已证明的跨进程 GPU 可撤销 lease 或
强抢占。每次 gate 通过后到下一检查点仍存在 TOCTOU；BelowNormal 只降低 OS 调度优先级，不能
证明 GPU 抢占。`cross_process_revocable_gpu_preemption` 仍是明确的 production gate。现场是否
保持约 2 秒路线时延只由 SQLite 中真实 rule receipts 的 rule_decision_latency_2s 门判定；后台
worker 存在、PID 存活或 idle probe PASS 都不能替代该现场时延证据。

请回传脚本最后打印的 SEND_BACK ZIP 和 SHA256。不要回传 work/control-token.txt。
