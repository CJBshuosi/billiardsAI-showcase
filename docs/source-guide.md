# 源码导览

一杆由现场采集与视觉分析、录像回放、规则与路线规划、跨端保存，以及分析服务和研发工具组成。以下按功能组织入口；点击模块可直接查看实现。

**产品链路：** 视频输入 → 录像与感知并行 → 击球事件与比赛状态 → 大屏回看、手动保存 → 本人片段同步 → 小程序查看。

**快速阅读：** [整体架构](architecture.md) · [录像与恢复](#2-录像与回放) · [规则与路线](#3-规则与路线规划) · [跨端与用户数据](#4-跨端与用户数据)。

## AI 工程入口

独立 AI 助手负责记录查询与规则问答；与现场录像服务分开，使用合成场次验证。先从 [AI 助手导览](../ai-assistant/README.md) 看数据路径，再进入实现与测试。

| 关注点 | 使用的技术与设计 | 源码入口 |
|---|---|---|
| RAG 知识库 | pypdf、条款/句子分块、页码与版本元数据 | [corpus.py](../ai-assistant/src/cueorbit_replay/rules/corpus.py)、[storage.py](../ai-assistant/src/cueorbit_replay/rules/storage.py) |
| Embedding | multilingual-e5-small / FastEmbed / ONNX Runtime，384维归一化 | [embeddings.py](../ai-assistant/src/cueorbit_replay/rules/embeddings.py) |
| 检索与查询优化 | PostgreSQL + pgvector 余弦检索、版本过滤、Top-5、显式双语术语扩展；独立关键词对照 | [retrieval.py](../ai-assistant/src/cueorbit_replay/rules/retrieval.py)、[query.py](../ai-assistant/src/cueorbit_replay/rules/query.py) |
| 工具调用与编排 | DeepSeek 原生 tool_calls、受限业务工具、轮数与时间预算 | [assistant.py](../ai-assistant/src/cueorbit_replay/assistant.py)、[workflow.py](../ai-assistant/src/cueorbit_replay/rules/workflow.py)、[tools.py](../ai-assistant/src/cueorbit_replay/tools.py) |
| 结构化回答与依据 | Pydantic 响应、检索结果成员校验、短引文匹配、规范来源回填 | [contracts.py](../ai-assistant/src/cueorbit_replay/rules/contracts.py)、[evidence.py](../ai-assistant/src/cueorbit_replay/rules/evidence.py) |
| 评估 | Recall@K、Hit@K、MRR、状态与引用有效性分开检查 | [检索评估](../ai-assistant/src/cueorbit_replay/rules/evaluation.py)、[回答评估](../ai-assistant/src/cueorbit_replay/evaluation.py)、[精选测试](../ai-assistant/tests/) |
| 服务与可观测性 | FastAPI、SQLAlchemy、ASGI、允许字段日志及未知用量处理 | [app.py](../ai-assistant/src/cueorbit_replay/app.py)、[repository.py](../ai-assistant/src/cueorbit_replay/repository.py)、[observability.py](../ai-assistant/src/cueorbit_replay/observability.py) |

## 1. 采集与视觉感知

| 模块 | 源码入口 | 核心职责 |
|---|---|---|
| 统一视频输入 | [CaptureHub](../src/billiards_ai/lab/mainline1_capture_hub.py)、[采集适配](../src/billiards_ai/lab/mainline1_capture.py) | 摄像头与黑盒视频复用输入接口，向录像、感知和预览分发帧。 |
| 现场运行时 | [Field Runtime](../src/billiards_ai/lab/mainline1_field_runtime.py) | 组装输入、模型、录像、协调器与现场服务。 |
| 球桌标定 | [四角标定](../src/billiards_ai/lab/mainline1_calibration.py)、[共享标定](../src/billiards_ai/shared/calibration/) | 将像素坐标转换为台面坐标。 |
| 球体检测与模型适配 | [Vision](../src/billiards_ai/lab/vision/) | 用统一帧契约接入 RF-DETR，隔离模型私有输出。 |
| 球体身份识别 | [身份特征](../src/billiards_ai/lab/mainline1_identity_backends.py)、[身份融合](../src/billiards_ai/lab/identity_fusion.py) | 融合球体身份与置信度，为后续轨迹关联提供依据。 |
| 连续跟踪 | [Realtime Tracker](../src/billiards_ai/lab/mainline1_realtime_tracker.py)、[轨迹片段](../src/billiards_ai/lab/shot_local/fragments.py) | 在检测锚点之间跟踪运动，保留遮挡与丢失状态。 |
| 白球与击球事件 | [白球状态机](../src/billiards_ai/lab/shot_local/cue.py)、[事件观察器](../src/billiards_ai/lab/shot_local/events.py) | 从轨迹观察起杆、接触、碰库与进袋；证据不足时保留未知。 |
| 单杆产物 | [Shot Product](../src/billiards_ai/lab/mainline1_shot_product.py)、[Shot Local](../src/billiards_ai/lab/shot_local/) | 汇总球位、轨迹、事件与可视叠加结果。 |

## 2. 录像与回放

| 模块 | 源码入口 | 核心职责 |
|---|---|---|
| 连续分段录像 | [Session Recording](../src/billiards_ai/lab/mainline1_session_recording.py) | 连续写入原片，完整提交后才允许后台消费。 |
| 检测进度与检查点 | [Recording Processing](../src/billiards_ai/lab/mainline1_recording_processing.py) | 持久化已接纳分段与处理位置，让慢分析独立推进。 |
| 中断恢复 | [Recording Recovery](../src/billiards_ai/lab/field_assistance/recording_recovery.py) | 校验来源与配置，重放已提交前缀并恢复处理状态。 |
| 异步出片 | [媒体处理](../src/billiards_ai/lab/mainline1_recording_media.py)、[片段作业](../src/billiards_ai/lab/mainline1_shot_jobs.py) | 按单杆区间生成视频，分别记录任务和成片状态。 |
| 赛后回看与选择 | [Postgame](../src/billiards_ai/edge/route_assistance/postgame.py)、[回放服务](../src/billiards_ai/lab/field_assistance/replay.py) | 汇总片段清单，在整场结束后回看与手动保存。 |

## 3. 规则与路线规划

| 模块 | 源码入口 | 核心职责 |
|---|---|---|
| 比赛状态与轮次 | [LiveMatchCoordinator](../src/billiards_ai/lab/field_assistance/live_match.py)、[Session](../src/billiards_ai/edge/route_assistance/session.py) | 协调场次、击球方、轮次和结束状态。 |
| 可观测规则事实 | [Rule Facts](../src/billiards_ai/edge/route_assistance/rule_facts.py)、[八球规则](../src/billiards_ai/edge/route_assistance/eight_ball_rules.py) | 将视觉观察投影为规则输入，由确定性规则处理比赛状态。 |
| 击球前路线推荐 | [Planning](../src/billiards_ai/lab/planning/)、[快照辅助](../src/billiards_ai/lab/field_assistance/snapshot_guidance.py) | 基于稳定球态生成候选路线，检查几何可行性、规则条件并排序。 |
| 大屏实时投影 | [Live Source](../src/billiards_ai/lab/field_assistance/live_source.py)、[页面](../src/billiards_ai/lab/field_assistance/pages.py) | 汇集比赛状态、辅助路线和片段进度。 |

## 4. 跨端与用户数据

| 模块 | 源码入口 | 核心职责 |
|---|---|---|
| 现场 HTTP 与视频传输 | [HTTP Transport](../src/billiards_ai/lab/field_assistance/http_transport.py)、[Server](../src/billiards_ai/lab/field_assistance/server.py) | 提供现场接口、安全媒体读取与 Range 播放。 |
| 本地持久化 | [Persistence](../src/billiards_ai/edge/route_assistance/persistence.py)、[Records](../src/billiards_ai/edge/route_assistance/records.py) | 用 SQLite 保存场次、逐杆事实、媒体与同步状态。 |
| 同步与后台任务 | [Sync](../src/billiards_ai/edge/route_assistance/sync.py)、[Cloud Runtime](../src/billiards_ai/lab/field_assistance/cloud_runtime.py) | 交付已保存片段及事实，处理重试和回执。 |
| 片段归属与云端访问 | [Shot Ownership](../apps/miniprogram/cloudfunctions/shared/shot-ownership.js)、[Sync API](../apps/miniprogram/cloudfunctions/sync-api/) | 约束用户、片段归属与媒体访问。 |
| 本人删除与媒体清理 | [云端删除](../apps/miniprogram/cloudfunctions/shared/owner-delete.js)、[本地清理](../src/billiards_ai/edge/route_assistance/recording_cleanup.py) | 协调删除与回执，在处理和用户选择结束后清理不再需要的媒体。 |
| 微信小程序 | [页面与组件](../apps/miniprogram/miniprogram/)、[Client API](../apps/miniprogram/cloudfunctions/client-api/) | 提供入场设置、本人已保存片段查看与删除。 |

## 5. 分析与扩展服务

下列代码提供独立服务或后续能力的实现入口，接入范围与现场验收分别推进。

| 模块 | 源码入口 | 核心职责 |
|---|---|---|
| 分析与物理计算 | [Engine](../src/billiards_ai/cloud/engine/) | 典型杆选择、规则建议、评分、报告与 pooltool 适配。 |
| 教练与训练计划 | [Coach Agent](../src/billiards_ai/cloud/coach_agent/)、[Training](../src/billiards_ai/cloud/training/) | 结构化上下文、输出校验、评估与训练计划服务。 |
| 成长与能力画像 | [Growth](../src/billiards_ai/cloud/growth/) | 从已审核证据生成能力指标与版本化快照。 |
| 人工复核与发布 | [Review](../src/billiards_ai/cloud/review/)、[数据库](../src/billiards_ai/cloud/db/) | 管理审核状态、不可变版本、事务与 outbox。 |
| 计划发布与端侧执行 | [Plan Publication](../src/billiards_ai/cloud/plan_publication/)、[Plan Admission](../src/billiards_ai/edge/plan_admission/)、[Coach Runtime](../src/billiards_ai/edge/coach_runtime/) | 发布与校验计划，在端侧用状态机执行并记录反馈。 |
| 球桌绑定与场次准入 | [Table Binding](../src/billiards_ai/cloud/table_binding/)、[Session Admission](../src/billiards_ai/cloud/session_admission/)、[Session Capture](../src/billiards_ai/cloud/session_capture/) | 管理可信身份、球桌关联、场次准入与受控上传。 |
| 受控媒体访问 | [Privacy](../src/billiards_ai/cloud/privacy/)、[Media](../src/billiards_ai/cloud/media/) | 按身份和用途限制媒体访问，记录访问与删除操作。 |
| 排名与结果结算 | [排名服务](../apps/miniprogram/cloudfunctions/shared/rating-engine.js)、[Ranking Worker](../apps/miniprogram/cloudfunctions/ranking-worker/) | 处理结果确认、评级账本、结算与榜单快照。 |

独立 AI/RAG 助手的代码见上方 AI 工程入口；真实现场数据接入仍单独推进。

## 6. 工程基础与研发工具

| 模块 | 源码入口 | 核心职责 |
|---|---|---|
| 共享数据契约 | [Shared Models](../src/billiards_ai/shared/models/)、[计划发布契约](../src/billiards_ai/shared/plan_publication/) | 统一端云数据模型、事件与发布格式。 |
| Windows 原生采集 | [Native Windows](../native/windows/) | 保留 C++ 与 C# 采集适配、进程间协议实现。 |
| 模型版本管理 | [模型包](../src/billiards_ai/lab/vision/package.py)、[注册表](../src/billiards_ai/lab/vision/registry.py) | 管理模型制品身份、原子激活与回退。 |
| 数据集与模型训练 | [Training Tools](../src/billiards_ai/lab/training/) | 数据切分、标注复核、身份特征训练与检测器评估。 |
| 测试与检查 | [Python 测试](../tests/)、[小程序测试](../apps/miniprogram/tests/)、[开发工具](../scripts/development/)、[候选检查](../scripts/ci/) | 覆盖模块合同、集成行为、回归、候选验证与交付检查。 |
| 配置与打包 | [Configs](../configs/)、[Deployment](../src/billiards_ai/deployment/) | 组织运行配置、现场程序包与 Windows 文件访问控制。 |

<details>
<summary>其他研究实现与历史客户端</summary>

| 方向 | 源码入口 | 内容 |
|---|---|---|
| 经典视觉处理 | [Cloud Perception](../src/billiards_ai/cloud/perception/) | 基于 OpenCV 的定位、分杆与观察流程。 |
| 视频掩码传播 | [Propagation](../src/billiards_ai/lab/propagation/) | 有界传播窗口、跨窗口拼接、结果复核与训练数据导出。 |
| 实时证据聚合 | [Realtime Chain](../src/billiards_ai/edge/realtime_chain/) | 将单杆观察投影为有质量约束的证据与画像增量。 |
| Flutter 客户端 | [Mobile](../apps/mobile/) | 保留的移动端界面与状态管理实现。 |

</details>

目录中的实现、产品当前启用范围与现场验证是不同层次；当前开发重点为连续记录、赛后回看和本人片段保存。
