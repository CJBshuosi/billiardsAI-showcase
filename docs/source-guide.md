# 源码导览

| 关注点 | 主要入口 | 设计重点 |
|---|---|---|
| 统一视频输入 | `src/billiards_ai/lab/mainline1_capture_hub.py` | 统一采集边界，让物理相机与黑盒输入复用处理链 |
| 现场组装 | `src/billiards_ai/lab/mainline1_field_runtime.py` | 在运行时工厂组装依赖，避免让业务模块反向依赖 CLI |
| 逐杆产物 | `src/billiards_ai/lab/mainline1_shot_product.py` | 汇总跟踪与事件产物，保留事实和可视展示边界 |
| 分段录像 | `src/billiards_ai/lab/mainline1_session_recording.py` | 管理录像生命周期，区别已提交媒体与处理中状态 |
| 录像消费 | `src/billiards_ai/lab/mainline1_recording_processing.py` | 让媒体处理消费持久录像，而不是以界面状态代替处理完成 |
| 大屏与 HTTP | `src/billiards_ai/lab/field_assistance/` | 分离路由、媒体安全读取、Range 传输和业务状态 |
| 本地状态与同步 | `src/billiards_ai/edge/route_assistance/` | 处理本地片段状态、同步、删除与清理 |
| 云端后台任务 | `src/billiards_ai/lab/field_assistance/cloud_runtime.py` | 独立处理事实、已保存媒体、回执和删除任务 |
| 分析与物理路径 | `src/billiards_ai/cloud/engine/` | 分离典型杆选择、规则建议、评分和 pooltool 物理适配 |
| 教练服务 | `src/billiards_ai/cloud/coach_agent/` | 结构化上下文、结果校验、评估和训练规划服务 |
| 复核与持久化 | `src/billiards_ai/cloud/review/` | 将审核状态、事务、发布和 outbox 分开组织 |
| 小程序 | `apps/miniprogram/` | UI、云函数与同步 API 分层，保留权限和片段归属约束 |

仓库保留了一些研究与历史客户端代码；目录存在不等于当前产品已经启用或完成现场验收。本公开稿没有把独立实验性的 AI/RAG 应用表述为此仓库中已集成的产品能力。
