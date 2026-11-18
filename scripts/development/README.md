# 开发验证工具入口

当前产品目标、最新开发基线、已验候选和开放门统一从[PROJECT-RESTART](../../PROJECT-RESTART.md)读取。前三批已批准的里程碑整理已完成，第三批[PR25](https://github.com/CJBshuosi/billiardsAI/pull/25)已合入dev `d6ce6cb8dde7597a227eefa14bc418c3fb85ba05`；本轮整理结束，不自动继续源码拆分或新实机测试。

2026-11-13的28片[冻结身份](../../docs/operations/windows-replay-milestone-2026-11-13.json)是历史验收基线，保留原source/input/environment及结论，不替代后续候选的新证据。[现行闭环操作](../../docs/operations/minimal-github-windows-blackbox-loop.md)、[原整理计划](../../docs/superpowers/plans/2026-11-13-windows-replay-milestone-cleanup.md)和[分支盘点快照](../../docs/operations/branch-audit-2026-11-13.md)继续作为工具及追溯入口；分支规范以[AGENTS](../../AGENTS.md)为准，历史盘点不是实时状态。

## 日常路径

Mac提交候选 → `blackbox_source.py`导出固定commit的白名单源码与SOURCE-LOCK → 审核后经GitHub源码分支同步 → Windows按固定输入/环境执行Preflight → 通过后新UUID执行FullgameComplete（就绪后须在原90秒窗口内实际观察原生页面播放）→ GitHub结果分支回传有限报告 → Mac校验身份、完整性和清理，并关联本次已观察到的原生播放证据；原生观察与自动结果分别记录。

`FullgameComplete`是直到真实完成的显式验证档，保留短setup/capture/I/O与owned清理门；旧限时Fullgame及CaptureEnd只支持各自合同，不替代它。这里不提供复制旧run命令的快捷方式；新运行使用新候选身份和新UUID，不重用历史许可。

| 工具 | 保留职责 |
| --- | --- |
| [s1_physics_input.py](s1_physics_input.py) | S1开发用物理输入：固定引擎、连续状态、75秒开场/后续12秒静止及实际视频解码；只生成原始像素和标定，不生成建议，不证明模型或大屏通过 |
| [s1_snapshot_probe.py](s1_snapshot_probe.py) | 真实模型组件预检；缺省保留冷启动门，显式prepared档分别记录准备和请求且须有路线/区域，不替代实际大屏 |
| [s1_physics_environment.py](s1_physics_environment.py) | 按固定Windows wheel清单下载并核对，不安装、不导入；实际隔离准备须按具体许可进行 |
| [blackbox_source.py](blackbox_source.py) | 只导出已提交白名单源码、生成/验证SOURCE-LOCK；不是全仓备份工具。README、测试和研发文档不进入现有白名单 |
| [blackbox_validation.py](blackbox_validation.py) | setup/worker/monitor、输入与运行环境校验、有限报告、取消与owned清理；实际profile入口 |
| [blackbox_report.py](blackbox_report.py) | 原子JSON及有界导出公共实现；默认replace语义不可因heartbeat专用重试而整体改变 |
| [Windows初始化](windows/Initialize-BlackboxValidation.ps1) | 准备本机依赖/配置并核验；已有环境不因整理重复初始化 |
| [Windows包装入口](windows/Invoke-BlackboxValidation.ps1) | 调用相同Python validator；可执行策略和本机环境不由本轮文档改变 |
| [整局驱动](../algorithm/run_recording_first_fullgame.py) | 产品端完整录制/结束/后台/媒体与原片校验及短时浏览器观察 |
| [进度观察](../algorithm/recording_fullgame_progress.py) | 有界真实时间线与heartbeat；Windows共享冲突仅这里使用专用短replace重试 |
| [reopen_recording_review.py](reopen_recording_review.py) | 独立持续只读查看已录制的原场媒体；不重新采集/编码，不开放保存/完成选片等业务写入 |

自动测试服务只保留原90秒浏览器观察窗口，期满关闭。持续只读查看有独立UUID、原报告绑定、重新完整媒体核验与关闭标记；它的播放证据和清理状态不能拼接回原测试run。当前实际大屏进程的关闭只针对其精确身份，不按名称杀Python。

## 条件诊断与共享兼容

| 文件/目录 | 保留原因与使用边界 |
| --- | --- |
| [blackbox_capture_end_probe.py](blackbox_capture_end_probe.py) | validator的CaptureEnd档实际调用；只证明结束与截止，不证明28片/浏览器 |
| [blackbox_media_diagnostic.py](blackbox_media_diagnostic.py) | 单组件媒体成本/发布诊断；被codec诊断按文件加载，不是默认整局入口 |
| [blackbox_codec_diagnostic.py](blackbox_codec_diagnostic.py) | AUTO/FFmpeg/预载对照诊断；缺少静态import不能证明未使用 |
| [recording_windows_entry.py](../algorithm/recording_windows_entry.py) | 名称来自旧包，但当前validator仍导入CODEC和summarize；不能按“旧传包已停”删除 |
| [recording_packet.py](../algorithm/recording_packet.py) | entry和历史package/diagnostic共享的路径/pin校验实现，继续保留 |
| [package_recording_first_windows.py](../algorithm/package_recording_first_windows.py)、[package_recording_capture_diagnostic.py](../algorithm/package_recording_capture_diagnostic.py) | 历史离线封装与复现兼容；不用于当前日常GitHub闭环，不自动产新包或上传 |
| [历史诊断目录](../../docs/operations/diagnostics/README.md) | 保存当时工具和证据引用；不直接重跑带已消费运行身份的脚本 |

本轮只整理导航，不移动/删除脚本、测试或Windows依赖。若以后退役工具，先查import、动态文件加载、profile、PS、打包清单、测试和历史证据引用，再单独提交和验证。
