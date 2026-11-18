# billiards-brain-v1 模型卡模板

## 更新记录

| 日期 | 版本 | 变更 | 修改人 |
|---|---|---|---|
| 2026-08-01 | 0.1 | 建立 R&D-0 finalist 模型卡必填结构；当前无 finalist，不构成模型发布 | Codex |

## 当前状态

`BLOCKED_NO_FINALIST`。本模板不得被当作已训练模型、真实视频准确率或商用就绪证明。

## 发布身份（封装前必填）

- Release ID：由冻结发布配置给出。
- Generation ID：由 locked-test 一次性 claim 给出。
- 代码 commit：40 位 Git commit。
- dependency lock SHA-256：冻结依赖锁哈希。
- 训练 seeds：完整列出，不允许只写最佳 seed。

## 数据与用途（封装前必填）

- 冻结数据集 revision 与 manifest SHA-256。
- REAL_VIDEO_GATE revision 和获准用途。
- 人物、会话、球台、机位的 group isolation 证据。
- 训练、dev、locked-test 的样本数与排除原因。
- 数据许可证、商业用途条件和不可再分发范围。

## 模型组成（封装前必填）

- 检测/实例分割 checkpoint 与 ONNX SHA-256。
- 球号身份 checkpoint 与 ONNX SHA-256。
- class map、输入尺寸、归一化与置信阈值。
- tracker/event 配置 SHA-256。
- 标定假设、坐标系和毫米换算条件。
- 运行环境、最低内存/显存和推理后端许可证。

## 支持边界（封装前必填）

- 支持事件：必须逐项列出并链接 locked-test 指标。
- 不支持事件：必须逐项列出；输出必须为 `unsupported`，不得猜测。
- 弃权策略：列出低置信、遮挡、歧义和证据不足条件。
- 已知失败矩阵：光照、模糊、遮挡、聚球、贴库/袋口、机位漂移和球号朝向。

## Locked-test 结论（封装前必填）

- 一次性 claim SHA-256、授权标识、generation 与数据集 revision。
- 机器可读 `locked-test/metrics.json` SHA-256；其 generation、数据集、阈值和硬门状态必须与 release manifest 一致。
- 打开次数必须等于 1。
- 每个硬门的值、阈值、分母、错误、弃权、coverage 和 95% CI。
- 总结论只能是 `PASS`、`STOP` 或 `NARROW`；任一硬门失败不得写 PASS。

## 商用与许可证（封装前必填）

- 每个实际引用的代码、基础权重、训练数据和运行时许可证。
- LICENSES 目录中的 notice 文件与 SHA-256。
- 尚需人工/法律复核的事项；模型卡不构成法律意见。
