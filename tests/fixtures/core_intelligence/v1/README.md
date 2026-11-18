# Core Intelligence v1 Contract Fixtures

这些fixtures只用于冻结主线间的结构化接口，不包含真实视频、媒体定位符、用户平台身份、token、Prompt正文或原始观察。

链路：

```text
shot-evidence-envelope
  → approved-evidence-batch
  → growth-evidence-snapshot
  → coach-assessment
  → drill-catalog
  → training-plan
  → published-plan-bundle
  → realtime-shot-observation
  → plan-outcome
```

它们不是算法准确率证据，也不证明Provider、真人教练审核或现场设备已经可用。`drill-catalog.json`含合成 approved、能力不足和 disabled 路径；不得将其视为训练有效性证据。每个消费者必须通过Pydantic模型解析，不得直接把JSON字典当业务权威。
