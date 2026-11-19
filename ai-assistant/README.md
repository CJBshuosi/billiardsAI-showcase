# AI 复盘助手

面向击球记录查询与中式八球规则问答的独立 AI 应用。采用原生工具调用组织查询、检索和回答校验；此目录使用合成场次，与现场台球运行时分开。

## 一次请求怎样执行

问题 → 绑定场次与规则版本 → 模型选择工具 → 查询记录 / 检索规则 → 结构化生成 → 引用校验 → 返回回答与依据。

## AI Engineer 阅读入口

| 关注点 | 实际技术与设计 | 代码 |
|---|---|---|
| 工具调用与编排 | 原生 `tool_calls`；服务端校验参数，最多 3 轮模型、4 次工具、30 秒共享预算 | [assistant.py](src/cueorbit_replay/assistant.py)、[workflow.py](src/cueorbit_replay/rules/workflow.py)、[execution.py](src/cueorbit_replay/execution.py) |
| 文档处理与分块 | pypdf；按条款与句子分块，保留页码，正文上限 400 tokens | [source.py](src/cueorbit_replay/rules/source.py)、[corpus.py](src/cueorbit_replay/rules/corpus.py) |
| Embedding | multilingual-e5-small，query/passage 前缀，384 维归一化；FastEmbed / ONNX Runtime | [embeddings.py](src/cueorbit_replay/rules/embeddings.py) |
| 向量检索 | PostgreSQL + pgvector，余弦距离；按规则集、版本、语料和编码器身份过滤，Top-5 召回 | [retrieval.py](src/cueorbit_replay/rules/retrieval.py)、[storage.py](src/cueorbit_replay/rules/storage.py) |
| 查询优化与对照 | 保留原问题，显式扩展中英术语；与关键词、原始向量两条基线分别对照 | [query.py](src/cueorbit_replay/rules/query.py)、[术语表](data/rules/query-glossary.json) |
| 生成与引用校验 | DeepSeek、Pydantic 结构化输出；引用限定为本次成功检索结果，并核对引文及来源 | [model.py](src/cueorbit_replay/rules/model.py)、[contracts.py](src/cueorbit_replay/rules/contracts.py)、[evidence.py](src/cueorbit_replay/rules/evidence.py) |
| 业务工具与数据访问 | 场次概览、单杆查询和分页列表；工具校验参数，仓储层限定场次 | [tools.py](src/cueorbit_replay/tools.py)、[repository.py](src/cueorbit_replay/repository.py) |
| 模型接入 | 统一模型接口、原生工具消息与 HTTP 错误解码 | [providers/](src/cueorbit_replay/providers/) |
| 检索与回答评估 | Recall@5、Hit@5、MRR@5 衡量召回与排序；工具、状态和引用另做合同检查 | [检索评估](src/cueorbit_replay/rules/evaluation.py)、[回答评估](src/cueorbit_replay/evaluation.py) |
| API 与可观测性 | FastAPI、ASGI 包装；有界执行、允许字段日志，未知用量保留 null | [app.py](src/cueorbit_replay/app.py)、[application.py](src/cueorbit_replay/application.py)、[observability.py](src/cueorbit_replay/observability.py) |

## 测试与展示范围

**代码验证：207 项离线测试通过。** 使用合成记录、脚本模型和模拟 HTTP；这不是 AI 效果测评，当前暂无对外发布的效果结论。

| 评估层次 | 检查什么 | 可查看的用例 |
|---|---|---|
| 检索 | 相关片段召回、首个相关结果排名、无相关标注时的分母处理 | [指标计算](tests/rules/unit/test_rules_retrieval.py) |
| 工具与证据 | 调用预算、错误参数、未返回的引用、伪造引文与缺失事实 | [编排](tests/rules/unit/test_rules_workflow.py)、[引用](tests/rules/unit/test_rules_evidence.py)、[组合问题](tests/rules/unit/test_assistant.py) |
| 回答语义 | 是否真正答对、是否理解原文歧义，需要结合源文复核 | [状态与证据评分](src/cueorbit_replay/evaluation.py)不替代语义判定 |

```bash
uv sync --frozen
uv run --no-sync python -m pytest tests -q
```

本目录仅展示源码和精选测试；未附私有评测报告、官方规则 PDF、模型权重、视频、密钥或部署配置。完整检索运行需要另外配置规则来源、专用数据库和本地编码器。引用成员与引文校验不等于自然语言语义正确率。

原创代码沿用展示仓库根目录的版权与使用说明，不授予额外使用许可。第三方库与模型各自适用原有条款。
