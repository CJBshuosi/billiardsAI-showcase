# CueOrbit · 一杆

面向中式八球的录像、击球分析与赛后复盘项目。

**开发状态：主要功能已开发，正在进行线下测试。** 本目录用于展示实现与系统设计，不是可直接部署的发行包。

## 从哪里开始看

- [AI 工程实现](ai-assistant/README.md)：RAG、Embedding、原生工具调用、引用校验和评估。

- [系统与模块](docs/architecture.md)：输入如何进入系统，感知、录像、回放和同步如何配合。
- [源码导览](docs/source-guide.md)：各功能的代码位置。
- [隐私与片段保留](docs/engineering/privacy/image-retention-deletion.md)：授权、保留和删除边界。
- [第三方依赖说明](third_party/pooltool-billiards-0.6.0.md)。

## 代码结构

| 目录 | 内容 |
|---|---|
| `ai-assistant` | 独立 AI 应用：规则 RAG、记录查询、工具编排与精选测试 |
| `src/billiards_ai/lab` | 摄像头/黑盒输入接入、现场运行时、大屏 HTTP 与媒体处理 |
| `src/billiards_ai/edge` | 本地录像、逐杆状态、路线辅助、同步与清理 |
| `src/billiards_ai/cloud` | 会话、媒体、分析、训练与复核服务 |
| `src/billiards_ai/shared` | 事件、数据模型与共享合同 |
| `apps/miniprogram` | 微信小程序及云函数；当前产品方向是读取已保存片段 |
| `apps/mobile` | 历史移动端实现，保留供技术参考，不代表当前交付客户端 |
| `tests` | 单元、合同、集成和文档约束测试 |

## 本地环境

Python 支持范围和依赖以 `pyproject.toml` / `uv.lock` 为准，小程序依赖以其 `package.json` / 锁文件为准。常规依赖准备命令：

```bash
uv sync --frozen --extra dev --extra cloud
```

模型权重、真实录像、设备凭据和实际部署配置不随源码提供。配置中的示例 AppID、云环境 ID 与用户路径已替换，不能直接用于现场部署。模型与物理引擎有各自依赖和许可，根许可证不改变第三方条款。

本展示稿保留原测试代码；依赖内部过程文档、现场资产和特定设备的测试不随展示稿提供完整运行条件。已完成的检查及范围见 [展示稿说明](PUBLICATION-STATUS.md)。

## 许可

项目原创代码保留权利，当前仅作源码展示，**不授予开源使用许可**。使用、修改、再分发或商业使用请另行取得授权；依法或依 GitHub 服务条款已享有的权利不受此说明限制。第三方组件沿用各自许可，详见 [版权与使用说明](COPYRIGHT.md)。
