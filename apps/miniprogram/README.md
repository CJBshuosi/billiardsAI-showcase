# billiardsAI 微信小程序

这是新版界面的长期用户客户端。已迁移微信身份、扫码入场、既有现场命令、事实报告、击球片段、收藏和受控分享；不在手机运行 CV 或路线规划。旧版逐杆人工控制保留在明确标记的现场调试区。

同城排名、球技分析、力度校准等尚未接入，仅提供不可用占位。迁移范围与验收记录见仓库的 `docs/implementation/2026-09-01-miniprogram-ui-migration.md`。

## 本地配置

当前开发和部署设备是 Mac；雷神 Windows 只运行现场同步，不编译小程序。

1. 复制 `project.private.config.example.json` 为 `project.private.config.json`，填入小程序 AppID。
2. 复制 `miniprogram/config/env.example.js` 为 `miniprogram/config/env.local.js`，填入 CloudBase 开发环境 ID。
3. 在微信开发者工具中导入本目录。

两个本地文件均已被 Git 忽略。不得提交 AppSecret、微信用户标识、设备凭证或真实媒体。

完整步骤见 `docs/operations/miniprogram-development-deployment.md`；雷神后续同步接入见 `docs/operations/miniprogram-raytheon-sync.md`。

## 测试

```bash
npm ci
npm ci --prefix cloudfunctions --ignore-scripts
npm test
```

三个云函数从 `cloudfunctions/` 这一受控源码单元部署，以便共享同一份留存规则和存储适配器；三个入口仍相互独立。不要在微信开发者工具里单独上传某个函数目录，正式准备统一使用总部署脚本。`media-retention` 当前没有定时触发器，物理删除也默认关闭，只允许显式的合成测试数据。
