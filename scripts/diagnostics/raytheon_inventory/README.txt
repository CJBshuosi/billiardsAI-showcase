雷神只读资产盘点包 20260905-v1

用途：核对当前 release、进程/任务/监听、启动路径、项目与 release 源码哈希、
状态库只读摘要，以及 RF-DETR、身份头、DINOv2、capture setup、正式标定和
shot-local.yaml 的路径、大小、SHA-256 与有限白名单元数据。

边界：不启动或停止服务，不切换 release，不改相机参数，不写状态库，不启用标定，
不复制模型或配置内容。结果始终 acceptance=NOT_EVALUATED、production_ready=false。
清单存在不表示资产适用于正式现场；必须由回传后的关系、哈希及主线预检继续判定。

运行前不必关闭现场服务或摄像头软件。本包不会打开摄像头。
运行结束后，把终端指出的 SEND_BACK-raytheon-asset-inventory-result-*.zip 发回。
