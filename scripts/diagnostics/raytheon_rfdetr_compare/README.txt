雷神 RF-DETR 双候选本机对照推理包 — 20260905-v1

本包在雷神本机读取已冻结的 slow、fast_cushion、pocket 三段原始 AVI，严格核对
原视频和两套 RF-DETR manifest/checkpoint 的 SHA-256，然后用相同抽帧集合分别推理。

抽帧：全片每 10 帧一张，加上最大像素运动帧前后各 60 帧的逐帧密集窗口，并固定
包含首帧和末帧。输出逐帧框/置信度、检测数、推理耗时及少量代表性标注 JPG。

边界：不开摄像头，不加载任何身份模型或独立 DINOv2 身份资产，不启动/停止现场服务，不写状态库、
标定库、比赛或成绩。原视频和权重不会进入回传 ZIP。每个模型在独立子进程运行；
最长 60 分钟或 10 分钟没有进度会终止该候选并保留其他证据。

NDJSON 时间是按 AVI 名义 60 fps 与帧号推算的 nominal_video_timestamp_ms，不是传感器时间。
所有结果均为 NOT_EVALUATED，identity_evaluated=false，production_ready=false。
运行完成后把终端指出的 SEND_BACK-raytheon-rfdetr-compare-result-*.zip 原文件发回。
