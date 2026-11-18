雷神 RF-DETR FP32 / FP16 对照诊断包 — 20260906-v1

本包只读取雷神现有的两份已核验 RF-DETR 候选权重，以及上一轮冻结的 slow、
fast_cushion、pocket 三段原始视频。每个候选先跑 FP32，再调用 RF-DETR 1.9.0
公开 inference(compile=False, inplace=True, dtype=torch.float16) 接口跑 FP16。

两种精度使用完全相同的抽帧，输出逐帧框、置信度、GPU 同步后的推理耗时、CUDA
显存峰值、FP32/FP16 框 IoU 和数量一致率，以及检测器单阶段的 30/60 fps 帧预算。

边界：不开摄像头，不启动/停止服务，不加载身份模型，不修改模型、状态库、标定、
比赛或成绩。视频和权重不进入回传 ZIP。该结果只用于人工选择后续唯一候选；不会
自动切换正式模型，不能代表端到端实时性能，也不能证明 production ready。

如 FP16 在本机失败或精度漂移，结果会 fail closed 并保留错误证据。完成后请把终端
指出的 SEND_BACK-raytheon-rfdetr-fp16-result-*.zip 原文件发回。
