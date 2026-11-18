雷神摄像头曝光对比包 — 20260905-v4
适用：RER-U3CAM4K60-CL100 / Sony IMX678
候选源码基线：9a144955377b289ce47de71769d1305dd8585de4

【本轮目的】
对比手动曝光 -7 和 -8 的运动拖影、亮度与录像帧率。
固定 DirectShow + 1920×1080 + 60 fps + MJPG。
每档：手动调参并关闭 AMCap → 重新确认相机 → 2 秒预检 → 确认准备 →
相机初始化后 10 秒准备/采样 → 30 秒录像。
两档共两段视频，不重跑 30 分钟长测，不运行识别模型或正式比赛服务。
-8 是待比较参数，不是已经选定的最终参数。收集完成不等于验收通过。

【开始前】
1. 雷神接电，避免自动休眠，至少留 5 GiB 硬盘空间。
2. 保持机位、USB 接口、缩放、焦点、白平衡、增益、亮度和现场灯光不变。
3. 自动对焦、自动白平衡关闭；曝光使用手动。不要改变其他参数来补偿 -8 的变暗。
4. 清走白纸、棋盘格等，台面六个袋口全部在画面内。
5. 准备白球和黑球；两档尽量用相同摆位、击球方向、近似力度。
   使用正常偏快的一杆，不要极限暴力击球。是否速度相当仍需回看视频核对。
6. 先关闭 AMCap、Windows 相机、OBS、摄像头网页，再启动本包。

【一行运行】
把原名 raytheon-camera-check-20260905-v4.zip 放进雷神 Downloads 文件夹。
无需手动解压，在任意 PowerShell 复制下面完整一行：

& { $ErrorActionPreference="Stop"; $CameraZip=Join-Path $env:USERPROFILE "Downloads\raytheon-camera-check-20260905-v4.zip"; $CameraDir=Join-Path $env:USERPROFILE ("Downloads\raytheon-camera-v4-"+[Guid]::NewGuid().ToString("N").Substring(0,8)); Expand-Archive -LiteralPath $CameraZip -DestinationPath $CameraDir; & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $CameraDir "run.ps1") -ProjectRoot "C:\Users\developer\Documents\billiardsAI-raytheon3" -OutputParent $CameraDir }

每次解压到新目录，不覆盖旧包或项目。无需管理员权限，不安装依赖。
已有 Python 从 current.json 或项目 .venv-rfdetr/.venv 查找。
若已经手动解压，在能看到 run.ps1 的目录执行下面一行：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -OutputParent "$PWD"
项目/环境不同时可追加 -ProjectRoot "实际项目目录" -PythonPath "已有环境\Scripts\python.exe"。

【运行步骤】
1. Press ENTER when ready：按回车。
2. 提示 Set MANUAL exposure -7：此时脚本尚未打开相机，可打开 AMCap，
   确认手动曝光 -7，保存两页参数截图，再完全关闭 AMCap，回到终端按回车。
3. 查看 choose-camera-dshow.jpg，输入顶置新相机编号。不一定总是 0。
   默认预览可能左右裁切，它只用于选相机；不要因此移动机位。
4. 预检后查看终端给出的 exposure-7\preflight\baseline.png，确认完整台面。
   把球摆好，准备就绪再按回车。相机初始化后还有 10 秒准备时间。
5. 等待 RECORDING NOW 或提示音，再在前 10～15 秒内完成一杆。
   录像总共 30 秒，击球后等球运动，不要立即重新摆球或打开 AMCap。
6. 第一段结束、相机子进程退出后，终端会提示 Set MANUAL exposure -8。
   只有看到这个提示才打开 AMCap，将曝光从 -7 改成 -8，
   其他参数和灯光都不变。保存两页参数截图，完全关闭 AMCap，回到终端按回车。
7. 再次看预览确认编号。预检后用相同摆位、相同方向、近似力度再打一杆：
   按回车 → 等准备结束 → RECORDING NOW → 击球 → 等录像结束。
8. 第二段结束后自动打包。没有长测，无需等待 30 分钟。
   摄像头最终保持你手动设置的 -8，脚本不会自动恢复成 -7；先不要再调其他参数。

【停止与失败】
- 每档手动调参后重新检查 AMCap/其他采集进程是否关闭，失败则不打开相机。
- 每次采集前核对预期曝光、自动对焦/白平衡状态；第二档还核对其他参数与第一档一致。
- 预期参数不符、模式读回不符、读帧失败或超时，会停止后续步骤并保留结果。
- 不要为了让脚本通过盲目改参数；成功或失败包都回传。
- 采集与录像每 5 秒更新进度，连续 30 秒无进度或超过总时限则终止本包创建的子进程。
  不会停止既有现场服务。这不是主线底层 camera read 取消适配的验收。
- 可按 Ctrl+C 中止并等待打包；强关窗口可能来不及打包，
  此时保留 camera-result-* 整个目录，手动压缩回传。
- 不拔插相机，不改变机位、灯光；本轮不测断流恢复。

【回传】
终端显示 SEND THIS ZIP BACK 和完整路径。
把 SEND_BACK-*.zip 作为“文件”发回来，不要把 AVI 转成微信视频或压缩画质。
同时附 -7、-8 各自的 AMCap 两页参数截图，注明曝光档位。
结果在新建的 Downloads\raytheon-camera-v4-* 内；压缩包可能有数百 MB，打包需稍等。
单段录像约 512 MiB 上限，达到时停止并保留，封装/最后一帧可能略超限。

【证据与安全边界】
- 相机图像控制只读；只有你在提示暂停时通过 AMCap 手动改变曝光。
- 录像是原分辨率解码图像重新编码为 MJPG AVI，无缩放、叠字、音频，不是原始 UVC 字节流。
- perf_counter_ns 时间戳是主机读取完成时间，不是传感器曝光时间。
- capture-times.csv、frames.csv 分别记录采集和录像映射；队列丢弃和文件大小限制均单独记录。
- 录像的名义播放帧率不能替代实测帧率；60 fps 不证明无驱动丢帧、重复曝光或运动模糊。
- OpenCV 读回一致不等于原生 controls 锁定。自动曝光 get 不受支持时仍无法独立证明手动锁定。
- AUTOFOCUS=2 是已测 DirectShow 手动标志；某些不支持的属性可能返回 0/-1，不能据此推断完整能力。
- 不切换 release、不部署、不注册服务、不迁移或修改比赛/成绩数据。
- 状态库只读版本、数量及最近场次标识；不导出数据库、用户明细、二维码或密钥。
- 进程命令行仅导出允许的本地路径参数与整行哈希，不导出令牌和同步 URL。
- 预览人工选择不构成可信 PnP↔capture handle 绑定。
- acceptance 始终 NOT_EVALUATED，production_ready=false。本地测试不等于 Windows 真机验收。

参考：
https://learn.microsoft.com/en-us/windows/win32/api/strmif/ne-strmif-cameracontrolproperty
https://docs.python.org/3.12/library/time.html#time.perf_counter_ns
