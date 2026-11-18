用途：只编译并审计 Windows 原生摄像头辅助程序，不启动摄像头、不改服务、不改生产数据。

操作：
1. 完整解压本包到一个新目录。
2. 在解压目录空白处打开 PowerShell。
3. 运行这一整行：

powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\run.ps1"

完成后把 PowerShell 最后一行 `SEND BACK:` 指向的 zip 发回。
如果提示缺少 CMake/Visual C++ Build Tools，也会生成回传 zip；不要自行安装，先把结果发回。
