用途：验证雷神自带的 Windows PowerShell/.NET C# 编译能力。

本包不会打开摄像头，不会安装软件，不会修改服务或生产数据。

操作：
1. 完整解压本包到一个新目录。
2. 在解压目录空白处打开 PowerShell。
3. 运行这一整行：

powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\run.ps1"

完成后，把 PowerShell 最后一行 `SEND BACK:` 指向的 zip 发回。
