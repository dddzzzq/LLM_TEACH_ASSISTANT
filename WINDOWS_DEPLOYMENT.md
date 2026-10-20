# Windows 本机部署

部署采用 Windows 11 + WSL2 Ubuntu 24.04。项目仍保存在
`D:\dzq\LLM_TEACH_ASSISTANT`，四个模型仍保存在 `D:\dzq\models`。
Python 由 uv 管理，运行依赖以 `ai_engine_python/pyproject.toml` 和 `uv.lock` 为准。
旧 `requirements.txt` 是原 Linux 环境的包清单，不用于本次部署。

## 当前已完成与待完成

2026-10-13 已完成：

- 安装 WSL 3.0.3，启用 Windows 的 WSL/虚拟机平台功能。
- 下载 Ubuntu 官方镜像并校验清单中的 SHA256，镜像保存在 D 盘。
- 将 WSL 的虚拟磁盘目标与交换文件配置到 D 盘。
- 在 D 盘创建 uv 管理的 Windows Python 3.11.17 环境并锁定依赖。
- 四个模型均通过 RTX 5080 Laptop GPU 的实际推理检查。
- Paddle CPU OCR 通过图片文本识别检查。
- 真实 gRPC 服务通过 ExtractText 和 DetectAIGC RPC 检查，检查后停止。
- Go 现有测试通过；前端 8 项交互测试和 Vite 构建通过。
- Windows 与 Linux 后端二进制编译通过，PowerShell/Bash 脚本语法检查通过。
- 重启后已成功导入并启动 TeachingUbuntu，虚拟磁盘实际位于 D 盘。
- Linux uv 环境安装 96 个运行依赖，兼容性检查通过。
- WSL 中的四模型 GPU 推理、实际 OCR、gRPC 调用均通过；PyTorch 显存分配峰值约 1751MB。
- WSL Python 33 项测试和 Go 测试通过，包含真实 Chromium 浏览器测试。
- Linux 前端 8 项测试及 Vite 构建通过。
- MySQL、Redis、Kafka、Worker、AI、Go、前端均已启动，Windows 访问前端返回 HTTP 200。

Go 测试中需要专用数据库的可选集成测试未开启；已通过实际应用接口验证本机数据库与业务链路。

**服务已启动：访问 `http://127.0.0.1:6006`。** 当前业务验收结果保存在
`.runtime/deployment-check.json`，部署阶段保存在 `.runtime/deployment-status.json`。
Windows 重启后需再次执行启动命令，目前没有配置开机自启动。真实作业批改、后台池化、
试卷 OCR 评分、Excel 导出、Go 主助手对话，以及停止再启动后的 JWT、成绩和任务状态
保留均已通过。模拟作业、试卷及会话已清理；用户数据保留。

用户已填写 DeepSeek 密钥，真实评分与 Go 主助手调用均已通过。报告仅记录密钥是否配置，不记录密钥。
真实教务平台登录及验证码需要使用者在控制台中完成，下载流程仍需在人工登录后验证。

## 磁盘布局

| 内容 | Windows 路径或对应位置 |
| --- | --- |
| 项目源码、前端依赖和构建结果 | `D:\dzq\LLM_TEACH_ASSISTANT` |
| 已有模型 | `D:\dzq\models` |
| Windows uv、Go 工具 | 项目 `.runtime\tools` |
| uv 管理的 Windows Python | 项目 `.runtime\python-windows` |
| Windows Python 验证环境 | 项目 `.runtime\python-env-windows` |
| Windows 包、模型、Go 和 npm 缓存 | 项目 `.runtime\cache` |
| 安装包与安装日志 | 项目 `.runtime\bootstrap` |
| Ubuntu 虚拟磁盘目标 | 项目 `.runtime\wsl\TeachingUbuntu` |
| WSL 交换文件 | 项目 `.runtime\wsl\swap.vhdx` |
| Linux 工具、uv 环境、数据库、Kafka 和日志 | Ubuntu 虚拟磁盘内，物理文件在 D 盘 |

WSL 系统组件属于 Windows 软件，仍由 Windows 安装到系统目录；用户目录下的
`.wslconfig` 是一个小型配置文件。大型发行版、交换文件、环境和缓存均指定到 D 盘。
Linux 应用运行目录为 `/opt/teaching-runtime`，数据库系统目录也在同一个 D 盘虚拟磁盘内。
当前 WSL 配置限制为 20GB 内存、12 个逻辑处理器和 8GB 交换空间。

## 填写密钥

编辑 `ai_engine_python\app\.env`：

```dotenv
DEEPSEEK_API_KEY=你的密钥
```

该文件已被 Git 忽略。`TEACH_MODELS_DIR=../models` 按**项目根目录**解析，
不随当前工作目录变化。Windows 下解析为 `D:\dzq\models`，WSL 中解析为
`/mnt/d/dzq/models`。启动脚本会读取 `.env` 并将配置传给 Go、Python 和 Worker。
数据库凭证、JWT 与 RPA 内部令牌由部署/启动脚本生成并保存到运行目录，勿提交到 Git。

## 重新准备完整环境

从项目根目录的 PowerShell 执行：

```powershell
Set-Location D:\dzq\LLM_TEACH_ASSISTANT
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\deploy.ps1
```

脚本复用名为 `TeachingUbuntu` 的独立实例；首次执行时从已校验的 Ubuntu 镜像导入，
不删除或覆盖其他发行版。随后在该实例中安装 MySQL、Redis、Java 21、Kafka 4.1.2、Go、Node 和 uv，
创建项目数据库与专用数据库账号，并准备 Python、浏览器和前端依赖。
依赖准备后会执行本地模型、OCR、RPC 和真实 Chromium 测试，通过后启动完整服务。
脚本不会重启 Windows。首次执行可能需要较长下载时间，可查看输出定位失败步骤。

部署脚本已在本机成功执行，完成 Linux 安装、本地模型与浏览器测试及服务启动。
已有应用运行时，应先妥善结束浏览器任务并执行停止脚本，再重新部署。

**注意 Windows 与 Linux 的 `node_modules` 不能混用。** 部署脚本的 Linux `npm ci`
会重新生成前端依赖，以获得 Linux 平台的 esbuild/Rollup 二进制。
Windows Python 验证环境与 Linux uv 环境也各自独立，不能复制虚拟环境替代安装。

## 启动与停止

Windows 可直接双击项目根目录下的 `start.bat` 启动、`stop.bat` 停止。
脚本会自动定位项目目录，窗口显示执行结果后等待按键关闭。
在 PowerShell 中也可以执行：

```powershell
Set-Location D:\dzq\LLM_TEACH_ASSISTANT
.\start.bat
.\stop.bat
```

命令行调用时如不需要等待按键，可以使用 `.\start.bat --no-pause` 或
`.\stop.bat --no-pause`。停止前请结束正在运行的批改或浏览器任务。

这两个命令文件调用已有 PowerShell / WSL 服务脚本；也可直接执行底层命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\stop.ps1
```

完整启动成功后访问 `http://127.0.0.1:6006`，在页面注册自己的账号。启动顺序为基础服务、Kafka、RPA Worker、
Python AI、Go 和前端。停止脚本停止它记录的应用进程及 Kafka，保留 MySQL/Redis。
启停脚本会核对服务工作目录、进程命令及启动时间；RPA 还会核对单实例锁，旧进程尚未退出时拒绝重复启动。
服务端口就绪后才保存正式 PID，启动失败不会留下失效的正式 PID。停止时先通知主进程清理子进程，
默认等待 30 秒；超时后只强制结束已核实身份的服务进程组。`STOP_TIMEOUT` 可调整等待秒数。
RPA 会将未结束的浏览器任务标记为中断，后台请求的优雅退出等待为 10 秒，任务清理等待最多 8 秒。
若日志出现运行锁占用，先执行停止命令再启动；不要删除 `rpa-worker.lock` 文件来绕过检查。
日志在 WSL 的 `/opt/teaching-runtime` 中。
例如在 PowerShell 执行：

```powershell
wsl -d TeachingUbuntu -u teacher -- tail -n 50 /opt/teaching-runtime/backend.log
```

修改 `.env` 后执行停止和启动，服务会重新读取配置。

本机默认只监听回环地址。Kafka 与 gRPC 地址现可分别用 `KAFKA_BROKERS`、
`AI_GRPC_ADDR` 配置。初始批改并发为 2，gRPC 线程为 4，RPA 并发为 2；
批量性能及资源峰值需要在完整链路验收后测量。

## 本次验证的深度学习组合

| 组件 | 本机 Windows 与 WSL 验证结果 |
| --- | --- |
| Python | uv 管理的 3.11.17 |
| GPU / 驱动 | RTX 5080 Laptop，16GB 显存；592.01 |
| PyTorch | 2.8.0+cu128，GPU 计算能力 12.0 |
| Transformers | 4.55.0 |
| PaddlePaddle / PaddleOCR | 3.2.1 CPU / 3.2.0 |
| gRPC / Protobuf | 1.78.0 / 6.33.6，生成代码导入通过 |

PyTorch 从显式 CUDA 12.8 索引安装，CUDA 相关依赖由 uv 解析，不手工混装原清单里的
NVIDIA 包。Paddle 先采用 CPU 版本。Windows 验证中发现 Paddle 先导入会引发 DLL 冲突，
已调整为先初始化 Torch，OCR 与实际 gRPC 调用随后通过。
WSL 中也已通过四模型推理和 CUDA 矩阵计算；WSL 使用 Windows 显卡驱动，不安装 Linux 显卡驱动。
OCR 使用 CPU，Torch 的 GPU 显存分配峰值不代表整个系统的显存总占用。

本地验证报告：项目 `.runtime\models-check-windows.json`、
`.runtime\ocr-check-windows.json`、`.runtime\grpc-check.json`。
WSL 报告：`.runtime\models-check-wsl.json`、`.runtime\grpc-check-wsl.json`。
测试图片和 gRPC 验证日志也在 `.runtime`，检查不调用 DeepSeek。

## 部署中解决的实际问题

- WSL 首次启用需要 Windows 重启；重启后虚拟机服务已正常生效。
- Windows 的本地代理在 NAT 模式下无法正常使用；已配置 mirrored 网络、自动代理及 DNS
  tunneling，内部服务加入 `NO_PROXY`。当前安装下载和云端调用均已连通。
- Apache 的 SHA512 文件按 8 位分组显示；已修正解析并校验 Kafka 安装包。
- Kafka 4 与 Sarama 默认的旧协议不兼容；生产者及消费者统一读取
  `KAFKA_VERSION=4.1.0`，地址读取 `KAFKA_BROKERS`。
- 成绩池化的数组要求与 DeepSeek JSON 对象模式冲突；现在要求
  `{"pooled_results": [...]}`，Python 校验学生名单和数值后返回 RPC 数组，保持 Go 契约。
  针对格式错误、重复或额外学生、无效分数的回归测试已通过。
- 前端依赖按原锁文件安装，npm 报告已有依赖安全告警，未进行可能破坏功能的批量升级。
  当前服务仅供本机使用，公开部署前需另行处理依赖告警和访问控制。

## 完整验收清单

本机部署已完成；真实教务下载需要人工登录后继续业务验证：

- [x] WSL 四模型 GPU 推理、CUDA 计算、实际 OCR。
- [x] MySQL、Redis、Kafka 运行；作业与试卷队列的实际生产/消费及结果入库。
- [x] Worker 的 `browser.v1`、`go-eino`、`browser.stream.v1` 协议和真实 Chromium 测试。
- [x] 注册、登录、令牌刷新、ZIP 批改、试卷 OCR 评分和 Excel 导出。
- [x] DeepSeek 真实评分、成绩池化和 Go 主助手对话。
- [x] 停止、再次启动，JWT、成绩及异步任务状态保留。
- [ ] 使用者完成教务平台登录和验证，检查真实下载及下载后批改。

业务验收报告为 `.runtime/deployment-check.json`，导出样例为
`.runtime/deployment-export.xlsx`，验收日志为 `.runtime/deployment-acceptance.log`。
日志包含修复前的一次失败和修复后的成功，最终报告的 `ok` 为 `true`。

Python 后续还增加并通过了 2 项成绩池化回归测试。`scripts/verify-wsl.sh` 会执行全部
本地检查，其 gRPC 检查要求 50051 端口空闲，应在应用停止后运行。
`scripts/check-deployment.py` 检查运行中的应用，创建模拟数据并调用真实 DeepSeek；
`--check-restart` 会停止并再次启动服务，仅在没有真实任务运行时使用。

相关官方资料：[uv 与 PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)、
[PyTorch 安装矩阵](https://pytorch.org/get-started/previous-versions/)、
[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)、
[WSL 命令](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)。
