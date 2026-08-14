# GOAI 2026 项目规范

GOAI 2026 通用双臂协同操作挑战赛参赛项目。

## 本地环境

macOS，无 GPU，磁盘 63GB。用于代码编写、结果数据分析，conda lerobot环境可用；所有训练和评测在远程服务器执行。

### 项目目录结构

```
goai_2026/
├── CLAUDE.md              # 项目规范（本文件）
├── RoboDojo/              # RoboDojo 代码仓库（submodule）
├── docs/                  # 文档
├── data/                  # 比赛数据示例（.gitignore）
├── eval_results/          # 仿真评测结果及分析（.gitignore）
├── papers/                # 参考论文（.gitignore）
├── notebooks/             # Jupyter notebooks（.gitignore）
├── outputs/               # 分析测试结果输出（.gitignore）
├── scripts/               # 工具脚本
└── utils/                 # Python 工具库
```

## 常用文档

| 文档                                                     | 说明                             |
| -------------------------------------------------------- | -------------------------------- |
| [docs/私人hf仓库操作指南.md](docs/私人hf仓库操作指南.md) | 私人 HF 仓库，用于存放训练的模型 |
| [docs/远程服务器详情.md](docs/远程服务器详情.md)         | 服务器配置、目录结构、操作规范   |
| [docs/仿真环境配置指南.md](docs/仿真环境配置指南.md)     | 训练数据下载、仿真环境配置       |
| [docs/仿真测试指南.md](docs/仿真测试指南.md)             | Isaac 仿真评测                   |
| [docs/策略服务启动指南.md](docs/策略服务启动指南.md)     | 策略服务启动规范与命令           |



## 铁律

- 在进入服务器进行操作前，请阅读 [docs/远程服务器详情.md](docs/远程服务器详情.md)（服务器操作规范文档），务必按照文档规范操作。
- 进行仿真评测前，请阅读 [docs/仿真测试指南.md](docs/仿真测试指南.md)，务必按照文档规范操作。
- 启动策略服务务必阅读 [docs/策略服务启动指南.md](docs/策略服务启动指南.md)，务必按照文档规范操作。
- 服务器中日志统一存放于 `/data/outputs` 下，且日志名称末尾需要带时间戳。
- 如在服务器上无法拉取私人仓库，可使用本地 `~/Documents/token/github` 中 token，切忌明文使用。
- 本地修改代码、文档请及时提交并推送仓库
