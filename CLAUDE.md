# GOAI 2026 项目规范

GOAI 2026 通用双臂协同操作挑战赛参赛项目。

## 本地环境

macOS，无 GPU，磁盘 63GB。用于代码编写、结果数据分析，所有训练和评测在远程服务器执行。

### 项目目录结构

```
goai_2026/
├── CLAUDE.md              # 项目规范（本文件）
├── RoboDojo/              # RoboDojo 代码仓库（submodule）
├── docs/                  # 文档（.gitignore）
├── data/                  # 比赛数据示例（.gitignore）
├── eval_results/          # 仿真评测结果及分析（.gitignore）
├── papers/                # 参考论文（.gitignore）
├── notebooks/             # Jupyter notebooks（.gitignore）
├── outputs/               # 分析测试结果输出（.gitignore）
├── scripts/               # 工具脚本
└── utils/                 # Python 工具库
```

## 远程服务器操作纪律

- **policy-server**：代码统一通过 git 获取和更新，禁止手动修改或非 git 方式更新代码
- **issac-server**：未经授意确认，禁止更新代码、禁止在虚拟环境下执行安装命令（如 `pip install`、`conda install` 等）

## 远程服务器操作指南

详见 [docs/远程服务器详情.md](docs/远程服务器详情.md)，包含 policy-server 和 issac-server 的配置、目录结构和操作规范。

## 自训模型上传下载

在远程服务器上上传和下载自训模型，HF token 已配置。详见 [docs/模型上传下载指南.md](docs/模型上传下载指南.md)。

**关键规则：远程服务器上模型统一下载到 `/data/checkpoints/` 目录下。**

## 训练数据下载

详见 [docs/仿真环境配置指南.md](docs/仿真环境配置指南.md) 中「LeRobot ee 数据下载」章节。

**下载数据文件夹放置于远程服务器 `/data` 目录下。**

## Isaac 仿真评测

详见 [docs/仿真测试指南.md](docs/仿真测试指南.md)。

**注意：在 `RoboDojo/XPolicyLab/policy/` 下新增模型文件夹后，需要同步到 issac-server 服务器对应目录。**
