# lerobot 数据加载基准与调试代码（规范副本）

数据加载性能试验的**规范代码副本**（2026-09-02 从 `outputs/lerobot_bench/` 整理而来，
原目录是 gitignored 的运行区，保留全部结果 JSON/日志作为归档）。所有结论与数值见
[docs/LeRobot_v3数据加载基准与结论汇总.md](../../docs/LeRobot_v3数据加载基准与结论汇总.md)，
本目录只放**数据加载试验代码**，不含临时改动编码方式的脚本（那些仍在 `outputs/lerobot_bench/`）。

> 以下脚本原本都在服务器（train 117.50.173.12）上执行，路径为当时的环境。
> 服务器当前已关机；重启后需按各自 env 的 `LD_LIBRARY_PATH` / Python 解释器核对后再跑
> （driver 脚本头部注释已写明各试验用的 env 与 torchcodec FFmpeg 库路径）。

## 目录结构

```
lerobot_data_bench/
├── README.md                 # 本文件
├── bench/                    # 端到端 DataLoader 基准 + 聚合
│   ├── bench_v2.py           # ★ 统一条件基准（真实训练 sampler/collate，跨 0.4.4/0.6.0）
│   ├── bench_data_loader.py  # gen1 初筛 harness（顺序 sampler，已被 bench_v2 取代，仅保留对照）
│   ├── smoke_224.py          # 224 子集 + torchcodec + return_uint8 冒烟（本地数据集短路 patch）
│   └── analyze_results.py    # 聚合 JSON -> markdown 表（中位数，从 p07 版改名而来）
├── profile/                  # 开销分解 / 纯解码上限 / Lance 探针
│   ├── profile_getitem.py    # 0.4.4 单样本 __getitem__ 分段计时（base/action/video）
│   ├── profile_060_getitem.py# 0.6.0 单样本 get_item 分段计时（base/action/query-ts/video/task）
│   ├── microbench_060.py     # 0.6.0 两个疑点微测：action 50 行整块读 vs 逐行；线程池复用
│   ├── phase3_bench.py       # 纯解码吞吐（torchcodec approximate-seek，500 随机样本/3 相机）
│   ├── probe_lance_compare.py# Lance 真值：纯 ds[idx] 随机 vs DataLoader(shuffle=True)
│   └── profile_lance_random.py # Lance 随机访问 __getitem__ 耗时分解
├── correctness/              # 语义一致性（验证性能优化不改变训练输入）
│   ├── check_backend_equiv.py   # pyav ↔ torchcodec 逐像素等价（0.4.4）
│   ├── check_uint8_l060.py      # float ↔ uint8 等价（0.6.0 + torchcodec）
│   └── check_uint8.py           # float ↔ uint8 等价（LanceDBDataset）
└── drivers/                  # 服务器批量运行入口（各试验矩阵，输出到 /data/outputs 带时间戳日志）
    ├── run_matrix51.sh       # 0.4.4 pyav 基线 + torchcodec nw=0/2/4  → out/mat51
    ├── run_matrix54.sh       # 0.6.0 torchcodec float vs uint8        → out/mat54
    ├── run_phase1_cache.sh   # 0.4.4 有界 LRU decoder cache 16/64/256  → out/phase1
    ├── run_06cache.sh        # 0.6.0 LEROBOT_VIDEO_DECODER_CACHE_SIZE   → out/p06cache
    ├── run_224p3.sh          # 0.6.0 224 vs 640（eps46-62）×float/uint8 → out/p06_224
    ├── run_phase3.sh         # 224 AV1 gop2/10 重编码 + 纯解码 + 质量   → p3/
    ├── launch_0609.sh        # item1(cache)+item2(224) 顺序执行入口
    ├── run_v44.sh / run_v60.sh / launch_p07.sh  # sim h264 640/224 全矩阵 → out/p07
```

## 分类说明（与用户整理要求的对应）

| 类别 | 脚本 | 对应试验 |
|---|---|---|
| 端到端基准 | `bench_v2.py` | matrix5.1/5.4、phase1 cache、0.6.0 item1/2、p07 sim 全矩阵 |
| 纯解码上限 | `phase3_bench.py` | 阶段三 224 AV1、sim h264 decode-only |
| getitem 分解 | `profile_getitem.py` / `profile_060_getitem.py` / `microbench_060.py` | action/state 非解码瓶颈归因 |
| Lance 访问真值 | `probe_lance_compare.py` / `profile_lance_random.py` | Lance 随机访问劣化结论 |
| 一致性校验 | `correctness/*.py` | uint8 不改变数据 / backend 不改变数据 |
| 驱动 | `drivers/*.sh` | 各试验矩阵入口 |

## 有意未纳入的脚本（临时编码/编码侧分析，仍保留在 `outputs/lerobot_bench/`）

按整理要求**排除**的是改变或评估编码方式的脚本，不属于"加载试验"：
`phase3_reescale.py`（AV1 重编码）、`phase3_quality.py`（PSNR/SSIM）、`phase3_viz.py`（画面对比）、
`phase2_video_stats.py`（码率/GOP 统计）。
若之后做编码侧验证，请去原目录取，勿与加载代码混用。

## 结果归档位置

原始 JSON / 日志（gitignored，本地镜像 + 服务器）：

| 试验 | JSON | 汇总 |
|---|---|---|
| gen1 初筛 | `outputs/lerobot_bench/out/result_l044_* / test1b_* / test2_* / test3_*` | [benchmark_report.md](../../outputs/lerobot_bench/benchmark_report.md) |
| matrix5.1/5.4 | `outputs/lerobot_bench/out/mat51/ out/mat54/` | [benchmark_report_v2.md](../../outputs/lerobot_bench/benchmark_report_v2.md) |
| phase1 cache（0.4.4） | `outputs/lerobot_bench/out/phase1/` | 同上 |
| 0.6.0 item1/2 | 服务器 `out/p06cache/ out/p06_224/`（本地未镜像） | [benchmark_report_item2_0609.md](../../outputs/lerobot_bench/benchmark_report_item2_0609.md) |
| 阶段三 | 服务器 `p3/`，本地 `p3_viz/` | [benchmark_report_phase3.md](../../outputs/lerobot_bench/benchmark_report_phase3.md) |
| p07 sim h264 | `outputs/lerobot_bench/p07/out/`（18 JSON） | [benchmark_report_p07_sim.md](../../outputs/lerobot_bench/p07/benchmark_report_p07_sim.md) |

## 运行注意（踩坑清单）

- **torchcodec 加载**：必须先 `import torchcodec` 再 import lerobot/PIL/cv2（libjpeg soname 冲突，
  见 memory `torchcodec-libjpeg-conflict`）；`LD_LIBRARY_PATH` 需含对应 env 的 FFmpeg 库。
- **0.6.0 读本地数据集**：`get_safe_version` 会打 hub → `bench_v2.py` 已做短路 patch。
- **0.6.0 conda env（lerobot060）无 psutil**：`bench_v2.py` 的 RSS 走 `getrusage` 兜底。
- 0.4.4 的 `LeRobotDataset` **无 `return_uint8` 参数**，该维度只在 0.6.0 上测。
- sim h264 数据集的 `--num-workers` 只测了 0（torchcodec nw≥2 已证内存爆炸，见 matrix5.1）。
- `--preload-action`：item4 方案 1 原型（action 整列内存预载），lerobot0.6；覆写 = 静态模块级
  `_ActionPreloadReader(DatasetReader)` + `reader.__class__` 交换（真 spawn-picklable，可跑 DataLoader nw>0）。
  自带逐位一致性哨兵（默认 vs 预载路径，horizon 自适应选行），ALL OK 后才计时。
  同会话 2×2 实测（sim224+uint8，机器有负载）：baseline nw0 88 / nw2 156；preload nw0 116 / nw2 211 sps；
  nw2 在 0.6 env 不炸内存（RSS 2.4–2.9GB）。详见表单文档试验 L。
- `--resize`/编码相关参数不在此代码中（走 XPolicyLab process_data 转换链，见项目 memory）。
