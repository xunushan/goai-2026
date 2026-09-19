# pi0.5 迁移包（pi05_portable）— 换新机器直接用

> 生成日期：2026-09-08 ｜ 源机：`train-4090`（RTX4090 24G）｜ 目标机：RTX 5090 32G（Blackwell, 可联网）
> 本包 = **数据(real+sim) + base 模型 + 代码 + venv 环境 + 运维脚本 + 迁移说明**（≈24G，含 env 9.5G）。
> 迁移策略：**优先复用打包的 venv**（快）；若新机上复用失败（路径/驱动不满足）→ **联网重装**兜底。

## 0. 目录结构（`data/` 下即原机 `/data` 的子树；镜像还原到 `/data` 则所有绝对路径原样有效）

```
pi05_portable/
├── data/
│   ├── goai/envs/                            # ★ venv 环境（含 uvpy base python，9.5G）
│   │   ├── pi05_l060/                        #   py3.12.14 + torch2.10.0+cu128 + lerobot0.6.0 ...
│   │   └── uvpy/cpython-3.12.14-.../          #   base python（venv 的 home 指向它，必须一起搬）
│   ├── checkpoints/pi05_base/                 # pi05 基础权重（orbax/ocdbt，12G）
│   ├── data/sim_lerobot_v30_joint-224x224/    # RoboDojo sim 数据集（621M，300ep dual_x5；冒烟/goai 用）
│   ├── data/real_lerobot_v30_joint-224x224/   # real 数据集（2.2G；当前 goai_pi05 配置不用，为后续真机数据实验备用）
│   ├── openpi/                                # 训练 fork 代码（含 .git + assets 归一化资产 + 冒烟仪表）
│   └── pi05_env_scripts/                      # 运维：setup/verify/openpi_env/lock + run_smoke 等
├── restore_to_new.sh                          # 新机还原到 /data 的脚本
└── README.md
```

## 1. 搬运（二选一）

**A. 整盘搬**：`/cloud/cloud-ssd1` 摘下挂到新机（路径建议保持 `/cloud/cloud-ssd1`）。

**B. 网络 rsync**（新机可达后执行一次，~24G）：
```bash
rsync -a --info=progress2 /cloud/cloud-ssd1/pi05_portable/ <新机>:/cloud/cloud-ssd1/pi05_portable/
```

## 2. 新机还原：优先镜像到相同绝对路径 `/data`
```bash
# root 执行；/data 需 ≥ ~25G 余量
bash /cloud/cloud-ssd1/pi05_portable/restore_to_new.sh      # = rsync data/ 子树 → /data/
```
**为什么推荐 /data**：venv 是"指针结构"——`pyvenv.cfg home=` 与 `bin/python` 软链指到
`/data/goai/envs/uvpy/...`，`fflib/*` 软链是 `/data/goai/envs/pi05_l060/.../av.libs/...` 的绝对路径，
数据软链、`run_smoke.sh` 里 /data/checkpoints、/data/data、/data/openpi 也都是绝对路径。
**镜像到 /data 后全部原样命中 → env 可直接复用、无需任何修改。**

## 3. venv 复用的条件与操作（若第 2 步路径一致 → 这节基本可跳过）

复用前提（缺一不可，任何一个不满足就跳到 §4 重装）：
1. **base(uvpy) 与 venv 一起搬**且绝对路径不变（即镜像到 /data）。变了需改 `pyvenv.cfg home=` 与 `bin/python` 软链（见下 relocate 片段），并重建 fflib。
2. **NVIDIA 驱动支持 CUDA ≥ 12.8**（torch/jax 是 cu128 轮子）。5090 出厂驱动满足。
3. **jax 0.5.3 cu12 在 Blackwell(sm_120) 能跑**（需实测，见 §5）。

还原后验证 env（不需重装，跑验收即可，应 ~1 分钟内全绿）：
```bash
# 先确保 fflib 软链有效（路径一致则天然有效；不放心可幂等跑一次 setup 重建）
GOAI_ROOT=/data/goai DATASET_ROOT=/data/data OPENPI_SRC=/data/openpi bash /data/pi05_env_scripts/verify_env.sh
# [1/3] 版本+import  [2/3] 本地解码  [3/3] openpi data_loader + 1 batch
```

若挂载点**不是** /data、必须换路径（relocate 一次性，两处指针 + 重建 fflib）：
```bash
# NEWROOT = env 新的上一级（示例：若环境最后落在 /cloud/goai/envs/pi05_l060，则 NEWROOT=/cloud/goai）
NEWROOT=/你的/goai
ENV=/你的/goai/envs/pi05_l060; OLD=/data/goai
# ① pyvenv.cfg home 改前缀
sed -i "s|^home = .*|home = $NEWROOT/envs/uvpy/cpython-3.12-linux-x86_64-gnu/bin|" $ENV/pyvenv.cfg
# ② bin/python 重新软链到新 base
rm -f $ENV/bin/python
ln -s $NEWROOT/envs/uvpy/cpython-3.12-linux-x86_64-gnu/bin/python3.12 $ENV/bin/python
# ③ 重建 fflib + 自检：GOAI_ROOT=$NEWROOT bash <pi05_env_scripts>/setup_pi05_l060_env.sh gpu
#    （setup 幂等，会重建 fflib 软链并跑自检；若它因 bin/python 可用而跳过创建，恰好就是我们要的复用）
```
> ⚠️ 若 relocate 后 `import torchcodec` 报 libav 找不到 = fflib 没重建成功 → 改回走 §4 重装更省事。

## 4. 兜底：新机联网重装 env（复用失败/嫌 relocate 麻烦时的首选替代，5–15 分钟）
```bash
# uv（无则装）
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
# 一键装（GOAI_ROOT 指向你放 env 的根；推荐 /data/goai 与包内路径一致）
cd /cloud/cloud-ssd1/pi05_portable/data/pi05_env_scripts
GOAI_ROOT=/data/goai nohup bash setup_pi05_l060_env.sh gpu > env_setup.log 2>&1 < /dev/null &
# 轮询日志到 “== 完成 ==”。幂等；耗时大头是 torch cu128(~4GB) 下载，已实测约 5–15 分钟。
GOAI_ROOT=/data/goai DATASET_ROOT=/data/data OPENPI_SRC=/data/openpi bash /data/pi05_env_scripts/verify_env.sh
```

## 5. Blackwell（RTX 5090 / sm_120）专项核验 ⚠️（复用与重装都要做）
```bash
export LD_LIBRARY_PATH=/data/goai/envs/pi05_l060/fflib:/data/goai/envs/pi05_l060/lib/python3.12/site-packages/av.libs
/data/goai/envs/pi05_l060/bin/python -c "import jax; print(jax.devices())"
# 期望 [CudaDevice(id=0)]。若报不支持/编译错 → jaxlib 需升到支持 sm_120 的版本（届时按报错处理）。
```
再跑 S1 冒烟 1–2 步确认端到端（脚本见 `data/pi05_env_scripts/run_smoke.sh`，流程见技能 pi05-smoke-test）。

## 6. 数据软链（训练用户下，一次；镜像到 /data 后目标存在）
```bash
mkdir -p ~/.cache/huggingface/lerobot
ln -sfn /data/data/sim_lerobot_v30_joint-224x224 ~/.cache/huggingface/lerobot/RoboDojo_sim_arx-x5_v30
```

## 7. 32G vs 24G 预期（来自 24G 冒烟推算，未在 5090 实测）
| 组合 | 24G 实测 | 32G 池28.8G 推算 |
|---|---|---|
| P1 noEMA | bs1 | bs4–5 |
| P0 noEMA | bs1 | bs6–7 |
| P1 + EMA | ✗ OOM | bs1–2 |
| P0 + EMA | bs1 | bs3–4 |
- load 串行停顿随显存余量大概率消除（9.2→~5s/步），放大物理 batch 再摊薄——需 5090 重跑冒烟定标。

## 8. 相关文档（本机 goai_2026/docs/，不入包）
`pi0.5_P0P1冒烟测试结论_24G.md` / `pi0.5混合参数微调与全量微调对比方案.md` / `pi05_l060_环境配置与使用.md`
