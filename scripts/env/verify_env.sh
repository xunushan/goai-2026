#!/usr/bin/env bash
# =============================================================================
# verify_env.sh — pi05_l060 env 的验收冒烟（setup 之后必跑；与本文档验证同源）
#   检查 1 : 关键版本 + lerobot.datasets import（transformers↔hub 冲突未复发）
#   检查 2 : LeRobotDataset(torchcodec) 本地数据集实际解码（3 索引）
#   检查 3 : openpi data_loader import + create_torch_dataset + 1 个 collate batch
#
# 用法
#   bash verify_env.sh                # auto(同 setup)
#   bash verify_env.sh gpu|cpu
#   OPENPI_SRC=/data/openpi bash verify_env.sh   # 覆盖 openpi 源码位置(默认自动探测)
# =============================================================================
set -euo pipefail
ENV_NAME="${ENV_NAME:-pi05_l060}"
ENV_DIR="/cloud/cloud-ssd1/goai/envs/$ENV_NAME"
PY="$ENV_DIR/bin/python"
SP="$ENV_DIR/lib/python3.12/site-packages"
DATASET_ROOT="${DATASET_ROOT:-/cloud/cloud-ssd1/lerobot_data}"
REPO="${REPO:-sim_lerobot_v30_joint-224x224}"

# openpi 源码位置（用户放 /data/openpi）
OPENPI_SRC="${OPENPI_SRC:-}"
if [ -z "$OPENPI_SRC" ] && [ -d "/data/openpi/src/openpi" ]; then
  OPENPI_SRC="/data/openpi"
fi
PP=""
[ -n "$OPENPI_SRC" ] && PP="$OPENPI_SRC/src:$OPENPI_SRC/packages/openpi-client/src"
export LD_LIBRARY_PATH="$ENV_DIR/fflib:$SP/av.libs"
[ -n "$PP" ] && export PYTHONPATH="$PP"   # 让 openpi 可 import（openpi 不装进 venv，走源码挂载）
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

echo "== verify_env.sh | env=$ENV_DIR | openpi=$OPENPI_SRC =="

echo; echo "-- [1/3] 版本 + lerobot.datasets import"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PYTHONPATH= "$PY" - <<'PY'
import torchcodec  # noqa
import torch, numpy, huggingface_hub as hub, transformers, lerobot, jax
print(f"  torch {torch.__version__} | jax {jax.__version__}({jax.default_backend()}) "
      f"| numpy {numpy.__version__} | hub {hub.__version__}")
print(f"  transformers {transformers.__version__} | lerobot {lerobot.__version__}")
import lerobot.datasets.lerobot_dataset
print("  lerobot.datasets.lerobot_dataset import OK")
PY

echo; echo "-- [2/3] 本地数据集解码 (repo=$REPO, torchcodec)"
timeout 300 "$PY" - "$DATASET_ROOT" "$REPO" <<'PY'
import sys, torchcodec  # noqa
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root, repo = sys.argv[1], sys.argv[2]
full = f"{root}/{repo}"
ds = LeRobotDataset(repo_id=repo, root=full, delta_timestamps={"action": [0.0]}, video_backend="torchcodec")
print(f"  len={len(ds)}")
for i in (0, 137, len(ds) - 1):
    s = ds[i]
    act, st = s["action"], s["observation.state"]
    n_cam = sum(1 for k in s if k.startswith("observation.images."))
    print(f"  idx={i} action{tuple(act.shape)} state{tuple(st.shape)} cams={n_cam}")
PY

if [ -n "$OPENPI_SRC" ]; then
echo; echo "-- [3/3] openpi data_loader 集成（create_torch_dataset + collate batch）"
timeout 500 "$PY" - "$DATASET_ROOT" "$REPO" <<'PY'
import sys, torch, torchcodec  # noqa
root, repo = sys.argv[1], sys.argv[2]
from openpi.training import config as _config
from openpi.training import data_loader as dl
dc = _config.DataConfig(repo_id=f"{root}/{repo}", video_backend="torchcodec",
                        action_sequence_keys=("action",))
ds = dl.create_torch_dataset(dc, action_horizon=50, model_config=None)
print(f"  create_torch_dataset len={len(ds)}")
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False,
                                     collate_fn=dl._collate_fn, num_workers=0)
b = next(iter(loader))
cams = [k for k in b if k.startswith("observation.images.")]
print(f"  batch: action{tuple(b['action'].shape)} state{tuple(b['observation.state'].shape)} cams={cams}")
assert tuple(b["action"].shape) == (8, 50, 14), b["action"].shape
print("  openpi data_loader consumed lerobot0.6 dataset: OK")
PY
else
  echo; echo "-- [3/3] 跳过：未找到 openpi 源码（设 OPENPI_SRC=/data/openpi 后重跑）"
fi

echo; echo "== verify_env.sh 全部通过 =="
