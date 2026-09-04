#!/usr/bin/env bash
# =============================================================================
# setup_pi05_l060_env.sh — 数据盘统一环境「pi05 忠实栈 + lerobot0.6」的构建/修复入口
#
# 目标 env : /cloud/cloud-ssd1/goai/envs/pi05_l060
#   内容   : py3.12.14 + lerobot 0.6.0，其余逐版本对齐 pi05_openpi（numpy1.26.4 /
#            hub0.35.3 / transformers4.53.2 / jax0.5.3+cuda12 / flax0.10.2 ...）
#   依据   : scripts/env/pi05_l060_lock.txt（已验证 clone 的精确 uv freeze snapshot）
#   运行   : 服务器上执行（train）；数据盘 /cloud/cloud-ssd1 跨重启保留
#
# 用法
#   bash setup_pi05_l060_env.sh            # 自动探测：nvidia-smi 可用=gpu，否则=cpu
#   bash setup_pi05_l060_env.sh gpu        # 强制 GPU 版（torch 2.10.0+cu128，训练用）
#   bash setup_pi05_l060_env.sh cpu        # 强制 CPU 版（torch 2.10.0+cpu，调试/无卡）
#   bash setup_pi05_l060_env.sh gpu --fresh  # 从零重建 venv（默认幂等修复/复用）
#
# 说明
#   - 幂等：env 已存在则跳过 venv 创建，仅按需补齐/修复 + 切换到当前模式的 torch。
#     数据盘持久 → 「启动 gpu 重配」多数情况只需 gpu 模式跑一遍 + verify_env.sh。
#   - lerobot0.6 的 datasets/__init__ 会急切 import transformers；transformers4.53.2
#     要求 hub<1.0 —— 故 hub 必须锁 0.35.3（lock 已含），勿被 resolver 升到 1.x。
#   - fflib = av==15.1.0 wheel 内哈希 .so 的 soname 软链，供 torchcodec dlopen；
#     锁死 av 版本后哈希不变，映射表内嵌，无需手工。
# =============================================================================
set -euo pipefail

# ---------- 配置 ----------
GOAI_ROOT="${GOAI_ROOT:-/cloud/cloud-ssd1/goai}"   # 统一目录根（envs/ src/ 在此）
UVPY_DIR="${UVPY_DIR:-/cloud/cloud-ssd1/goai/envs/uvpy}"   # base python 安装目录（uv 管理，与 venv 同放 envs/）
PYVER=3.12.14
ENV_NAME="${ENV_NAME:-pi05_l060}"
ENV_DIR="$GOAI_ROOT/envs/$ENV_NAME"
BASE="$UVPY_DIR/cpython-${PYVER}-linux-x86_64-gnu/bin/python${PYVER%.*}"
LOCK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$LOCK_DIR/pi05_l060_lock.txt"
TENCENT_MIRROR="https://mirrors.cloud.tencent.com/pypi/simple"

MODE="${1:-auto}"
[ "$MODE" = "auto" ] && { if command -v nvidia-smi >/dev/null 2>&1; then MODE=gpu; else MODE=cpu; fi; }
FRESH=0
for a in "$@"; do [ "$a" = "--fresh" ] && FRESH=1; done
case "$MODE" in gpu|cpu) ;; *) echo "MODE 须为 gpu|cpu|auto，收到: $MODE"; exit 2;; esac

echo "== setup_pi05_l060_env.sh | mode=$MODE fresh=$FRESH | env=$ENV_DIR =="

# ---------- 0) base python（uv python-build-standalone，自包含可跨机） ----------
echo "-- base python $PYVER ($UVPY_DIR)"
export UV_PYTHON_INSTALL_DIR="$UVPY_DIR"
uv python install "$PYVER" 2>&1 | tail -2 || true
[ -x "$BASE" ] || { echo "base python 不可用: $BASE"; exit 1; }

# ---------- 1) venv ----------
if [ "$FRESH" = 1 ]; then
  [ -d "$ENV_DIR" ] && rm -rf "$ENV_DIR"   # 仅删本 env 目录（受控于 GOAI_ROOT/envs 下）
fi
if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "-- 创建 venv"
  uv venv --python "$BASE" "$ENV_DIR"
else
  echo "-- 复用已有 venv $ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
SP="$ENV_DIR/lib/python${PYVER%.*}/site-packages"

# ---------- 2) 安装 lock（--no-deps 精确回放；跳过 torch/torchvision，按模式单装） ----------
echo "-- 回放 lock（排除 torch/torchvision，共 $(wc -l <"$LOCK") 行源）"
FILTERED=$(mktemp)
grep -vE '^(torch==|torchvision==|torch==|torchvision==|nvidia-|triton)' "$LOCK" > "$FILTERED" || true
# 排除行里不存在的包名残余清理（numpy/hub/transformers/lerobot 保留——lock 内已是正确 pin，
# --no-deps 下不会触发 resolver 冲突）
uv pip install --python "$PY" -r "$FILTERED" --no-deps \
  --index-url "$TENCENT_MIRROR" 2>&1 | tail -3 || { echo "lock 回放失败"; exit 1; }
rm -f "$FILTERED"

# ---------- 3) torch 按模式 ----------
case "$MODE" in
  gpu) TORCH_INDEX="https://download.pytorch.org/whl/cu128"; TAG="+cu128";;
  cpu) TORCH_INDEX="https://download.pytorch.org/whl/cpu";  TAG="+cpu";;
esac
echo "-- torch 2.10.0$TAG / torchvision 0.25.0$TAG  <- $TORCH_INDEX"
if [ "$MODE" = gpu ]; then
  # cu128 wheel 不含 CUDA 运行库，须让 resolver 拉 nvidia-*/triton（lock 已排除这些行）。
  # 若 --no-deps 单装会缺 libcublas → import torch 报 libcublas not found（2026-09-04 train-4090 实测）。
  uv pip install --python "$PY" --index-url "$TORCH_INDEX" \
    "torch==2.10.0$TAG" "torchvision==0.25.0$TAG" 2>&1 | tail -2
else
  # cpu wheel 自包含，--no-deps 精确回放不与 lock 冲突
  uv pip install --python "$PY" --no-deps --index-url "$TORCH_INDEX" \
    "torch==2.10.0$TAG" "torchvision==0.25.0$TAG" 2>&1 | tail -2
fi
# torchcodec 由 lock 回放时已装（pypi 0.10.0，与 FFmpeg 链接、构建无关）

# ---------- 4) fflib：soname 软链集（av==15.1.0 固定哈希，映射内嵌） ----------
make_fflib() {  # $1=av.libs 绝对路径, $2=输出目录
  local A=$1 OUT=$2
  mkdir -p "$OUT"
  while IFS='|' read -r soname hashed; do
    [ -z "$soname" ] && continue
    if [ -e "$A/$hashed" ]; then ln -sf "$A/$hashed" "$OUT/$soname"
    else echo "  [warn] $A/$hashed 缺失 → 跳过 $soname"; fi
  done <<'MAP'
libaom.so.3|libaom-170d518b.so.3.11.0
libasound.so.2|libasound-d5229d1a.so.2.0.0
libavcodec.so.61|libavcodec-7ee0753d.so.61.19.101
libavdevice.so.61|libavdevice-0a717e7d.so.61.3.100
libavfilter.so.10|libavfilter-7ceaa51a.so.10.4.100
libavformat.so.61|libavformat-f6caa08d.so.61.7.100
libavutil.so.59|libavutil-a63ffd27.so.59.39.100
libcrypto.so.1|libcrypto-bdaed0ea.so.1.1.1k
libdav1d.so.7|libdav1d-f1894f21.so.7.0.0
libdrm.so.2|libdrm-b0291a67.so.2.4.0
libgmp.so.10|libgmp-29b2ba5e.so.10.5.0
libgnutls.so.30|libgnutls-cd598300.so.30.40.3
libhogweed.so.6|libhogweed-033e28eb.so.6.10
libmp3lame.so.0|libmp3lame-68ba0ecb.so.0.0.0
libnettle.so.8|libnettle-a4970681.so.8.10
libogg.so.0|libogg-9af999c3.so.0.8.5
libopenh264.so.2|libopenh264-7bd47c3a.so.2.6.0
libopus.so.0|libopus-a676965d.so.0.10.1
libsharpyuv.so.0|libsharpyuv-2777c64a.so.0.1.1
libspeex.so.1|libspeex-dd5a2d1c.so.1.5.2
libsrt.so.1|libsrt-ccd6ae88.so.1.5.4
libssl.so.1|libssl-60250281.so.1.1.1k
libswresample.so.5|libswresample-f1bdf0d4.so.5.3.100
libswscale.so.8|libswscale-5efb2ca5.so.8.3.100
libtwolame.so.0|libtwolame-dfe0c2c6.so.0.0.0
libunistring.so.5|libunistring-7eaffe9f.so.5.2.0
libvorbis.so.0|libvorbis-7463f6bd.so.0.4.9
libvorbisenc.so.2|libvorbisenc-131c2ed7.so.2.0.12
libvpx.so.11|libvpx-09740bc5.so.11.0.0
libwebp.so.7|libwebp-bc89f640.so.7.1.10
libwebpmux.so.3|libwebpmux-601b9199.so.3.1.1
libx264.so.165|libx264-b1bb65f5.so.165
libx265.so.215|libx265-169666e3.so.215
libxcb.so.1|libxcb-5ddf6756.so.1.1.0
MAP
}
make_fflib "$SP/av.libs" "$ENV_DIR/fflib"
echo "-- fflib: $(ls "$ENV_DIR/fflib" | wc -l) symlinks -> $SP/av.libs"

# ---------- 5) 冒烟：关键 import ----------
echo "-- 冒烟 import"
LD_LIBRARY_PATH="$ENV_DIR/fflib:$SP/av.libs" PYTHONPATH= "$PY" - <<PY 2>&1 | tail -8
import torchcodec  # noqa 必须先于 lerobot/PIL（libjpeg soname）
import torch, numpy, huggingface_hub as hub, transformers, lerobot
print(f"  torch {torch.__version__} | numpy {numpy.__version__} | hub {hub.__version__}")
print(f"  transformers {transformers.__version__} | lerobot {lerobot.__version__}")
import lerobot.datasets.lerobot_dataset  # 触发 transformers 检查（hub<1.0 需 0.35.3）
print("  lerobot.datasets.lerobot_dataset import OK")
PY
echo
echo "== 完成。验收请跑: bash $LOCK_DIR/verify_env.sh $MODE =="
