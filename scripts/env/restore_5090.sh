#!/usr/bin/env bash
# restore_5090.sh — train-5090 关机/重启后的一键恢复（容器被重建时 /data 与 /root 会丢）
#
# 恢复内容：
#   1) /data  <- /cloud/cloud-ssd1/pi05_portable/data/  （24G：venv + base 权重 + 数据 + openpi 代码）
#   2) ~/.cache/huggingface/lerobot/RoboDojo_sim_arx-x5_v30 软链
#   3) 私有仓库 git 凭据（/root/.openpi-gh-cred，需从本机重新灌 token）
#   4) 校验 venv 指针 + 打印下一步
#
# 用法（root）：bash /cloud/cloud-ssd1/restore_5090.sh
# 幂等：/data 已存在且非空时默认跳过复制，用 FORCE=1 强制重灌。
set -uo pipefail

SRC=/cloud/cloud-ssd1/pi05_portable/data
ENVDIR=/data/goai/envs/pi05_l060
PY="$ENVDIR/bin/python"
HFHOME="${HOME:-/root}/.cache/huggingface/lerobot"

echo "== [0/4] 前置检查 =="
[ -d "$SRC" ] || { echo "FATAL: 迁移包不存在：$SRC（持久盘是否掉挂？）"; exit 1; }
echo "  源包: $SRC ($(du -sh "$SRC" 2>/dev/null | cut -f1))"
echo "  根盘可用: $(df -h / | awk 'NR==2{print $4}')"

echo; echo "== [1/4] 还原 /data =="
if [ -n "$(ls -A /data 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "  /data 已非空，跳过复制（要重灌请 FORCE=1）"
else
  mkdir -p /data
  cp -a "$SRC/." /data/ && sync
  echo "  复制完成: $(du -sh /data 2>/dev/null | cut -f1)"
fi

echo; echo "== [2/4] 数据集软链 =="
mkdir -p "$HFHOME"
ln -sfn /data/data/sim_lerobot_v30_joint-224x224 "$HFHOME/RoboDojo_sim_arx-x5_v30"
ls -la "$HFHOME/RoboDojo_sim_arx-x5_v30"

echo; echo "== [3/4] git 凭据 =="
CRED=/root/.openpi-gh-cred
if [ -s "$CRED" ]; then
  echo "  $CRED 已存在（$(wc -c < "$CRED") 字节），沿用"
else
  echo "  缺失！从本机执行下面这条（token 不回显）："
  echo "    { printf 'https://xunushan:'; tr -d '\\n\\r' < ~/Documents/token/github; printf '@github.com\\n'; } \\"
  echo "      | ssh train-5090 'umask 077; cat > /root/.openpi-gh-cred; chmod 600 /root/.openpi-gh-cred'"
fi

echo; echo "== [4/4] venv 指针校验 =="
if [ -x "$PY" ]; then
  echo "  解释器: $($PY -V 2>&1)"
  echo "  pyvenv home: $(grep '^home' "$ENVDIR/pyvenv.cfg" 2>/dev/null)"
  [ -e "$(readlink -f "$ENVDIR/bin/python")" ] && echo "  bin/python 目标存在 OK" || echo "  ⚠ bin/python 目标缺失"
  [ -e "$ENVDIR/fflib/libavcodec.so.61" ] && echo "  fflib 软链 OK" || echo "  ⚠ fflib 软链断裂（需跑 setup_pi05_l060_env.sh gpu 重建）"
else
  echo "  ⚠ 解释器缺失: $PY"
fi

cat <<'EOF'

== 恢复完成。下一步 ==
  A) 冒烟环境验收:
     cd /data && GOAI_ROOT=/data/goai DATASET_ROOT=/data/data OPENPI_SRC=/data/openpi \
       bash /data/pi05_env_scripts/verify_env.sh
  B) Blackwell / jax sm_120 核验（需 GPU 已挂上）:
     export LD_LIBRARY_PATH=/data/goai/envs/pi05_l060/fflib:/data/goai/envs/pi05_l060/lib/python3.12/site-packages/av.libs
     /data/goai/envs/pi05_l060/bin/python -c "import jax; print(jax.devices())"
  C) 边界扫描（remat off + EMA on, eff32）:
     OUT=/cloud/cloud-ssd1/pi05_smoke_5090 bash /data/pi05_env_scripts/run_5090_sweep.sh scan
EOF
