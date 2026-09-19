#!/usr/bin/env python3
"""只读状态采集：X0 cosine 训练。输出 KEY=VALUE，供本地看门狗解析。

部署位置 /data/outputs/x0_cosdecay_status.py（train-4090）。只读日志与 nvidia-smi，
不写任何训练产出。
"""

import glob
import re
import subprocess

LOG = "/data/outputs/x0_ee6d_sim_cosdecay_20260913_155416.log"
OUT = "/cloud/cloud-ssd1/x0_ee6d_sim_cosdecay"
TOTAL = 24000


def sh(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception:
        return ""


def kv(key: str, value) -> None:
    print(f"{key}={value}")


kv("ALIVE", 1 if sh("pgrep -f '/data/X-VLA/train.py'") else 0)
kv("SERVER_TIME", sh("date '+%m-%d %H:%M'"))
kv("GPU", sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
             "--format=csv,noheader | head -1") or "NA")
kv("DISK_SSD1_KB", sh("df -P /cloud/cloud-ssd1 | awk 'NR==2{print $4}'"))
kv("DISK_ROOT_KB", sh("df -P / | awk 'NR==2{print $4}'"))
kv("N_CKPT", len(glob.glob(f"{OUT}/pretrained/ckpt-*")))

try:
    with open(LOG, errors="replace") as f:
        lines = f.readlines()
except OSError:
    lines = []

# 全部字段取自最后一条 step 行，保证互相自洽
step_re = re.compile(r"\[(\d+)/%d\]" % TOTAL)
step_lines = [ln for ln in lines if step_re.search(ln)]
cur = step_lines[-1] if step_lines else ""


def g(pat: str) -> str:
    m = re.search(pat, cur)
    return m.group(1) if m else ""


kv("STEP", step_re.search(cur).group(1) if cur else "")
kv("LOSS", g(r"loss=([0-9.]+)"))
kv("POS", g(r"position=([0-9.]+)"))
kv("ROT", g(r"rotate6D=([0-9.]+)"))
kv("GRIP", g(r"gripper=([0-9.]+)"))
kv("SPEED", g(r"\(([0-9.]+)s/it\)"))
kv("LRCORE", g(r"lr_core=([0-9.eE+-]+)"))
kv("LRVLM", g(r"lr_vlm=([0-9.eE+-]+)"))
kv("GRADNORM", g(r"grad_norm=([0-9.]+)"))
kv("KEYRATIO", g(r"key_ratio=([0-9.]+%)"))

nan = err = 0
for ln in lines:
    if re.search(r"loss=(nan|inf|NaN|Inf)", ln):
        nan += 1
    if re.search(r"Traceback|CUDA out of memory|RuntimeError|Killed", ln):
        err += 1
kv("NAN", nan)
kv("ERR", err)
kv("LASTLINE", (lines[-1].strip()[:200] if lines else ""))
