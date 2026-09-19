#!/usr/bin/env python3
"""X0 cosine 训练看门狗：本地运行，ssh 轮询 train-4090，飞书心跳 + 异常告警。

飞书凭据只存在于本机，因此轮询在本地发起，服务器上只放只读采集脚本
/data/outputs/x0_cosdecay_status.py。

用法：python3 scripts/env/watch_x0_cosdecay.py [--interval 1800]
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

HOST = "train-4090"
COLLECTOR = "/data/outputs/x0_cosdecay_status.py"
SENDER = os.path.expanduser("~/.claude/skills/feishu-sender/scripts/send.py")
TOTAL = 24000
PEAK_CORE_LR = 1e-4  # learning_rate；vlm/soft_prompt 为其 0.1 倍

DISK_SSD1_MIN_KB = 15 * 1024 * 1024  # 15 GiB
DISK_ROOT_MIN_KB = 3 * 1024 * 1024  # 3 GiB
STALL_POLLS = 3  # 连续 3 次（90 min）step 不变视为卡住


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def feishu(text: str) -> None:
    try:
        r = subprocess.run([sys.executable, SENDER, "text", "-"], input=text,
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            log(f"飞书发送失败: {r.stderr.strip()[:200]}")
        else:
            log("飞书已发送")
    except Exception as exc:  # noqa: BLE001
        log(f"飞书异常: {exc}")


def collect() -> dict:
    """返回 KEY=VALUE 字典；ssh 失败返回空 dict。"""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", HOST,
             f"python3 {COLLECTOR}"],
            capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        log(f"ssh 异常: {exc}")
        return {}
    if r.returncode != 0:
        log(f"ssh rc={r.returncode}: {r.stderr.strip()[:200]}")
        return {}
    out = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def num(d: dict, key: str, default=None):
    try:
        return float(d[key])
    except (KeyError, ValueError):
        return default


def gib(kb) -> str:
    return "NA" if kb is None else f"{kb / 1024 / 1024:.1f}G"


def lr_str(core, step) -> str:
    if step < 1000:
        return "冻结中（freeze_steps=1000，之后解冻）"
    if core is None:
        return "NA"
    ratio = core / PEAK_CORE_LR
    return f"core={core:.2e}（峰值 {ratio * 100:.1f}%）"


def compose(d: dict, eta: str, note: str = "") -> str:
    step = num(d, "STEP", 0)
    total = TOTAL
    pct = 100 * step / total
    head = f"【X0 cosine 训练心跳 {datetime.now():%m-%d %H:%M}（服务器 {d.get('SERVER_TIME', '?')}）】"
    body = [
        head,
        f"step {int(step)}/{total} ({pct:.1f}%) · {d.get('SPEED', '?')}s/it · ETA {eta}",
        f"loss={d.get('LOSS', '?')} (pos {d.get('POS', '?')} / rot {d.get('ROT', '?')} / grip {d.get('GRIP', '?')})"
        f" grad_norm={d.get('GRADNORM', '?')} key={d.get('KEYRATIO', '?')}",
        f"LR {lr_str(num(d, 'LRCORE'), step)}",
        f"GPU {d.get('GPU', 'NA')}",
        f"磁盘 ssd1 {gib(num(d, 'DISK_SSD1_KB'))} 可用 · 根盘 {gib(num(d, 'DISK_ROOT_KB'))} 可用"
        f" · ckpt {d.get('N_CKPT', '?')} 个",
    ]
    if note:
        body.append(note)
    return "\n".join(body)


def estimate_eta(step: float, speed: float) -> str:
    if not speed or speed <= 0 or step <= 0:
        return "NA"
    secs = (TOTAL - step) * speed
    return f"{(datetime.now() + timedelta(seconds=secs)):%H:%M}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印消息不发送、不退出，用于验证格式")
    args = ap.parse_args()

    if args.dry_run:
        global feishu  # noqa: PLW0603
        feishu = lambda text: print("--- 飞书消息预览 ---\n" + text, flush=True)

    if not os.path.exists(SENDER):
        log(f"FATAL 找不到飞书脚本 {SENDER}")
        return 2

    log(f"看门狗启动，轮询间隔 {args.interval}s")
    last_step = None
    stall_count = 0
    ssh_fails = 0
    active_warn: set[str] = set()

    while True:
        d = collect()
        if not d:
            ssh_fails += 1
            if ssh_fails >= 3:
                feishu(f"【X0 cosine 看门狗告警】{datetime.now():%H:%M}\n"
                       f"连续 {ssh_fails} 次无法连接 {HOST}，无法确认训练状态。")
                return 1
            time.sleep(args.interval)
            continue
        ssh_fails = 0

        step = num(d, "STEP", 0) or 0
        speed = num(d, "SPEED")
        alive = d.get("ALIVE") == "1"
        eta = estimate_eta(step, speed)
        warn = []   # 非致命：随心跳附带，首次出现时额外单发一条
        fatal = []  # 致命：发送并退出

        # 正常完成
        if step >= TOTAL:
            feishu(compose(d, "已完成", note=f"✅ 训练到达 {TOTAL} 步，正常结束。"))
            log("训练完成，看门狗退出")
            return 0

        # 异常退出
        if not alive:
            tail = d.get("LASTLINE", "")
            feishu(compose(d, "N/A", note=
                           f"❌ 训练进程已不在（停在 step {int(step)}/{TOTAL}），非正常结束。\n"
                           f"日志末行：{tail}\n"
                           f"请检查 {HOST}:/data/outputs/ 下日志。"))
            log("训练进程消失，看门狗退出")
            return 1

        # 数值异常 / 报错（致命）
        if (num(d, "NAN", 0) or 0) > 0:
            fatal.append(f"❌ 日志出现 {int(num(d, 'NAN', 0))} 行 loss=nan/inf")
        if (num(d, "ERR", 0) or 0) > 0:
            fatal.append(f"❌ 日志出现 {int(num(d, 'ERR', 0))} 行报错关键字"
                         f"（Traceback/OOM/RuntimeError/Killed）")

        # 磁盘（非致命：本轮已遇到一次，删 model_state 后恢复，不该因此停掉监控）
        if (num(d, "DISK_SSD1_KB", 1e18) or 0) < DISK_SSD1_MIN_KB:
            warn.append(f"⚠️ 数据盘仅剩 {gib(num(d, 'DISK_SSD1_KB'))}")
        if (num(d, "DISK_ROOT_KB", 1e18) or 0) < DISK_ROOT_MIN_KB:
            warn.append(f"⚠️ 根盘仅剩 {gib(num(d, 'DISK_ROOT_KB'))}")

        # 卡住检测（致命）
        if last_step is not None and step == last_step:
            stall_count += 1
        else:
            stall_count = 0
        last_step = step
        if stall_count >= STALL_POLLS:
            fatal.append(f"❌ step 连续 {stall_count} 次轮询未推进（卡住）")

        # 新出现的磁盘告警单发一条，避免等到下一个心跳周期
        fresh_warn = [w for w in warn if w not in active_warn]
        if fresh_warn:
            feishu(compose(d, eta, note="新告警：\n" + "\n".join(fresh_warn)))
            active_warn = set(warn)
        elif not warn:
            active_warn = set()
        # 已发过的新告警不再重复单发，但仍会出现在每 30min 心跳里
        if not fresh_warn:
            feishu(compose(d, eta, note="\n".join(warn)))

        log(f"step={int(step)} loss={d.get('LOSS')} eta={eta} "
            f"warn={len(warn)} fatal={len(fatal)}")

        if fatal:
            log("检测到致命异常，告警已发出，看门狗退出")
            return 1
        if args.dry_run:
            return 0

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
