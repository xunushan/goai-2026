#!/usr/bin/env python3
"""把某个 task 的 keyframe 图拼成联系表, 便于逐帧核对后写 observation。"""
import sys
import glob
from pathlib import Path
from PIL import Image, ImageDraw

task = sys.argv[1]
cam = sys.argv[2] if len(sys.argv) > 2 else "cam_high"
per_sheet = int(sys.argv[3]) if len(sys.argv) > 3 else 11
base = sys.argv[4] if len(sys.argv) > 4 else "outputs/in_context_learning_sim"
root = Path(base) / task / "images"
tag = Path(base).name.replace("in_context_learning_", "")
fs = sorted(glob.glob(str(root / f"*-{cam}.jpg")))
cols = int(sys.argv[5]) if len(sys.argv) > 5 else 4
scale = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5625   # 640x480 的缩放
w, h = int(640 * scale), int(480 * scale)
for p in range(0, len(fs), per_sheet):
    sub = fs[p:p + per_sheet]
    rows = (len(sub) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(sub):
        im = Image.open(f).resize((w, h))
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        sheet.paste(im, (x, y))
        d.text((x + 4, y + h + 4), Path(f).name, fill="black")
    out = f"/tmp/probe/sheet_{tag}_{task}_{cam}_{p//per_sheet}.png"
    sheet.save(out)
    print(out, len(sub))
