#!/usr/bin/env python3
"""视频 + 轨迹图联动的单页 HTML 生成工具。

对有本地视频的 episode:
1. 用 ffmpeg 把三路相机视频 (cam_high + 左腕 + 右腕) 合成为一个马赛克视频
   (640x720, H.264, -g 5 改善 seek 精度), 参考 utils/episode_analysis.compose_episode_video;
2. 生成单个自包含 HTML (内嵌 Plotly.js + 全部轨迹数据), 提供
   task -> episode 选择, 播放/暂停 + 帧滑块, 视频与轨迹图竖线实时同步;
   episode 切换时重载视频与轨迹, 视频播完自动暂停, 可拖动回看。

用法:
    python tools/episode_video_sync.py                      # 每任务随机选 3 个有视频的 episode
    python tools/episode_video_sync.py --episodes 85,83,86  # 只处理指定的 episode 白名单
    python tools/episode_video_sync.py --per-task 5 --seed 0
    python tools/episode_video_sync.py --regen-video        # 强制重合成视频
    python tools/episode_video_sync.py --out outputs/episode_insight/interactive
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.episode_state_insight import (  # noqa: E402
    DEFAULT_CSV,
    QUAT_L,
    QUAT_R,
    TASK_SLUGS,
    load_tasks,
    episode_state,
)
from utils.episode_analysis import compute_rotation_deg  # noqa: E402

VIDEO_COLS = {
    "high": "high_video",
    "left": "left_video",
    "right": "right_video",
}
CAMERA_SUBDIR = {
    "high": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}
FPS = 25.0

PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ---------------------------------------------------------------------------
# 视频合成
# ---------------------------------------------------------------------------

def compose_mosaic_video(
    data_root: Path, video_segs: dict, out_path: Path, fps: int = 25,
) -> None:
    """三路相机视频合成 640x720 马赛克 (H.264, -g 5)。

    video_segs: {key: (path, from_ts, dur)}，每路相机用各自的 from_ts 截取——
    同一 episode 在三路相机文件中的起始时间戳不同（见 CSV *_from_timestamp），
    统一用 high 的 from_ts 会导致三路画面不同步/不是同一 episode。
    """
    filter_complex = (
        "[0:v]scale=640:480,setsar=1[top];"
        "[1:v]scale=320:240,setsar=1[left];"
        "[2:v]scale=320:240,setsar=1[right];"
        "[left][right]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[out]"
    )
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for key in ("high", "left", "right"):
        path, from_ts, dur = video_segs[key]
        cmd += ["-ss", f"{from_ts:.4f}", "-t", f"{dur:.4f}", "-i",
                str(data_root / path)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-g", "5",                       # 每 5 帧一个关键帧, 改善拖动 seek 精度
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def load_episode_data(df: pd.DataFrame, episode_index: int, fps: float = FPS) -> dict:
    """提取单个 episode 的轨迹数据 (xyz / 旋转角 / gripper / 帧时刻)。"""
    state, _ = episode_state(df, episode_index)
    n = state.shape[0]
    from_ts = float(
        df[df["episode_index"] == episode_index]["high_video_from_timestamp"].iloc[0])
    t = from_ts + np.arange(n) / fps
    rot_l = compute_rotation_deg(
        np.tile(state[0, QUAT_L].astype(float), (n, 1)),
        state[:, QUAT_L].astype(float))
    rot_r = compute_rotation_deg(
        np.tile(state[0, QUAT_R].astype(float), (n, 1)),
        state[:, QUAT_R].astype(float))
    return {
        "frames": int(n),
        "fps": fps,
        "from": float(from_ts),
        "t": t.tolist(),
        "xyz_l": np.asarray(state[:, 0:3]).tolist(),
        "xyz_r": np.asarray(state[:, 8:11]).tolist(),
        "rot_l": rot_l.tolist(),
        "rot_r": rot_r.tolist(),
        "grip_l": [float(v) for v in state[:, 7]],
        "grip_r": [float(v) for v in state[:, 15]],
    }


def get_plotly_js(cache_path: Path) -> str:
    """获取 Plotly.js 源码 (缓存或下载), 用于内嵌。"""
    if not cache_path.is_file():
        import urllib.request

        print(f"  downloading plotly.js -> {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PLOTLY_URL, cache_path)
    return cache_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML 模板
# ---------------------------------------------------------------------------

HEADER = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Episode 视频-轨迹联动分析</title>
<style>
  body { margin:0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
         background:#fcfcfb; color:#0b0b0b; }
  .wrap { max-width:1500px; margin:0 auto; padding:14px 18px; }
  h1 { font-size:17px; margin:0 0 10px; font-weight:600; }
  .select-bar { display:flex; align-items:center; gap:10px; margin-bottom:10px;
         font-size:14px; color:#52514e; }
  .select-bar select { font-size:14px; padding:3px 8px; border:1px solid #c3c2b7;
         border-radius:4px; background:#fff; }
  #info { font-size:13px; color:#898781; }
  .layout { display:flex; gap:16px; flex-wrap:wrap; }
  .video-col { flex:1 1 480px; min-width:480px; }
  .plot-col { flex:1 1 600px; min-width:600px; }
  video { width:100%; display:block; background:#000; }
  .controls { margin-top:10px; display:flex; align-items:center; gap:12px; }
  .controls button { font-size:14px; padding:5px 18px; cursor:pointer;
         border:1px solid #c3c2b7; background:#fff; border-radius:4px; }
  .controls input[type=range] { flex:1; }
  .frame-label { font-variant-numeric:tabular-nums; font-size:13px; color:#52514e; }
  .plot { height:215px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Episode 视频-轨迹联动分析</h1>
  <div class="select-bar">
    <label>任务 <select id="sel-task"></select></label>
    <label>Episode <select id="sel-ep"></select></label>
    <span id="info"></span>
  </div>
  <div class="layout">
    <div class="video-col">
      <video id="vid" muted controls playsinline preload="auto"></video>
      <div class="controls">
        <button id="btn">▶ 播放</button>
        <span style="font-size:13px;color:#52514e">帧</span>
        <input type="range" id="slider" min="0" step="1" value="0">
        <span class="frame-label" id="frame-label">0</span>
      </div>
    </div>
    <div class="plot-col">
      <div id="p-l" class="plot"></div>
      <div id="p-r" class="plot"></div>
      <div class="plot" style="height:185px" id="p-rot"></div>
      <div class="plot" style="height:185px" id="p-grip"></div>
    </div>
  </div>
</div>
"""

# __PLOTLY__ 占位 -> 内嵌 plotly.min.js;  __DATA__ 占位 -> 数据 JSON
SCRIPT = """
<script>
__PLOTLY__
</script>
<script>
const DATA = __DATA__;
const TASKS = DATA.tasks, EPS = DATA.eps;

const video = document.getElementById("vid");
const btn = document.getElementById("btn");
const slider = document.getElementById("slider");
const frameLabel = document.getElementById("frame-label");
const taskSel = document.getElementById("sel-task");
const epSel = document.getElementById("sel-ep");
const info = document.getElementById("info");

let dragging = false;
let cur = null;   // 当前 episode id

// ---------- 绘图 ----------
const C = { x:"#2a78d6", y:"#eb6834", z:"#1baf7a", l:"#2a78d6", r:"#eb6834" };
const layoutBase = { margin:{l:44,r:10,t:30,b:34}, paper_bgcolor:"#fcfcfb",
  plot_bgcolor:"#fcfcfb", font:{family:"system-ui", size:11, color:"#0b0b0b"},
  xaxis:{title:"frame index", gridcolor:"#e1e0d9"},
  yaxis:{gridcolor:"#e1e0d9"}, showlegend:true,
  legend:{font:{size:10}, orientation:"h", x:0, y:1.08} };
const mk = c => ({ color:c, size:3.5, opacity:0.6 });

function renderPlots(d){
  const fr = d.t.map((_,i)=>i);
  const xyzTr = (prefix, arr) => ["x","y","z"].map((k,i) => ({
    x:fr, y:arr.map(r=>r[i]), type:"scatter", mode:"markers",
    marker:mk(C[k]), name:prefix+" "+k }));
  const lineTr = (y, c, name) => ({ x:fr, y, type:"scatter", mode:"lines",
    line:{color:c, width:1.4}, name });
  const lay = (title, yt, extra={}) => Object.assign({}, layoutBase,
    { title, yaxis:Object.assign({}, layoutBase.yaxis, {title:yt}) }, extra);
  Plotly.react("p-l",   xyzTr("L", d.xyz_l), lay("Left arm: position xyz", "position (m)"));
  Plotly.react("p-r",   xyzTr("R", d.xyz_r), lay("Right arm: position xyz", "position (m)"));
  Plotly.react("p-rot", [lineTr(d.rot_l, C.l, "Left arm"), lineTr(d.rot_r, C.r, "Right arm")],
               lay("rotation from initial pose", "angle (deg)"));
  Plotly.react("p-grip",[lineTr(d.grip_l, C.l, "Left arm"), lineTr(d.grip_r, C.r, "Right arm")],
               lay("gripper opening", "gripper (0~1)", {yaxis:Object.assign({}, layoutBase.yaxis,{range:[-0.05,1.05]})}));
}

// ---------- 同步 ----------
const PLOT_IDS = ["p-l","p-r","p-rot","p-grip"];
function vlineShape(k){ return { type:"line", x0:k, x1:k, yref:"paper", y0:0, y1:1,
  line:{color:"#0b0b0b", width:1, dash:"dot"} }; }
function setFrame(k){
  const d = EPS[cur];
  k = Math.max(0, Math.min(d.frames-1, k|0));
  const t = k/d.fps;   // 马赛克视频时间轴从 0 开始 (ffmpeg -ss 前置截取重置了 PTS)
  if (Math.abs(video.currentTime - t) > 0.1) video.currentTime = t;
  slider.value = k;
  frameLabel.textContent = k;
  const shape = vlineShape(k);
  PLOT_IDS.forEach(id => Plotly.relayout(id, { shapes:[shape] }));
}

function loadEpisode(ep){
  cur = String(ep);
  const d = EPS[cur];
  video.src = d.video;
  slider.max = d.frames - 1; slider.value = 0;
  info.textContent = d.frames + " frames · " + d.fps + " fps · from " + d.from.toFixed(2) + "s";
  renderPlots(d);
  video.pause(); btn.textContent = "▶ 播放";
  // 等新源 metadata 加载后再定位 (切换 src 后立即 seek 会因未加载而失败)
  const seek = () => setFrame(0);
  if (video.readyState >= 1) seek();
  else video.addEventListener("loadedmetadata", seek, { once: true });
}

// ---------- 选择器 ----------
Object.entries(TASKS).forEach(([tid, t]) => {
  const o = document.createElement("option");
  o.value = tid; o.textContent = tid + " · " + t.name;
  taskSel.appendChild(o);
});
function fillEpisodes(){
  epSel.innerHTML = "";
  const t = TASKS[taskSel.value];
  t.episodes.forEach(ep => {
    const o = document.createElement("option");
    o.value = ep; o.textContent = "episode " + ep;
    epSel.appendChild(o);
  });
  loadEpisode(epSel.value);
}
taskSel.addEventListener("change", fillEpisodes);
epSel.addEventListener("change", () => loadEpisode(epSel.value));

// ---------- 播放 / 滑块 ----------
btn.addEventListener("click", () => {
  if (video.paused) { video.play(); btn.textContent = "⏸ 暂停"; }
  else { video.pause(); btn.textContent = "▶ 播放"; }
});
video.addEventListener("play",  () => { btn.textContent = "⏸ 暂停"; });
video.addEventListener("pause", () => { btn.textContent = "▶ 播放"; });
video.addEventListener("ended", () => { btn.textContent = "▶ 播放"; });
video.addEventListener("timeupdate", () => {
  if (!dragging && cur) {
    const d = EPS[cur];
    const k = Math.round(video.currentTime * d.fps);
    if (k >= 0 && k < d.frames) {
      slider.value = k; frameLabel.textContent = k;
      PLOT_IDS.forEach(id => Plotly.relayout(id, { shapes:[vlineShape(k)] }));
    }
  }
});
slider.addEventListener("input", () => { dragging = true;  if(cur) setFrame(+slider.value); });
slider.addEventListener("change", () => { dragging = false; });

window.addEventListener("load", fillEpisodes);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "episode_insight" / "interactive"))
    parser.add_argument("--regen-video", action="store_true", help="强制重合成马赛克视频")
    parser.add_argument("--episodes", default=None,
                        help="只处理这些 episode, 逗号分隔 (如 85,83,86)")
    parser.add_argument("--per-task", type=int, default=3,
                        help="每任务随机挑选的 episode 数 (无 --episodes 时, 默认 3)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子(可复现)")
    args = parser.parse_args()

    data_root = Path(args.csv).parent / Path(args.csv).stem
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"

    task_names = load_tasks(ROOT / "data" / "lerobot_v30_ee" / "meta" / "tasks.parquet")
    df = pd.read_csv(
        args.csv,
        usecols=["episode_index", "task_index", "frame_index", "length",
                 "observation.state", "high_video_path", "left_video_path", "right_video_path",
                 "high_video_from_timestamp", "left_video_from_timestamp", "right_video_from_timestamp",
                 "high_video_to_timestamp", "left_video_to_timestamp", "right_video_to_timestamp"],
    )
    df["_has_video"] = df["high_video_path"].apply(lambda p: (data_root / p).is_file())
    all_ok = sorted(
        df.groupby("episode_index")["_has_video"].all().loc[lambda s: s].index.tolist()
    )
    # episode -> task 映射（用首帧）
    ep_task = {
        int(ep): int(ti)
        for ep, ti in df.groupby("episode_index")["task_index"].first().items()
    }

    if args.episodes is not None:
        # 白名单（保留有视频的）
        ok_episodes = sorted(int(e) for e in args.episodes.split(",")
                             if int(e) in all_ok)
    else:
        # 每任务随机挑 --per-task 个有视频的 episode
        rng = np.random.default_rng(args.seed)
        ok_episodes = []
        for ti in sorted(set(ep_task.values())):
            cand = [ep for ep in all_ok if ep_task[ep] == ti]
            if not cand:
                continue
            ok_episodes += sorted(int(e) for e in
                                  rng.choice(cand, size=min(args.per_task, len(cand)),
                                             replace=False))
        ok_episodes.sort()
    print(f"{len(all_ok)} episodes have full video; "
          f"processing {len(ok_episodes)} "
          f"({'from --episodes whitelist' if args.episodes else 'random sample'})")

    # 1) 合成马赛克视频 (幂等)
    videos = {}
    for ep in sorted(ok_episodes):
        task_idx = int(df[df["episode_index"] == ep]["task_index"].iloc[0])
        slug = TASK_SLUGS.get(task_idx, f"task_{task_idx:03d}")
        mp4 = video_dir / f"{slug}_{ep:03d}.mp4"
        # 相对 HTML (out_dir) 的路径, 便于 HTML 目录整体移动
        videos[str(ep)] = os.path.relpath(mp4, out_dir)
        if mp4.is_file() and not args.regen_video:
            continue
        row = df[df["episode_index"] == ep].iloc[0]
        # 每路相机用各自 from/to 截取 (三路时间戳不同, 统一用 high 会错位)
        segs = {}
        for key, prefix in VIDEO_COLS.items():
            segs[key] = (
                str(row[f"{prefix}_path"]),
                float(row[f"{prefix}_from_timestamp"]),
                float(row[f"{prefix}_to_timestamp"]) - float(row[f"{prefix}_from_timestamp"]),
            )
        print(f"  compose ep {ep:3d} ({slug}) "
              f"high={segs['high'][1]:.2f} left={segs['left'][1]:.2f} "
              f"right={segs['right'][1]:.2f}s ...")
        compose_mosaic_video(data_root, segs, mp4)

    # 2) 组装数据
    tasks = {}
    eps = {}
    for ep in sorted(ok_episodes):
        task_idx = int(df[df["episode_index"] == ep]["task_index"].iloc[0])
        slug = TASK_SLUGS.get(task_idx, f"task_{task_idx:03d}")
        key = str(task_idx)
        if key not in tasks:
            tasks[key] = {
                "name": task_names.get(task_idx, slug),
                "slug": slug,
                "episodes": [],
            }
        tasks[key]["episodes"].append(ep)
        d = load_episode_data(df, ep)
        d["task"] = key
        d["video"] = videos[str(ep)]
        eps[str(ep)] = d

    # 3) 生成单 HTML (内嵌 Plotly + 数据)
    payload = {"tasks": tasks, "eps": eps}
    plotly_js = get_plotly_js(out_dir / "plotly.min.js")
    html = (
        HEADER
        + SCRIPT
            .replace("__PLOTLY__", plotly_js)
            .replace("__DATA__", json.dumps(payload))
    )
    out_html = out_dir / "episode_sync.html"
    out_html.write_text(html, encoding="utf-8")

    print(f"\nHTML -> {out_html}  ({out_html.stat().st_size/1e6:.1f} MB, "
          f"{len(ok_episodes)} episodes)")


if __name__ == "__main__":
    main()
