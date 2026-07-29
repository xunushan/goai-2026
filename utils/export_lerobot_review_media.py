#!/usr/bin/env python3
"""Export three-camera review media from a numeric review queue.

Run this script on the server that contains the LeRobot videos. The generated
directory is self-contained: copy it to a local machine and open index.html to
label findings without running a web server.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

LABELS = {
    "data_label": [
        {"value": "", "zh": "未选择", "description": "尚未判断数据是否存在问题"},
        {"value": "valid", "zh": "数据正常", "description": "画面、状态、动作和时间关系均合理"},
        {"value": "frame_misalignment", "zh": "帧时序错位", "description": "前后帧顺序、帧号或时间位置不一致"},
        {"value": "camera_misalignment", "zh": "相机不同步", "description": "三路相机显示的动作时刻不一致"},
        {"value": "action_state_misalignment", "zh": "动作状态错位", "description": "action 与对应的机器人 state 在时间上不匹配"},
        {"value": "timestamp_problem", "zh": "时间戳异常", "description": "视频或数值数据的时间戳跳变、重复或不连续"},
        {"value": "quaternion_sign_only", "zh": "仅四元数符号翻转", "description": "q 与 -q 变化但真实姿态连续"},
        {"value": "action_discontinuity", "zh": "动作不连续", "description": "相邻动作出现不符合正常示范的突变"},
        {"value": "corrupted_video", "zh": "视频损坏", "description": "视频缺帧、黑屏、花屏或无法解码"},
        {"value": "uncertain", "zh": "无法判断", "description": "现有画面或指标不足以确认数据质量"},
    ],
    "behavior_label": [
        {"value": "", "zh": "未选择", "description": "尚未判断行为语义"},
        {"value": "normal_task_progress", "zh": "正常任务过程", "description": "行为属于当前堆叠任务的正常步骤"},
        {"value": "normal_next_block_attempt", "zh": "正常抓取下一个方块", "description": "返回或再次闭合是任务要求的下一次抓取"},
        {"value": "failed_grasp", "zh": "抓取失败", "description": "夹爪尝试抓取但未稳定抓住目标"},
        {"value": "failed_grasp_then_recovered", "zh": "抓取失败后恢复成功", "description": "首次抓取失败，随后重新对准并成功抓取"},
        {"value": "collision", "zh": "发生碰撞", "description": "机械臂、夹爪或所持物体碰撞了环境或其他物体"},
        {"value": "collision_then_recovered", "zh": "碰撞后恢复成功", "description": "碰撞后策略进行了有效纠正并继续任务"},
        {"value": "object_dropped", "zh": "物体掉落", "description": "已抓取物体脱离夹爪或掉到非目标位置"},
        {"value": "drop_then_recovered", "zh": "掉落后恢复成功", "description": "物体掉落后被重新定位、抓取并继续任务"},
        {"value": "retry_without_success", "zh": "尝试恢复但未成功", "description": "出现重新对准或重新抓取，但最终没有恢复"},
        {"value": "oscillation", "zh": "无效振荡", "description": "机械臂在局部区域反复运动且没有产生有效进展"},
        {"value": "stagnation", "zh": "停滞", "description": "机械臂或任务长时间没有有效进展"},
        {"value": "false_positive", "zh": "规则误报", "description": "数值规则命中，但视频中没有对应异常"},
        {"value": "uncertain", "zh": "无法判断", "description": "画面不足或行为含义不明确"},
    ],
    "recovery_result": [
        {"value": "", "zh": "未选择", "description": "尚未判断恢复结果"},
        {"value": "not_applicable", "zh": "不适用", "description": "片段中没有失败或不需要恢复"},
        {"value": "recovery_success", "zh": "恢复成功", "description": "异常后重新回到有效任务流程"},
        {"value": "recovery_failed", "zh": "恢复失败", "description": "尝试恢复但未能继续完成有效步骤"},
        {"value": "recovery_not_attempted", "zh": "未尝试恢复", "description": "异常发生后没有可见的恢复行为"},
        {"value": "uncertain", "zh": "无法判断", "description": "无法从当前片段判断恢复结果"},
    ],
}


@dataclass(frozen=True)
class Segment:
    path: Path
    start: float
    end: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-keyframes", action="store_true")
    parser.add_argument("--export-clips", action="store_true")
    parser.add_argument("--export-full-task-episodes", action="store_true")
    parser.add_argument("--task", help="Required for --export-full-task-episodes")
    parser.add_argument("--severity", action="append", choices=["critical", "high", "medium", "low", "control"])
    parser.add_argument("--max-items", type=int, default=0, help="0 means no limit")
    parser.add_argument("--crf", type=int, default=26)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_metadata(dataset: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    info = json.loads((dataset / "meta" / "info.json").read_text())
    episodes: dict[int, dict[str, Any]] = {}
    for path in sorted((dataset / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episodes[int(row["episode_index"])] = row
    return info, episodes


def episode_task(row: dict[str, Any]) -> str:
    tasks = row.get("tasks") or []
    return str(tasks[0]) if tasks else ""


def segments_for_episode(
    dataset: Path, info: dict[str, Any], row: dict[str, Any]
) -> list[Segment]:
    result = []
    for camera in CAMERAS:
        prefix = f"videos/{camera}"
        relative = info["video_path"].format(
            video_key=camera,
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
        )
        result.append(
            Segment(
                dataset / relative,
                float(row[f"{prefix}/from_timestamp"]),
                float(row[f"{prefix}/to_timestamp"]),
            )
        )
    return result


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def input_args(segments: list[Segment], local_start: float, duration: float | None) -> list[str]:
    result: list[str] = []
    for segment in segments:
        result += ["-ss", f"{segment.start + local_start:.6f}"]
        if duration is not None:
            result += ["-t", f"{duration:.6f}"]
        result += ["-i", str(segment.path)]
    return result


def mosaic_filter() -> str:
    return (
        "[0:v]scale=640:480,setsar=1[top];"
        "[1:v]scale=320:240,setsar=1[left];"
        "[2:v]scale=320:240,setsar=1[right];"
        "[left][right]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[out]"
    )


def export_frame(
    segments: list[Segment], frame: int, fps: float, output: Path,
    overwrite: bool, dry_run: bool,
) -> None:
    if output.is_file() and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    command += input_args(segments, frame / fps, None)
    command += [
        "-filter_complex", mosaic_filter(), "-map", "[out]",
        "-frames:v", "1", str(output),
    ]
    run(command, dry_run)


def export_clip(
    segments: list[Segment], start_frame: int, end_frame: int, fps: float,
    output: Path, crf: int, preset: str, overwrite: bool, dry_run: bool,
) -> None:
    if output.is_file() and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = (end_frame - start_frame + 1) / fps
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    command += input_args(segments, start_frame / fps, duration)
    command += [
        "-filter_complex", mosaic_filter(), "-map", "[out]",
        "-frames:v", str(end_frame - start_frame + 1),
        "-r", f"{fps:g}", "-c:v", "libx264", "-preset", preset,
        "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    run(command, dry_run)


def missing_sources(segments: list[Segment]) -> list[str]:
    return [str(segment.path) for segment in segments if not segment.path.is_file()]


def option_html(group: str) -> str:
    return "".join(
        f'<option value="{html.escape(item["value"])}">'
        f'{html.escape(item["zh"])} — {html.escape(item["description"])}</option>'
        for item in LABELS[group]
    )


def build_html(output: Path, items: list[dict[str, Any]]) -> None:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    labels = json.dumps(LABELS, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>LeRobot 训练数据人工审查</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:20px;background:#f5f6f8;color:#222}}
.toolbar{{position:sticky;top:0;background:#fff;padding:12px;z-index:3;border:1px solid #ddd}}
.item{{background:white;margin:18px 0;padding:16px;border:1px solid #ddd;border-radius:8px}}
.meta{{white-space:pre-wrap;background:#f4f4f4;padding:8px;font-size:13px}}
.frames{{display:flex;gap:8px;overflow-x:auto}} .frames img{{height:260px}}
video{{max-width:760px;width:100%}} label{{display:block;margin-top:8px;font-weight:600}}
select,textarea{{width:100%;padding:7px}} textarea{{min-height:70px}}
.saved{{color:#087f23}} .help{{font-size:12px;color:#555}}
</style></head><body>
<div class="toolbar"><b>LeRobot 训练数据人工审查</b>
 <span id="progress"></span>
 <button onclick="exportReview()">导出标注 JSON</button>
 <button onclick="clearReview()">清空本地标注</button>
 <div class="help">标注保存在当前浏览器 localStorage；完成后务必点击“导出标注 JSON”。</div>
</div><div id="items"></div>
<script>
const ITEMS={payload}; const LABELS={labels}; const KEY="lerobot-review-v1";
let reviews=JSON.parse(localStorage.getItem(KEY)||"{{}}");
function opts(group,value){{return LABELS[group].map(x=>`<option value="${{x.value}}" ${{x.value===value?"selected":""}}>${{x.zh}} — ${{x.description}}</option>`).join("")}}
function save(id,field,value){{reviews[id]=reviews[id]||{{issue_id:id}};reviews[id][field]=value;reviews[id].reviewed_at=new Date().toISOString();localStorage.setItem(KEY,JSON.stringify(reviews));renderProgress()}}
function renderProgress(){{let n=Object.values(reviews).filter(x=>x.behavior_label||x.data_label).length;document.getElementById("progress").textContent=` 已标注 ${{n}} / ${{ITEMS.length}}`}}
function render(){{
 let root=document.getElementById("items");
 root.innerHTML=ITEMS.map((x,i)=>{{let r=reviews[x.issue_id]||{{}};
 let frames=(x.media&&x.media.keyframes||[]).map(p=>`<img loading="lazy" src="${{p}}">`).join("");
 let clip=x.media&&x.media.clip?`<video controls preload="metadata" src="${{x.media.clip}}"></video>`:"";
 let questions=(x.questions||[]).map(q=>"• "+q).join("\\n");
 return `<section class="item"><h3>${{i+1}}. ${{x.issue_id}}</h3>
 <div>${{x.category_cn||x.category}}｜episode ${{x.episode_index}}｜frames ${{x.start_frame}}–${{x.end_frame}}｜${{x.severity}}</div>
 <div class="meta">原因：${{(x.reasons||[]).join(", ")}}\\n待确认：\\n${{questions}}\\n指标：${{JSON.stringify(x.metrics||{{}},null,2)}}</div>
 <div class="frames">${{frames}}</div>${{clip}}
 <label>数据质量标签</label><select onchange="save('${{x.issue_id}}','data_label',this.value)">${{opts("data_label",r.data_label||"")}}</select>
 <label>行为标签</label><select onchange="save('${{x.issue_id}}','behavior_label',this.value)">${{opts("behavior_label",r.behavior_label||"")}}</select>
 <label>恢复结果</label><select onchange="save('${{x.issue_id}}','recovery_result',this.value)">${{opts("recovery_result",r.recovery_result||"")}}</select>
 <label>备注</label><textarea onchange="save('${{x.issue_id}}','notes',this.value)">${{r.notes||""}}</textarea></section>`}}).join("");
 renderProgress();
}}
function exportReview(){{let blob=new Blob([JSON.stringify({{schema_version:1,labels:LABELS,reviews:Object.values(reviews)}},null,2)],{{type:"application/json"}});let a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="confirmed_review.json";a.click();URL.revokeObjectURL(a.href)}}
function clearReview(){{if(confirm("确认清空当前浏览器中的全部标注？")){{localStorage.removeItem(KEY);reviews={{}};render()}}}}
render();
</script></body></html>"""
    output.write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found in PATH")
    if args.export_full_task_episodes and not args.task:
        raise ValueError("--task is required with --export-full-task-episodes")
    args.dataset = args.dataset.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info, episodes = load_metadata(args.dataset)
    fps = float(info["fps"])
    items = load_rows(args.review_queue)
    if args.severity:
        items = [item for item in items if item.get("severity") in set(args.severity)]
    if args.max_items:
        items = items[: args.max_items]
    manifest: list[dict[str, Any]] = []
    exported_items: list[dict[str, Any]] = []

    for item in items:
        episode = int(item["episode_index"])
        row = episodes.get(episode)
        if row is None:
            manifest.append({"issue_id": item["issue_id"], "status": "missing_episode_metadata"})
            continue
        segments = segments_for_episode(args.dataset, info, row)
        missing = missing_sources(segments)
        if missing:
            manifest.append({
                "issue_id": item["issue_id"], "episode_index": episode,
                "status": "missing_video", "details": ";".join(missing),
            })
            continue
        length = int(row["length"])
        issue_dir = args.output_dir / "candidates" / item["issue_id"]
        media = {"keyframes": [], "clip": ""}
        if args.export_keyframes:
            for frame in sorted(set(int(x) for x in item.get("key_frames", []))):
                frame = max(0, min(length - 1, frame))
                output = issue_dir / f"frame_{frame:06d}_mosaic.jpg"
                export_frame(segments, frame, fps, output, args.overwrite, args.dry_run)
                media["keyframes"].append(str(output.relative_to(args.output_dir)))
        if args.export_clips:
            padding = int(item.get("clip_padding_frames", 50))
            start = max(0, int(item["start_frame"]) - padding)
            end = min(length - 1, int(item["end_frame"]) + padding)
            output = issue_dir / "mosaic_clip.mp4"
            export_clip(
                segments, start, end, fps, output, args.crf, args.preset,
                args.overwrite, args.dry_run,
            )
            media["clip"] = str(output.relative_to(args.output_dir))
        exported = dict(item)
        exported["media"] = media
        exported_items.append(exported)
        manifest.append({
            "issue_id": item["issue_id"], "episode_index": episode,
            "status": "dry_run" if args.dry_run else "ok",
            "keyframes": len(media["keyframes"]), "clip": media["clip"],
        })

    if args.export_full_task_episodes:
        for episode, row in sorted(episodes.items()):
            if episode_task(row) != args.task:
                continue
            segments = segments_for_episode(args.dataset, info, row)
            if missing_sources(segments):
                manifest.append({
                    "issue_id": "", "episode_index": episode,
                    "status": "full_episode_missing_video",
                })
                continue
            length = int(row["length"])
            output = args.output_dir / "full_episodes" / f"episode_{episode:04d}_mosaic.mp4"
            export_clip(
                segments, 0, length - 1, fps, output, args.crf, args.preset,
                args.overwrite, args.dry_run,
            )
            manifest.append({
                "issue_id": "", "episode_index": episode,
                "status": "full_episode_dry_run" if args.dry_run else "full_episode_ok",
                "clip": str(output.relative_to(args.output_dir)),
            })

    with (args.output_dir / "media_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["issue_id", "episode_index", "status", "keyframes", "clip", "details"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    (args.output_dir / "review_items.json").write_text(
        json.dumps(exported_items, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "label_definitions.json").write_text(
        json.dumps(LABELS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    build_html(args.output_dir / "index.html", exported_items)
    print(json.dumps({
        "queue_items_selected": len(items),
        "items_exported": len(exported_items),
        "manifest_rows": len(manifest),
        "output_dir": str(args.output_dir),
        "open_locally": str(args.output_dir / "index.html"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
