#!/usr/bin/env python3
"""把逐帧图像文件夹按顺序合成 mp4，并删除原图像。

典型用途：策略服务端 deploy.yml `save_images` 落盘的一帧一个 jpg 目录
（<root>/<时间戳>_<task>/env<idx>_<episode>/<相机名>/000000.jpg …）转成视频，
省磁盘又方便回看。

    # 预演：只打印计划，不写不删
    python tools/images_to_video.py /data/outputs/sim_images/<run> --dry-run
    # 转换 + 删帧
    python tools/images_to_video.py /data/outputs/sim_images/<run>

布局约定（一条规则，无例外）：
    视频写到「图像所在目录的**旁边**」，文件名取该目录名 ——
        <episode>/cam_head/000000.jpg …  ->  <episode>/cam_head.mp4
        <flat_dir>/000000.jpg …          ->  <flat_dir>.mp4
    写完并**校验通过**后删除该目录内的帧；目录空了就删掉目录本身，因此
    save_images 的 episode 目录最后只剩各路的 mp4。

发现方式：递归扫描输入目录，**任何直接含图像的目录**都视为一路视频；
一个 episode 下的多路相机各自成一个 mp4（不拼接、不重编码）。

顺序与帧率：
    帧按文件名里的**数字**排序（000123.jpg / frame_123.png 都认），无数字的按字面序。
    默认 --fps 25（与 RoboDojo 仿真视频同惯例），用 --hold N 让每帧重复 N 遍。
    注意 save_images 存的是「每次重规划一帧」（eval_batch 下约 actions_per_chunk 个
    仿真步一帧），所以真机时间轴要靠 --hold 还原：25Hz 仿真、h=30 时
    `--fps 25 --hold 30` 得到的时长与真机一致；只想快速回看用默认 --hold 1 即可。
    不推荐把 fps 直接压到 1 以下（<1 fps 的 mp4 多数播放器/工具不友好）。

安全：
    - 只有视频**写成且校验通过**（ffmpeg 退出码 0 + ffprobe 帧数吻合）才删帧；
    - 目标 mp4 已存在时默认跳过（连同其帧一起保留），--overwrite 可覆盖；
    - 帧数 < --min-frames、或同一路内帧尺寸不一致时跳过该路（不写不删）；
    - --keep 只转不删，--dry-run 只打印。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# 发现与排序
# ---------------------------------------------------------------------------

def find_streams(root: Path, exts: set[str]) -> list[Path]:
    """递归找出所有**直接含图像**的目录（每个目录 = 一路视频）。"""
    streams: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(Path(name).suffix.lower() in exts for name in filenames):
            streams.append(Path(dirpath))
    return sorted(streams)


def frame_paths(stream_dir: Path, exts: set[str]) -> list[Path]:
    """该目录下的帧文件，按文件名中的数字排序（无数字者按字面序）。"""
    files = [
        path
        for path in stream_dir.iterdir()
        if path.is_file() and path.suffix.lower() in exts
    ]
    return sorted(files, key=_frame_sort_key)


def _frame_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else -1, path.name)


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------

def read_frame(path: Path):
    """读成 HWC uint8 RGB。"""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def frame_size(path: Path) -> tuple[int, int]:
    """只读图像头拿 (width, height)，用于编码前的尺寸一致性校验（不解码像素）。"""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def encode_ffmpeg(
    out_path: Path,
    frames: Iterable[Path],
    size: tuple[int, int],
    *,
    fps: float,
    hold: int,
    crf: int,
) -> tuple[Path, str]:
    """ffmpeg 管道编码（与 RoboDojo utils/save_file.py 的 VideoStreamWriter 同参数）。"""
    width, height = size
    pad_w, pad_h = width + (width % 2), height + (height % 2)  # yuv420p 要求偶数
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{pad_w}x{pad_h}", "-framerate", str(fps),
        "-i", "-",
        "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", str(crf),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for path in frames:
            data = _pad_to_even(read_frame(path), pad_w, pad_h).tobytes()
            for _ in range(hold):
                proc.stdin.write(data)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg 编码失败")
    return out_path, "ffmpeg"


def encode_opencv(
    out_path: Path,
    frames: Iterable[Path],
    size: tuple[int, int],
    *,
    fps: float,
    hold: int,
    crf: int,
) -> tuple[Path, str]:
    """opencv mp4v 兜底（无 ffmpeg 时用；crf 不生效）。"""
    import cv2

    width, height = size
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter 打不开输出文件")
    try:
        for path in frames:
            rgb = read_frame(path)
            for _ in range(hold):
                writer.write(rgb[:, :, ::-1].copy())  # RGB -> BGR
    finally:
        writer.release()
    return out_path, "opencv"


def _format_bytes(count: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.1f} {unit}" if unit != "B" else f"{count} B"
        count /= 1024.0
    return f"{count:.1f} GB"


def _pad_to_even(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.shape[0] == height and array.shape[1] == width:
        return array
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: array.shape[0], : array.shape[1]] = array[:, :, :3]
    return canvas


# ---------------------------------------------------------------------------
# 校验与清理
# ---------------------------------------------------------------------------

def verify_video(out_path: Path, expected_frames: int) -> str | None:
    """返回 None 表示校验通过，否则返回失败原因。"""
    if not out_path.exists():
        return "输出文件不存在"
    if out_path.stat().st_size == 0:
        return "输出文件为空"
    if shutil.which("ffprobe") is None:
        return None  # 只能用大小兜底
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        got = int(probe.stdout.strip().splitlines()[0])
    except Exception as exc:  # ffprobe 存在但解析失败
        return f"ffprobe 校验异常: {exc!r}"
    if got != expected_frames:
        return f"帧数不符: 期望 {expected_frames}, 实际 {got}"
    return None


def delete_frames(frames: Sequence[Path], stream_dir: Path) -> int:
    """删帧并返回释放字节数；目录空了则删掉目录本身。"""
    freed = 0
    for path in frames:
        try:
            freed += path.stat().st_size
            path.unlink()
        except OSError as exc:
            print(f"  !! 删除失败 {path}: {exc!r}", flush=True)
    try:
        if stream_dir.exists() and not any(stream_dir.iterdir()):
            stream_dir.rmdir()
    except OSError:
        pass
    return freed


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def convert_stream(
    stream_dir: Path,
    args: argparse.Namespace,
    exts: set[str],
) -> dict:
    """处理一路（一个目录）：校验 → 编码 → 校验 → 删帧。返回结果字典。"""
    frames = frame_paths(stream_dir, exts)
    result = {
        "dir": str(stream_dir),
        "frames": len(frames),
        "video": "",
        "status": "",
        "freed": 0,
    }
    if len(frames) < args.min_frames:
        result["status"] = f"跳过：帧数 {len(frames)} < min-frames {args.min_frames}"
        return result

    out_path = stream_dir.parent / f"{stream_dir.name}.mp4"
    if out_path.exists() and not args.overwrite:
        result["status"] = f"跳过：{out_path.name} 已存在（--overwrite 可覆盖）"
        return result

    size = frame_size(frames[0])  # (width, height)，只读头
    for path in frames[1:]:
        if frame_size(path) != size:
            result["status"] = (
                f"跳过：同路帧尺寸不一致（首帧 {size[0]}x{size[1]}，{path.name} 不符）"
            )
            return result

    if args.dry_run:
        result["video"] = str(out_path)
        result["status"] = f"预演：将写 {len(frames)} 帧（x{args.hold}）@{args.fps}fps 并删原帧"
        return result

    backend = args.codec
    if backend == "auto":
        backend = "ffmpeg" if shutil.which("ffmpeg") else "opencv"
    encoder = encode_ffmpeg if backend == "ffmpeg" else encode_opencv
    try:
        encoder(out_path, frames, size, fps=args.fps, hold=args.hold, crf=args.crf)
    except Exception as exc:
        result["status"] = f"失败：{backend} 编码出错 {exc!r}"
        return result

    failure = verify_video(out_path, expected_frames=len(frames) * args.hold)
    if failure is not None:
        result["status"] = f"失败：{failure}（已保留原帧）"
        return result

    result["video"] = str(out_path)
    if args.keep:
        result["status"] = f"转换完成（--keep 保留原帧，{backend}）"
        return result
    freed = delete_frames(frames, stream_dir)
    result["freed"] = freed
    result["status"] = f"转换完成并删帧（{backend}）"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把逐帧图像目录按顺序合成 mp4 并删除原图像（见模块 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", type=Path, help="待处理目录（可含多层 episode/相机子目录）")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="播放帧率，默认 25（与 RoboDojo 仿真视频同惯例）")
    parser.add_argument("--hold", type=int, default=1,
                        help="每帧重复写 N 遍，用来表达真实时长（h=30 的稀疏帧用 --hold 30），默认 1")
    parser.add_argument("--crf", type=int, default=23, help="libx264 质量，默认 23（越小越清晰）")
    parser.add_argument("--codec", choices=("auto", "ffmpeg", "opencv"), default="auto",
                        help="编码后端，默认 auto（有 ffmpeg 用 libx264，否则 cv2 mp4v）")
    parser.add_argument("--min-frames", type=int, default=2,
                        help="少于该帧数的一路不转换也不删帧，默认 2")
    parser.add_argument("--keep", action="store_true", help="只转视频，不删原帧")
    parser.add_argument("--overwrite", action="store_true", help="目标 mp4 已存在时覆盖")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写不删")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.dir = args.dir.expanduser()
    if not args.dir.is_dir():
        print(f"输入目录不存在或不是目录: {args.dir}", file=sys.stderr)
        return 2
    if args.hold < 1:
        print("--hold 必须 >= 1", file=sys.stderr)
        return 2
    exts = set(IMAGE_EXTS)
    streams = find_streams(args.dir, exts)
    if not streams:
        print(f"未找到任何图像目录（扩展名 {sorted(exts)}）: {args.dir}", file=sys.stderr)
        return 1

    print(
        f"[images_to_video] 根目录 {args.dir}\n"
        f"  发现 {len(streams)} 路图像目录 · fps={args.fps} · hold={args.hold} · "
        f"codec={args.codec} · {'预演' if args.dry_run else ('只转不删' if args.keep else '转完删帧')}",
        flush=True,
    )
    results = [convert_stream(stream, args, exts) for stream in streams]

    ok = [r for r in results if r["video"]]
    skipped = [r for r in results if not r["video"]]
    total_frames = sum(r["frames"] for r in ok)
    total_freed = sum(r["freed"] for r in results)
    for result in results:
        mark = "OK  " if result["video"] else "SKIP"
        print(f"  [{mark}] {result['dir']}  ({result['frames']} 帧)  {result['status']}", flush=True)
    print(
        f"\n完成：{len(ok)} 路/{total_frames} 帧 -> mp4"
        + (f"，释放 {_format_bytes(total_freed)}" if total_freed else "")
        + (f"，跳过 {len(skipped)} 路" if skipped else ""),
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
