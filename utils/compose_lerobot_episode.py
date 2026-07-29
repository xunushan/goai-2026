#!/usr/bin/env python3
"""Export one LeRobot v3 episode as a lightweight three-camera mosaic."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass(frozen=True)
class VideoSegment:
    path: Path
    start: float
    duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose cam_high and two wrist cameras for one LeRobot v3 episode."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/lerobot_v30_ee"),
        help="LeRobot v3 dataset root (default: data/lerobot_v30_ee)",
    )
    parser.add_argument("--episode", type=int, required=True, help="Zero-based episode index")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 (default: outputs/episode_NNNN_mosaic.mp4)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 quality: lower is better/larger (default: 23)",
    )
    return parser.parse_args()


def load_episode_segments(
    dataset: Path, episode: int
) -> tuple[list[VideoSegment], int, float]:
    try:
        from lerobot.datasets import LeRobotDatasetMetadata
    except ImportError as exc:
        raise SystemExit(
            "This script must run in the same Python environment as LeRobot. "
            "Install it with: pip install lerobot"
        ) from exc

    info = json.loads((dataset / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    metadata = LeRobotDatasetMetadata(
        repo_id=dataset.name,
        root=dataset,
        local_files_only=True,
    )
    episodes = metadata.episodes
    episode_ids = list(episodes["episode_index"])
    try:
        row = episode_ids.index(episode)
    except ValueError as exc:
        raise ValueError(
            f"episode {episode} does not exist; available range is "
            f"{min(episode_ids)}..{max(episode_ids)}"
        ) from exc

    frame_count = int(episodes["length"][row])
    segments: list[VideoSegment] = []
    for camera in CAMERAS:
        prefix = f"videos/{camera}"
        chunk_index = int(episodes[f"{prefix}/chunk_index"][row])
        file_index = int(episodes[f"{prefix}/file_index"][row])
        start = float(episodes[f"{prefix}/from_timestamp"][row])
        end = float(episodes[f"{prefix}/to_timestamp"][row])
        relative_path = info["video_path"].format(
            video_key=camera,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        segments.append(
            VideoSegment(
                path=dataset / relative_path,
                start=start,
                duration=end - start,
            )
        )
    return segments, frame_count, fps


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = (
        args.output
        or Path("outputs") / f"episode_{args.episode:04d}_mosaic.mp4"
    ).resolve()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found in PATH")

    segments, frame_count, fps = load_episode_segments(dataset, args.episode)
    missing = [str(segment.path) for segment in segments if not segment.path.is_file()]
    if missing:
        raise FileNotFoundError("Missing camera video(s):\n" + "\n".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-y"]
    for segment in segments:
        command += [
            "-ss",
            f"{segment.start:.6f}",
            "-t",
            f"{segment.duration:.6f}",
            "-i",
            str(segment.path),
        ]

    # 640x480 overhead view above two 320x240 wrist views.
    command += [
        "-filter_complex",
        (
            "[0:v]scale=640:480,setsar=1[top];"
            "[1:v]scale=320:240,setsar=1[left];"
            "[2:v]scale=320:240,setsar=1[right];"
            "[left][right]hstack=inputs=2[bottom];"
            "[top][bottom]vstack=inputs=2[out]"
        ),
        "-map",
        "[out]",
        "-frames:v",
        str(frame_count),
        "-r",
        f"{fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]

    print(f"Episode {args.episode}: frames={frame_count}, fps={fps:g}")
    for camera, segment in zip(CAMERAS, segments, strict=True):
        print(
            f"  {camera}: {segment.path.relative_to(dataset)}, "
            f"{segment.start:.3f}s..{segment.start + segment.duration:.3f}s"
        )
    subprocess.run(command, check=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
