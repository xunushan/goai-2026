"""从 HF 数据集仓库下载指定子目录 (数据集) 的文件。

用法（下载非视频文件 meta+data）:
    conda run -n lerobot python scripts/download_real_lerobot.py --only-nonvideo
    conda run -n lerobot python scripts/download_real_lerobot.py --only-nonvideo --dry-run

下载指定视频文件:
    conda run -n lerobot python scripts/download_real_lerobot.py --videos \
        videos/observation.images.cam_high/chunk-000/file-000.mp4,...
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "tianSeconds/goai_2026_lerobot"
SUB = "real_lerobot_v30_ee"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO, help="HF 数据集仓库 id")
    parser.add_argument("--sub", default=SUB,
                        help="仓库内子目录 (数据集名), 如 real_lerobot_v30_ee")
    parser.add_argument("--only-nonvideo", action="store_true",
                        help="只下载 meta/ 与 data/ 下的非视频文件")
    parser.add_argument("--videos", default=None,
                        help="逗号分隔的视频文件相对路径 (如 "
                             "videos/observation.images.cam_high/chunk-000/file-000.mp4,"
                             "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4)")
    parser.add_argument("--dry-run", action="store_true", help="只列出匹配文件，不下载")
    args = parser.parse_args()
    repo, sub = args.repo, args.sub

    if args.videos:
        patterns = [f"{sub}/{v}" for v in args.videos.split(",") if v.strip()]
    elif args.only_nonvideo:
        patterns = [f"{sub}/meta/**", f"{sub}/data/**"]
    else:
        patterns = [f"{sub}/**"]

    if args.dry_run:
        from huggingface_hub import list_repo_tree
        from huggingface_hub.hf_api import RepoFile
        files = [
            r.path for r in list_repo_tree(repo, repo_type="dataset", recursive=True)
            if isinstance(r, RepoFile)
        ]
        import fnmatch
        for pat in patterns:
            for f in sorted(files):
                if fnmatch.fnmatch(f, pat):
                    print(f)
        return

    p = snapshot_download(
        repo_id=repo,
        allow_patterns=patterns,
        local_dir=str(LOCAL_DIR),
        repo_type="dataset",
    )
    print("downloaded to:", p)


if __name__ == "__main__":
    main()
