#!/usr/bin/env python3
"""彻底删除 HF 仓库（删除后校验 404，确保不残留占用空间）。

支持的输入：
1. 直接传仓库 URL（自动识别类型），/tree/main 等路径后缀会被忽略；
2. --repo-id + --repo-type 成对指定。

删除流程：repo_info 确认存在 -> delete_repo 删除 -> repo_info 再次确认
返回 404，任一环节失败即报错退出，避免"以为删了但还占空间"。

用法:
    conda run -n lerobot python tools/hf_repo_delete.py \
        https://huggingface.co/datasets/tianSeconds/goai_2026_lerobot \
        https://huggingface.co/tianSeconds/goai-2026-arx-flow-policy/tree/main

    # 显式指定 repo_id + 类型（可多次）
    conda run -n lerobot python tools/hf_repo_delete.py \
        --repo-id tianSeconds/goai_2026_lerobot --repo-type dataset

    # 只检查存在性与类型，不删除
    conda run -n lerobot python tools/hf_repo_delete.py \
        https://huggingface.co/datasets/tianSeconds/goai_2026_lerobot --dry-run

token 沿用本地 ~/.cache/huggingface 缓存凭据（参见 docs/archived/私人hf仓库操作指南.md），
不接收明文 token。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

#: URL 路径首段 -> repo_type
_URL_PREFIX = {"datasets": "dataset", "spaces": "space"}


@dataclass(frozen=True)
class Repo:
    repo_id: str
    repo_type: str  # model | dataset | space

    @property
    def url(self) -> str:
        prefix = "datasets" if self.repo_type == "dataset" else (
            "spaces" if self.repo_type == "space" else "")
        host = "huggingface.co"
        return f"https://{host}/{prefix}/{self.repo_id}" if prefix else f"https://{host}/{self.repo_id}"


def parse_url(url: str) -> Repo:
    """从 URL 解析 (repo_id, repo_type)。如 https://huggingface.co/datasets/a/b/tree/main。"""
    parsed = urlparse(url.strip())
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"无法从 URL 解析 repo_id: {url}")
    if parts[0] in _URL_PREFIX:
        repo_type = _URL_PREFIX[parts[0]]
        owner, name = parts[1], parts[2]
    else:
        # 默认 model；huggingface.co/{owner}/{name}[/...]
        repo_type = "model"
        owner, name = parts[0], parts[1]
    return Repo(f"{owner}/{name}", repo_type)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("urls", nargs="*", help="HF 仓库 URL（自动识别类型），或带 /tree/main 的完整 URL")
    parser.add_argument("--repo-id", action="append", default=[],
                        help="repo_id（owner/name），可多次指定，与 --repo-type 成对")
    parser.add_argument("--repo-type", action="append", default=[],
                        choices=["model", "dataset", "space"],
                        help="与 --repo-id 成对的类型")
    parser.add_argument("--dry-run", action="store_true",
                        help="只检查仓库存在性与类型，不删除")
    parser.add_argument("--yes", action="store_true",
                        help="跳过交互确认，直接删除")
    args = parser.parse_args(argv)

    repos: list[Repo] = [parse_url(u) for u in args.urls]
    if len(args.repo_id) != len(args.repo_type):
        parser.error("--repo-id 与 --repo-type 数量必须一致")
    repos += [Repo(rid, rt) for rid, rt in zip(args.repo_id, args.repo_type)]

    if not repos:
        parser.error("请提供仓库 URL 或 --repo-id/--repo-type")
    return args, repos


def check_exists(api: HfApi, repo: Repo) -> bool:
    """repo_info 是否仍能取到；404 视为已删除。"""
    try:
        api.repo_info(repo.repo_id, repo_type=repo.repo_type)
        return True
    except RepositoryNotFoundError:
        return False
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return False
        raise


def main(argv: list[str] | None = None) -> None:
    args, repos = parse_args(argv)
    api = HfApi()  # 用 ~/.cache/huggingface 缓存凭据

    print(f"待处理 {len(repos)} 个仓库:")
    for r in repos:
        print(f"  [{r.repo_type}] {r.repo_id}  ({r.url})")

    # 1. 删除前确认存在（剔除已不存在的，避免误报）
    existing: list[Repo] = []
    for r in repos:
        if check_exists(api, r):
            print(f"[前查] {r.repo_id} 存在")
            existing.append(r)
        else:
            print(f"[前查] {r.repo_id} 不存在（已删除？）")
    repos = existing

    if not repos:
        print("没有需要删除的仓库。")
        return

    if args.dry_run:
        print("[dry-run] 未删除。")
        return

    # 2. 交互确认
    if not args.yes:
        print("\n删除后不可恢复。输入仓库 owner/name 以确认，或直接回车取消：")
        if input("> ").strip().lower() != "yes":
            print("已取消。")
            return

    # 3. 删除
    for r in repos:
        api.delete_repo(r.repo_id, repo_type=r.repo_type)
        print(f"[删除] {r.repo_id} delete_repo 调用成功")

    # 4. 删除后校验 404
    failed = False
    for r in repos:
        if check_exists(api, r):
            print(f"[校验] ✗ {r.repo_id} 仍可访问！请人工检查")
            failed = True
        else:
            print(f"[校验] ✓ {r.repo_id} 已删除（repo_info 404）")
    if failed:
        sys.exit("部分仓库删除后仍存在，请人工检查。")
    print("全部删除完成。")


if __name__ == "__main__":
    main()
