#!/usr/bin/env python3
"""Print the text blocks that the bridge generates for one experience."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from codex_agent.bridge.experience import render_demo_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=Path)
    args = parser.parse_args()
    demo = json.loads(args.demo.read_text(encoding="utf-8"))
    print("\n\n".join(render_demo_text(demo, str(demo["task_slug"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
