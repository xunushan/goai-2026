"""Write down what happened, without touching the pixels.

Two artefacts per episode, both under ``workspace/output/<episode_id>/``:

``observations/<camera>/{step_id:06d}_{request_id}.<ext>``
    The images, at the resolution they arrived in. The bytes are written exactly
    as they were received -- no resize, no re-encode, no recompress. That is the
    point: the record has to be evidence of what the model was shown, and an
    image that has been through an encoder is evidence of something slightly
    different. It also keeps this module free of Pillow, which matters because
    the bridge runs on the operator's machine and not in the policy server's
    environment.
``rollout.jsonl``
    One line per turn, decision or failure, flushed and fsynced as it is written.
    A run that is killed mid-episode still has every turn up to that point,
    which is the only way to tell "the model was slow" from "the bridge died".

The directory is keyed by ``step_id`` and ``request_id``.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import ImageInput, PolicyValidationError

OBSERVATIONS_DIRNAME = "observations"
SCRATCH_DIRNAME = "scratch"
ROLLOUT_FILENAME = "rollout.jsonl"

# Sniffed from the bytes rather than trusted from the packet: the extension is a
# claim about the file's contents, and the contents are right here.
_MAGIC = ((b"\xff\xd8\xff", "jpeg", ".jpg"), (b"\x89PNG\r\n\x1a\n", "png", ".png"))
_MIME_TYPE = {"image/jpeg": "jpeg", "image/png": "png"}

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def detect_magic(payload: bytes) -> tuple[str, str] | None:
    """``(type, extension)`` for a payload we recognise, else ``None``."""
    for prefix, kind, suffix in _MAGIC:
        if payload.startswith(prefix):
            return kind, suffix
    return None


def request_slug(request_id: str, *, limit: int = 40) -> str:
    """A filesystem-safe shortening of a request id, stable for the same id."""
    slug = _SLUG_UNSAFE.sub("-", request_id).strip("-.")
    return (slug or "request")[:limit]


def turn_file_stem(step_id: int, request_id: str) -> str:
    return f"{step_id:06d}_{request_slug(request_id)}"


def episode_dir(workspace: Path, episode_id: str) -> Path:
    """``<workspace>/output/<episode_id>``, created if missing.

    The episode id arrives from the adapter, so it is sanitised before it becomes
    a path component -- a stray ``/`` would otherwise let a caller write outside
    the output tree it was given.
    """
    slug = _SLUG_UNSAFE.sub("-", episode_id).strip("-.")
    if not slug:
        raise PolicyValidationError("episode_id does not yield a usable directory name")
    return workspace / "output" / slug


def prepare(workspace: Path, episode_id: str) -> Path:
    """Create the episode's directories and return them.

    ``scratch/`` is the one directory the model may write to, and it is created
    here rather than by Codex because an absent write target would show up as a
    mysterious permission failure inside a turn instead of as a startup error.
    """
    root = episode_dir(workspace, episode_id)
    (root / OBSERVATIONS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (root / SCRATCH_DIRNAME).mkdir(parents=True, exist_ok=True)
    return root


def store_images(
    images: Sequence[ImageInput],
    *,
    record_dir: Path,
    step_id: int,
    request_id: str,
) -> tuple[Path, ...]:
    """Write the packet's images verbatim and return where they landed."""
    written: list[Path] = []
    for image in images:
        sniffed = detect_magic(image.data)
        declared = _MIME_TYPE[image.mime]
        if sniffed is None:
            kind, suffix = declared, ".jpg" if declared == "jpeg" else ".png"
        else:
            kind, suffix = sniffed
            if kind != declared:
                raise PolicyValidationError(
                    f"image {image.name!r} is declared {image.mime} but its bytes are {kind}"
                )
        directory = record_dir / OBSERVATIONS_DIRNAME / image.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{turn_file_stem(step_id, request_id)}{suffix}"
        path.write_bytes(image.data)
        written.append(path)
    return tuple(written)


def relative_paths(paths: Iterable[Path], record_dir: Path) -> list[str]:
    """Paths as they are stored in the log: relative to the episode directory."""
    return [str(path.relative_to(record_dir)) for path in paths]


def append(record_dir: Path, payload: dict[str, Any]) -> Path:
    """Append one turn to ``rollout.jsonl`` and make sure it is on disk.

    ``fsync`` on every line is deliberate. The interesting failures here are
    crashes and kills, and those are exactly the cases where a buffered line
    would be lost -- the one line that says what was being attempted.
    """
    path = record_dir / ROLLOUT_FILENAME
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def turn_record(
    *,
    episode_id: str,
    request_id: str,
    step_id: int,
    turn_index: int,
    image_paths: Sequence[str],
    observation_state: dict[str, Any],
    ok: bool,
    decision: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    error_kind: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """The shape of one ``rollout.jsonl`` line.

    A failed turn gets a line too, and it carries no ``decision``: an invented
    action in the log would be indistinguishable from one the model actually
    chose, and the log exists precisely to be trusted afterwards.
    """
    return {
        "episode_id": episode_id,
        "request_id": request_id,
        "step_id": step_id,
        "turn_index": turn_index,
        "observation_state": observation_state,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "ok": ok,
        "decision": decision,
        "usage": usage,
        "latency_ms": latency_ms,
        "error_kind": error_kind,
        "error": error,
        "images": list(image_paths),
    }
