"""HTTP client for the local Codex bridge, plus image encoding.

Deliberately stdlib-only for transport (``urllib.request``): the policy server
runs inside an existing conda environment (XVLA) and we must not add packages to
it. Image encoding uses PIL, which that environment already has.

The bridge lives on the operator's Mac and is reached over an SSH reverse
tunnel, so every call here has a hard wall-clock budget. Requests never retry:
a timeout means a Codex turn that already cost tokens, and retrying would both
double the latency and burn quota.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
}


def encode_image(
    rgb: Any,
    *,
    quality: int = 88,
    max_width: int = 640,
) -> tuple[bytes, str, str]:
    """Encode an HWC uint8 RGB array as JPEG.

    Returns ``(payload, mime, extension)``. JPEG rather than PNG because the
    images are photographs: PNG at 640x480 runs 300-600 kB, JPEG about a tenth
    of that, which matters when the payload crosses a tunnel.
    """
    import numpy as np
    from PIL import Image

    array = np.asarray(rgb)
    if array.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3 and array.shape[0] in (3, 4):
        array = np.transpose(array[..., :3], (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    image = Image.fromarray(array, mode="RGB")
    if max_width and image.width > int(max_width):
        ratio = int(max_width) / float(image.width)
        image = image.resize(
            (int(max_width), max(1, int(round(image.height * ratio)))),
            Image.Resampling.BILINEAR,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue(), "image/jpeg", "jpg"


def detect_image_format(payload: bytes) -> str | None:
    """Sniff the magic bytes so a truncated or mislabelled payload is caught."""
    for magic, name in _MAGIC.items():
        if payload.startswith(magic):
            return name
    return None


def build_image_payload(name: str, rgb: Any, *, quality: int, max_width: int) -> dict[str, str]:
    payload, mime, _ = encode_image(rgb, quality=quality, max_width=max_width)
    if detect_image_format(payload) is None:
        raise ValueError(f"encoder produced an unrecognised image for view {name!r}")
    return {
        "name": str(name),
        "format": "image/jpeg",
        "mime": mime,
        "b64": base64.b64encode(payload).decode("ascii"),
    }


@dataclass
class BridgeResult:
    ok: bool
    error_kind: str = ""
    error: str = ""
    thread_id: str | None = None
    parsed: dict[str, Any] | None = None
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    runs_dir: str = ""
    http_status: int | None = None

    @property
    def retryable_fresh_thread(self) -> bool:
        """The bridge lost the thread: replay this turn on a brand-new thread."""
        return self.error_kind == "thread_not_found"


class BridgeClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        connect_timeout_s: float = 5.0,
        runs_dir: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.connect_timeout_s = float(connect_timeout_s)
        self.runs_dir = runs_dir

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _request(
        self, path: str, payload: dict[str, Any] | None, timeout_s: float
    ) -> tuple[int, dict[str, Any]]:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("X-Bridge-Token", self.token)
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            body = response.read()
            status = int(response.status)
        try:
            return status, json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"bridge returned a non-JSON body ({len(body)} bytes)") from exc

    def healthz(self) -> tuple[bool, str]:
        """Cheap liveness probe. Distinguishes 'bridge down' from 'Codex slow'."""
        try:
            _, body = self._request("/healthz", None, self.connect_timeout_s)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return False, str(exc)
        return bool(body.get("ok")), str(body.get("model", ""))

    # ------------------------------------------------------------------ #
    # decision
    # ------------------------------------------------------------------ #
    def decide(
        self,
        *,
        episode_id: str,
        thread_id: str | None,
        turn_index: int,
        prompt: str,
        schema: dict[str, Any] | None,
        images: list[dict[str, str]],
        timeout_s: float,
    ) -> BridgeResult:
        """Ask the bridge for one Codex decision.

        Never raises for a remote-side problem: a failure is returned as an
        ``ok=False`` result so the caller can degrade to a hold action instead of
        letting an exception kill the episode.
        """
        started = time.monotonic()
        payload = {
            "episode_id": str(episode_id),
            "thread_id": thread_id,
            "turn_index": int(turn_index),
            "prompt": prompt,
            "schema": schema,
            "images": images,
        }
        try:
            status, body = self._request("/v1/decide", payload, timeout_s)
        except urllib.error.HTTPError as exc:
            # The bridge reports a real failure (Codex timeout, bad request) with a
            # non-2xx status, so the body still carries the useful diagnosis.
            body: dict[str, Any] = {}
            try:
                body = json.loads(exc.read().decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001 - diagnostics must never mask the error
                body = {}
            if not isinstance(body, dict):
                body = {}
            return BridgeResult(
                ok=False,
                error_kind=str(body.get("error_kind") or "bridge_http_error"),
                error=str(body.get("error") or f"HTTP {exc.code}"),
                thread_id=body.get("thread_id"),
                latency_ms=int(body.get("latency_ms") or (time.monotonic() - started) * 1000),
                runs_dir=str(body.get("runs_dir") or ""),
                http_status=int(exc.code),
            )
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return BridgeResult(
                ok=False,
                error_kind="bridge_unreachable",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except ValueError as exc:
            return BridgeResult(
                ok=False,
                error_kind="bridge_bad_response",
                error=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
            )

        latency_ms = int(body.get("latency_ms", (time.monotonic() - started) * 1000))
        if not body.get("ok"):
            return BridgeResult(
                ok=False,
                error_kind=str(body.get("error_kind") or "codex_error"),
                error=str(body.get("error") or "unknown bridge failure"),
                thread_id=body.get("thread_id"),
                latency_ms=latency_ms,
                runs_dir=str(body.get("runs_dir") or ""),
                http_status=status,
            )
        return BridgeResult(
            ok=True,
            thread_id=body.get("thread_id"),
            parsed=body.get("parsed"),
            text=str(body.get("text") or ""),
            usage=dict(body.get("usage") or {}),
            latency_ms=latency_ms,
            runs_dir=str(body.get("runs_dir") or ""),
            http_status=status,
        )


def image_size_bytes(images: list[dict[str, str]]) -> int:
    """Total decoded size of a payload's images, for the bridge's size guard."""
    total = 0
    for entry in images:
        try:
            total += len(base64.b64decode(entry["b64"], validate=True))
        except (KeyError, binascii.Error, TypeError):
            continue
    return total
