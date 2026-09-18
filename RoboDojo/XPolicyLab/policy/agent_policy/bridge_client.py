"""HTTP client for the local Codex bridge, plus image encoding.

Deliberately stdlib-only for transport (``urllib.request``): the policy server
runs inside an existing conda environment (XVLA) and we must not add packages to
it. Image encoding uses PIL, which that environment already has.

This module is transport and nothing else. It does not decide what a turn says
(the bridge assembles the text) and it does not decide what the model may reply
(the bridge validates the reply against the schema), so what goes in is the
observation packet ``observation.build_request`` produced and what comes out is
a decision in the shape ``protocol.parse_decision`` expects.

The bridge lives on the operator's Mac and is reached over an SSH reverse
tunnel, so every call here has a hard wall-clock budget. Requests never retry:
a timeout means a Codex turn that already cost tokens, and retrying would both
double the latency and burn quota.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
}


def encode_image(rgb: Any, *, quality: int = 88) -> tuple[bytes, str, str]:
    """Encode an HWC uint8 RGB array as JPEG. Returns ``(payload, mime, extension)``.

    JPEG rather than PNG because the images are photographs: PNG at 640x480 runs
    300-600 kB, JPEG about a tenth of that, which matters when the payload
    crosses a tunnel.

    There is deliberately no resize here. An earlier version bounded the longest
    side at 480 px to save image tokens; that was removed because the frames are
    wanted at full resolution elsewhere in the pipeline and because cropping a
    view before the policy sees it throws away detail the model is choosing a
    target pose from. What is sent is the simulator's own frame, at the
    simulator's own resolution -- and it is what the bridge records, byte for
    byte, so a run can be audited against what the model was actually shown.
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

    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue(), "image/jpeg", "jpg"


def detect_image_format(payload: bytes) -> str | None:
    """Sniff the magic bytes so a truncated or mislabelled payload is caught."""
    for magic, name in _MAGIC.items():
        if payload.startswith(magic):
            return name
    return None


def build_image_payload(name: str, rgb: Any, *, quality: int) -> dict[str, str]:
    """One camera view as the request carries it.

    The declared ``mime`` is not taken on trust at the far end: the bridge sniffs
    the magic bytes and refuses a payload that disagrees with its label, so a
    truncated frame is caught before it reaches the model as a corrupt image.
    """
    payload, mime, _ = encode_image(rgb, quality=quality)
    if detect_image_format(payload) is None:
        raise ValueError(f"encoder produced an unrecognised image for view {name!r}")
    return {
        "name": str(name),
        "mime": mime,
        "b64": base64.b64encode(payload).decode("ascii"),
    }


@dataclass
class BridgeResult:
    ok: bool
    error_kind: str = ""
    error: str = ""
    decision: dict[str, Any] | None = None
    action_chunk: list[dict[str, Any]] | None = None
    continuation: dict[str, Any] | None = None


class BridgeClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _request(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/v1/decide", data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("X-Bridge-Token", self.token)
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            body = response.read()
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"bridge returned a non-JSON body ({len(body)} bytes)") from exc
        if not isinstance(value, dict):
            raise ValueError("bridge returned a non-object JSON body")
        return value

    # ------------------------------------------------------------------ #
    # decision
    # ------------------------------------------------------------------ #
    def decide(self, packet: dict[str, Any], *, timeout_s: float) -> BridgeResult:
        """Send one observation packet and read back one decision.

        Never raises for a remote-side problem: a failure is returned as an
        ``ok=False`` result so the caller can degrade to a hold action instead of
        letting an exception kill the episode.
        """
        try:
            body = self._request(packet, timeout_s)
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
            )
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return BridgeResult(
                ok=False,
                error_kind="bridge_unreachable",
                error=f"{type(exc).__name__}: {exc}",
            )
        except ValueError as exc:
            return BridgeResult(
                ok=False,
                error_kind="bridge_bad_response",
                error=str(exc),
            )

        if not body.get("ok"):
            return BridgeResult(
                ok=False,
                # A 200 that says ok=false without naming a kind is the bridge
                # breaking its own contract, not a Codex-side failure: there is
                # nothing to report but "the response was wrong".
                error_kind=str(body.get("error_kind") or "bridge_bad_response"),
                error=str(body.get("error") or "unknown bridge failure"),
            )
        chunk = body.get("action_chunk")
        if not isinstance(chunk, list) or not chunk:
            return BridgeResult(ok=False, error_kind="bridge_bad_response", error="bridge returned no action_chunk")
        return BridgeResult(
            ok=True,
            decision=body.get("decision"),
            action_chunk=chunk,
            continuation=body.get("continuation"),
        )
