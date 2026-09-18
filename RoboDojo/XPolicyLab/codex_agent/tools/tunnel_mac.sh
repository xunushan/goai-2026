#!/usr/bin/env bash
set -euo pipefail

# Keep a Codex bridge reachable from a remote GPU machine.
#
#   tools/tunnel_mac.sh [ssh_host] [port]
#
# Run this on the Mac, where the Codex CLI and its credentials live. The policy
# server runs on the GPU machine and dials http://localhost:<port>, so the
# direction that works is a *reverse* forward: -R asks the remote sshd to listen
# on its loopback and hand connections back here. A plain -L would point the
# wrong way and would also expose the bridge to the whole remote network.
#
#   Mac                                 GPU machine
#   codex_bridge :8765  <── ssh -R ──   localhost:8765 ─→ policy server
#
# The bridge binds 127.0.0.1 on the Mac, so nothing but this tunnel can reach it.
#
# `ssh -N` opens no shell, so the tunnel stays up exactly as long as this
# script runs; Ctrl-C (or closing the terminal) tears it down.

if (( $# > 2 )); then
    echo "Usage: $0 [ssh_host] [port]" >&2
    exit 2
fi

ssh_host=${1:-issac-server}
port=${2:-8765}

if ! command -v ssh >/dev/null 2>&1; then
    echo "[tunnel] ssh not found" >&2
    exit 1
fi

echo "[tunnel] ${ssh_host} localhost:${port} -> this Mac's localhost:${port}"
echo "[tunnel] verify from the GPU machine with:"
echo "[tunnel]   curl -s http://localhost:${port}/healthz"
echo "[tunnel] Ctrl-C to close."

# ServerAliveInterval keeps the tunnel from being reaped by a NAT/idle timeout
# mid-episode: a dropped tunnel at decision 7 of 10 wastes the whole run.
exec ssh -N \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -R "${port}:localhost:${port}" \
    "${ssh_host}"
