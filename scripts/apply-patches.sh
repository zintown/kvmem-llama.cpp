#!/usr/bin/env bash
# Replay the Bonsai adapter patch without modifying the submodule pin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${KVMEM_LLAMA_DIR:-$ROOT/llama.cpp}"
PATCH="$ROOT/patches/llama-kvmem-current.patch"
RDNA2_FATTN="$ROOT/patches/0005-hip-rdna2-quantized-kv-fa-vec.patch"
HIP_SWAR="$ROOT/patches/0006-hip-swar-byte-ops.patch"
RDNA2_FA_OCC="$ROOT/patches/0007-hip-rdna2-fattn-vec-occupancy.patch"
PTQ1_DOT="$ROOT/patches/0008-hip-ptq1_0-reuse-q8_1-sum.patch"
cd "$LLAMA"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "Bonsai KVMem patch already applied"
elif git apply --check "$PATCH"; then
    git apply "$PATCH"
    echo "applied Bonsai KVMem patch to Prism"
else
    echo "Source does not match the Prism baseline; no files changed" >&2
    exit 1
fi

if git apply --reverse --check "$RDNA2_FATTN" 2>/dev/null; then
    echo "gfx1030 RDNA2 quantized-KV Flash Attention patch already applied"
else
    git apply --check "$RDNA2_FATTN"
    git apply "$RDNA2_FATTN"
    echo "applied gfx1030 RDNA2 quantized-KV Flash Attention patch"
fi

if git apply --reverse --check "$HIP_SWAR" 2>/dev/null; then
    echo "HIP SWAR byte-op patch already applied"
else
    git apply --check "$HIP_SWAR"
    git apply "$HIP_SWAR"
    echo "applied HIP SWAR byte-op patch"
fi

if git apply --reverse --check "$RDNA2_FA_OCC" 2>/dev/null; then
    echo "RDNA2 vector Flash Attention occupancy patch already applied"
else
    git apply --check "$RDNA2_FA_OCC"
    git apply "$RDNA2_FA_OCC"
    echo "applied RDNA2 vector Flash Attention occupancy patch"
fi

if git apply --reverse --check "$PTQ1_DOT" 2>/dev/null; then
    echo "HIP PTQ1_0 dot-product patch already applied"
else
    git apply --check "$PTQ1_DOT"
    git apply "$PTQ1_DOT"
    echo "applied HIP PTQ1_0 dot-product patch"
fi
