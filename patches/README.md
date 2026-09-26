# Prism patch replay

`llama-kvmem-current.patch` targets Prism commit `9a9394a895b96003ca842a6041cb28ac49a108f7`.
It includes the KVMem adapter, rc3 server/template fixes, snapshot MTP restoration,
PTQ1 small-batch CUDA optimization and the required PDL synchronization fix.
The submodule stays pinned to upstream; commit changes as this cumulative patch,
not an unpublished submodule commit. GDN Record/Fold is not included.

Run `scripts/apply-patches.sh` after initializing a fresh submodule. The Windows
build script performs the same full-patch checks and accepts an already patched tree.
An older partially patched working tree must first have its old patch reversed;
preserve any local changes before updating. Do not stack a new cumulative patch
on the old cumulative patch.

## Kernel provenance

The kernel changes originate from [Prism PR #218](https://github.com/PrismML-Eng/llama.cpp/pull/218),
including SM120 compilation, shared-memory guards and four-column dispatch fixes
through `285542d98d37d0f07f491cd206aefa31f1848f33`. Imported patch revisions and hashes
are recorded in `prism-bonsai-ptq1-kernels.json`.

`prism-bonsai-ptq1-kernels.patch` and `prism-bonsai-ptq1-pdl-sync.patch` are retained
for attribution and independent reproduction. **Both are already included in the
cumulative patch; do not apply them again.** The local PDL fix waits for activation
quantization before the new PTQ1 mat-vec reads its input on SM90+.
See [measurements and numerical checks](../docs/bonsai-kernel-comparison.md).

`python scripts/prepare-bonsai-kernel-build.py` snapshots the already integrated
working sources to `build-win-bonsai-kernel/source` and adds the upstream numerical
test target. It checks cumulative-patch presence and refuses to overwrite snapshots.

`prism-bonsai-mtp-embedding.patch` is only for the independent Prism comparison;
do not apply it on top of the cumulative patch. `build-prism-reference.ps1` builds
its own pristine GGML so that the reference never reuses optimized KVMem libraries.

`llama-kvmem-rc3-reference.patch`, numbered patches and `*-upgrade.patch` target
older baselines; do not apply them to this Prism checkout.

`0005-hip-rdna2-quantized-kv-fa-vec.patch` is an RX 6000 (RDNA2) HIP fix. It
routes quantized-KV Flash Attention to the existing VEC kernel, avoiding the
zero-occupancy assertion (`max_blocks_per_sm > 0`) of the larger tile kernel.
It is replayed after the cumulative KVMem patch and is limited to RDNA2.

`0006-hip-swar-byte-ops.patch` replaces HIP's per-byte `__vsub4` and
`__vcmpne4` emulation with SWAR arithmetic. RDNA has no packed-byte subtract or
compare, and the IQ2/IQ3 dot products call both once per 4 weights. `__vsub4`
now wraps like CUDA's instead of saturating. On RX 6900 XT, IQ2/IQ3 mat-vec
kernels run 27-33% faster; `test-backend-ops -o MUL_MAT` passes 1253/1253.

`0007-hip-rdna2-fattn-vec-occupancy.patch` raises the VEC Flash Attention
kernel's assumed occupancy on RDNA2 from one to two blocks per WGP. HIP reports
one for the D=256 kernel, so decode attention was split into too few KV chunks
(about three waves per SIMD) to hide memory latency. With GQA 6 (Qwen3.8-27B:
24 Q / 4 KV heads, D=256) on RX 6900 XT, one decode FA call at 32K Q8 KV drops
from 858 to 509 us (8K: 223 -> 153, 64K: 2492 -> 1428).

`0008-hip-ptq1_0-reuse-q8_1-sum.patch` (Bonsai only) drops the second dp4a per
word that the HIP PTQ1_0 dot product spent summing activations; Q8_1 already
stores that sum in `ds.y`, as Q4_0 uses it. Decode on RX 6900 XT: 40.5 -> 42.8
tok/s. `scripts/apply-patches.sh` replays 0005-0008 after the Bonsai patch; all
are HIP-only or RDNA2-only and leave CUDA builds unchanged.
