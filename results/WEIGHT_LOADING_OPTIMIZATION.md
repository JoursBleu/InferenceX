# DeepSeek-R1-0528 Weight Loading Optimization

## Hardware
- **GPU**: 8× AMD Instinct MI355X (gfx950), ~288GB HBM3e VRAM each, XGMI all-to-all
- **CPU**: 2× AMD EPYC 9575F 64-Core (192 CPUs), 2 NUMA nodes, 2TB RAM
- **Storage**: NVMe local disk

## Problem: TP0/TP7 14× Slower Than Other Ranks

When loading DeepSeek-R1-0528 (671B MoE, FP8, 163 safetensors, 1.3TB) with TP=8:
- TP1-6: **261-331s** (4.5-5.5 min)
- TP0/TP7: **3700+s** (63 min)
- Ratio: **~14× slower**

### Root Cause: mmap PTE Creation Contention

All 8 TP processes simultaneously `mmap()` the same 163 safetensor files. When `safetensors.safe_open()` accesses tensors, it triggers page faults. With 8 processes × 32 async threads each = 256 concurrent page faults, the kernel's virtual memory subsystem creates severe contention on page table entry (PTE) creation.

**Critical finding**: This is NOT a disk I/O or page cache problem. Even with 100% warm page cache (all 1.3TB pre-read into memory), TP0/TP7 remain slow. The bottleneck is in the kernel's mmap page-fault handling path — specifically PTE creation for MAP_PRIVATE mappings across 8 concurrent processes.

TP0 (GPU0, NUMA0) and TP7 (GPU7, NUMA1) are the first GPU on each NUMA socket and consistently get starved by the VM subsystem.

## Approaches Tested

| # | Approach | TP1-6 (s) | TP0/TP7 (s) | Total (s) | Result |
|---|---|---|---|---|---|
| 1 | Default mmap (baseline) | 261-331 | 3700+ | 3700+ | ❌ 14× asymmetry |
| 2 | `posix_fadvise(WILLNEED+SEQUENTIAL)` | ~similar | ~still slow | N/A | ❌ Kernel ignores hints |
| 3 | `mmap(MAP_POPULATE)` | 288-306 | still slow | N/A | ❌ VSZ balloons to 974GB |
| 4 | `--weight-loader-disable-mmap` (cold) | 533.9 | 533.9 | **534** | ✅ Uniform, no anomaly |
| 5 | Preload + mmap (warm cache) | 298-309 | still slow | N/A | ❌ PTE contention persists |
| 6 | **Preload + `--weight-loader-disable-mmap`** | 371-379 | 371-379 | **~407** | ✅ **Best solution** |
| 7 | Rank0-only staggered + mmap | N/A | N/A | >1300 est | ❌ Too slow (only 1 reader) |

## Recommended Solution: Parallel Preload + disable-mmap

Two-phase optimization:

### Phase 1: Parallel Page Cache Warmup (~28s)
Each TP rank cooperatively reads a disjoint 1/8 subset of the 163 checkpoint files using sequential `read()`. This warms the kernel page cache for the entire 1.3TB checkpoint at ~24 GB/s aggregate throughput (3 GB/s per rank).

### Phase 2: Weight Loading from Warm Cache (~379s)
All ranks load weights simultaneously using `safe_open(disable_mmap=True)`. Since the data is already in page cache, `read()` copies from RAM rather than disk, achieving **30% speedup** vs cold `disable-mmap` (379s vs 534s).

### End-to-end: ~407s (6.8 min) — uniform across all 8 ranks

Compared to:
- Baseline mmap: 3700+s for TP0/TP7 = **~9× faster**
- Plain disable-mmap: 534s = **24% faster**

## Implementation

### Patch 1: `python/sglang/srt/model_loader/loader.py`
In `DefaultModelLoader.load_model()`, before `load_weights_and_postprocess()`:
- Get TP rank/size
- If `SGLANG_PRELOAD_PAGE_CACHE=1` (default) and TP>1:
  - Each rank reads its partition of checkpoint files via sequential `read()`
  - Barrier synchronization
- Then proceed with normal weight loading

### Patch 2: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
Limit `ThreadPoolExecutor` workers via `SGLANG_WEIGHT_LOAD_WORKERS` env var (default: 8) to reduce concurrent page faults when mmap is used.

## Usage

```bash
# Optimal: preload + disable-mmap (recommended)
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-R1-0528 \
  --tp 8 \
  --trust-remote-code \
  --weight-loader-disable-mmap \
  --host 0.0.0.0 --port 30000

# Disable preload if needed (env var)
SGLANG_PRELOAD_PAGE_CACHE=0 python3 -m sglang.launch_server ...
```

## Key Insights

1. **mmap PTE contention is the root cause**, not disk I/O or page cache misses
2. **`disable-mmap` is the only reliable fix** — it avoids the kernel mmap path entirely
3. **Page cache preloading provides 30% speedup** for `disable-mmap` mode by warming the page cache before `read()` calls
4. The problem is specific to **multi-process mmap** on large files; single-process mmap works fine
5. TP0 and TP7 (first GPU on each NUMA socket) are consistently the victims
