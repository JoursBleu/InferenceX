# sglang DeepSeek-R1-0528 权重加载优化分析

## 问题描述

在 8x MI355X 上用 sglang TP=8 加载 DeepSeek-R1-0528 (671B MoE, FP8) 时，**TP0 和 TP7 的权重加载时间是其他 rank 的 14 倍**：

| TP Rank | 加载时间 (s) | 倍率 |
|---------|-------------|------|
| TP1 | 263 | 1.0x |
| TP2 | 261 | 1.0x |
| TP3 | 263 | 1.0x |
| TP4 | 279 | 1.07x |
| TP5 | 280 | 1.07x |
| TP6 | 279 | 1.07x |
| **TP0** | **3736** | **14.3x** |
| **TP7** | **3779** | **14.5x** |

端到端加载时间 = max(所有rank) = **63 分钟**。

## 根因分析

### 排除的假设

1. **❌ CPU 核心数不足**
   - 每个 TP worker 绑定 24 CPU 核心 (192/8)
   - `set_gpu_proc_affinity()` in `srt/utils/common.py:2227`
   - TP0: CPUs 0-23, TP1: CPUs 24-47, ..., TP7: CPUs 168-191

2. **❌ Shared experts fusion 导致工作不均**
   - `num_fused_shared_experts=1` 对所有 rank 一致
   - 每个 rank 都有 257 experts (256 routed + 1 shared)
   - EP size = 1，不做 expert parallelism

3. **❌ post_load_weights / process_weights_after_loading 慢**
   - Timing 显示 post_load_weights 只需 0.24-0.32s（所有 rank）
   - process_weights_after_loading (FP8 normalize + shuffle) 只需 0.18-0.22s

### 确认的根因

**mmap page fault 竞争**。详细 timing 数据：

```
model.load_weights() 内部阶段分解:
  ├─ weights iteration (safetensor读取+分发):  7-12s (所有rank)
  ├─ ThreadPoolExecutor futures (异步H2D copy): ~250s (TP1-6), ~3700s (TP0/TP7) ← 瓶颈!
  ├─ post_load_weights:                        0.3s (所有rank)
  └─ process_weights_after_loading:            0.2s (所有rank)
```

权重加载流水线：
1. `buffered_multi_thread_safetensors_weights_iterator` 用 mmap 打开 163 个 safetensor 文件
2. `deepseek_weight_loader.load_weights()` 遍历权重名，通过 `maybe_executor_submit()` 提交到 `ThreadPoolExecutor`
3. ThreadPoolExecutor 中的异步任务做: `loaded_weight.transpose(-2,-1)` (CPU) + `narrow()` (CPU) + `expert_data.copy_()` (H2D)
4. 这些操作触发 mmap page fault → 8 进程 × 32 线程 = 最多 256 并发 page fault

**TP0/TP7 所在的 NUMA 边界 GPU 进程**在高并发 mmap page fault 时被系统调度器/内存管理器"饿死"，导致 14x 延迟。

### 证据

- TP0/TP7 的 VMS 在加载期间从 990GB 缓慢降至 420GB（page 逐步回收）
- TP0/TP7 的 RSS 比其他 rank 大一倍 (~8GB vs ~5GB)
- py-spy 显示 TP0/TP7 主要时间在 `_load_w13` / `_weight_loader_impl` 中
- CPU 使用率 TP0/TP7 ~90-100%（vs TP1-6 完成后 ~15%空等）

## 优化方案验证

### 方案1: `--weight-loader-disable-mmap` ✅ 有效!

| TP Rank | 原始 (mmap) | disable-mmap | 变化 |
|---------|-----------|-------------|------|
| TP0 | **3736s** | 388s | **-89.6%** |
| TP1 | 263s | 381s | +44.9% |
| TP2 | 261s | 388s | +48.7% |
| TP3 | 263s | 390s | +48.3% |
| TP4 | 279s | 447s | +60.2% |
| TP5 | 280s | 445s | +58.9% |
| TP6 | 279s | 445s | +59.5% |
| TP7 | **3779s** | 447s | **-88.2%** |
| **端到端** | **3779s** | **447s** | **-88.2%** |

**效果**: 端到端从 63 分钟降到 7.5 分钟。所有 rank 均匀分布，消除了 TP0/TP7 倾斜问题。
**代价**: 每个 rank 比原来的快速 rank (TP1-6) 慢 ~60%，因为 `read()` 比 `mmap` 的文件读取慢。

### 方案2: `enable_multithread_load` + mmap ❌ 无效

使用 `--json-model-override-args '{"enable_multithread_load": true, "num_threads": 16}'`：
- TP1-6: 261-280s（和原始一样）
- TP0/TP7: 仍然 ~3700s+
- 结论: 多线程文件读取不能解决 mmap page fault 竞争问题

### 方案3: `enable_multithread_load` + `disable-mmap` ❌ 更慢

- 所有 rank > 500s，部分 rank 触发 watchdog timeout
- 原因: 16线程 × 8进程 = 128 并发 `read()` 导致更严重的 I/O 瓶颈

## 最佳实践

对于 **DeepSeek-R1 671B FP8 在 8x MI355X** 上，推荐：

```bash
python3 -m sglang.launch_server \
    --weight-loader-disable-mmap \
    --attention-backend aiter \
    --model-path /models/DeepSeek-R1-0528 \
    --tensor-parallel-size 8 \
    --chunked-prefill-size 196608 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs 64
```

加载时间: **~7.5 分钟** (vs 原始 63 分钟)

## 进一步优化方向

1. **Pre-shard 权重文件**: 预处理成 8 个独立的 per-rank safetensor 文件，每个 rank 只读自己的 shard (~80GB)
2. **Rank0 加载 + NCCL broadcast**: Rank0 读完所有权重后通过 XGMI (~896GB/s) broadcast 给其他 rank
3. **fastsafetensors**: 使用 `--load-format fastsafetensors`，专为分布式加载优化
4. **NUMA-aware 文件读取**: 确保每个 rank 从本地 NUMA 节点的内存读取文件

## 关键代码路径

- 权重加载入口: `sglang/srt/models/deepseek_common/deepseek_weight_loader.py:145`
- FusedMoE expert 分发: `sglang/srt/layers/moe/fused_moe_triton/layer.py:562`
- 文件读取迭代器: `sglang/srt/model_loader/weight_utils.py:830` (buffered_multi_thread_safetensors_weights_iterator)
- CPU 亲和性: `sglang/srt/utils/common.py:2227` (set_gpu_proc_affinity)
- FP8 后处理: `sglang/srt/layers/quantization/fp8.py:914` (process_weights_after_loading_block_quant)
- 异步提交: `sglang/srt/model_loader/utils.py:158` (maybe_executor_submit)

## 日期

2026-03-15
