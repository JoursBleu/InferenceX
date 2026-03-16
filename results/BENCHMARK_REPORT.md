# DeepSeek-R1-0528 FP8 Benchmark Results - MI355X

## 硬件配置

| 项目 | 规格 |
|------|------|
| GPU | 8x AMD Instinct MI355X (gfx950) |
| VRAM | 288 GB HBM3e per GPU |
| 互联 | XGMI all-to-all |
| CPU | 2x AMD EPYC 9575F 64-Core (192 physical cores, HT off) |
| NUMA | 2 nodes: NUMA0=GPU0-3 (CPUs 0-95), NUMA1=GPU4-7 (CPUs 96-191) |

## 软件配置

| 项目 | 版本 |
|------|------|
| Docker | lmsysorg/sglang:v0.5.9-rocm700-mi35x |
| sglang | v0.5.10.dev4 |
| 模型 | deepseek-ai/DeepSeek-R1-0528 (671B MoE, FP8) |
| TP | 8 |
| 注意力后端 | aiter |
| KV Cache | fp8_e4m3 |

## 服务器启动参数

```bash
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_NUMA_BIND_V2=0

python3 -m sglang.launch_server \
    --attention-backend aiter \
    --model-path /models/DeepSeek-R1-0528 \
    --host=0.0.0.0 --port 8888 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --chunked-prefill-size 196608 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs 64 \
    --weight-loader-disable-mmap \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --numa-node 0 0 0 0 1 1 1 1
```

## Benchmark 方法

使用 InferenceX 标准方法 (`benchmarks/single_node/dsr1_fp8_mi355x.sh`)：
- `--request-rate inf` (infinite, 尽快发送)
- `--max-concurrency CONC` (限制并发数)
- `--num-warmups 2*CONC` (预热请求)
- `--random-range-ratio 0.8`
- `--num-prompts CONC*10`

## 结果汇总

### ISL=1024, OSL=1024

| CONC | Prompts | Output Tput (tok/s) | Total Tput (tok/s) | Tput/GPU (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) |
|-----:|--------:|--------------------:|-------------------:|-----------------:|---------------:|--------------:|---------------:|--------------:|
| 4 | 40 | 359.7 | 722.9 | **90.4** | 478.6 | 3136.5 | 10.3 | 11.7 |
| 8 | 80 | 696.2 | 1387.3 | **173.4** | 285.1 | 1733.5 | 10.9 | 12.6 |
| 16 | 160 | 1087.5 | 2186.6 | **273.3** | 321.5 | 1951.1 | 14.1 | 15.7 |
| 32 | 320 | 1585.5 | 3166.0 | **395.8** | 390.0 | 2367.1 | 19.3 | 22.4 |
| 64 | 640 | 2485.9 | 4972.9 | **621.6** | 622.4 | 4757.3 | 24.5 | 26.9 |

### ISL=1024, OSL=8192

| CONC | Prompts | Output Tput (tok/s) | Total Tput (tok/s) | Tput/GPU (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) |
|-----:|--------:|--------------------:|-------------------:|-----------------:|---------------:|--------------:|---------------:|--------------:|
| 4 | 40 | 395.3 | 445.7 | **55.7** | 318.3 | 1588.2 | 9.92 | 10.06 |
| 8 | 80 | 710.3 | 799.4 | **99.9** | 278.6 | 1690.9 | 10.95 | 11.14 |
| 16 | 160 | 1157.3 | 1302.5 | **162.8** | 296.4 | 1896.8 | 13.45 | 13.77 |
| 32 | 320 | 1744.1 | 1960.7 | **245.1** | 332.5 | 2285.5 | 17.90 | 18.19 |
| 64 | 640 | 2785.0 | 3132.7 | **391.6** | 576.7 | 4476.1 | 22.29 | 22.82 |

### ISL=8192, OSL=1024

| CONC | Prompts | Output Tput (tok/s) | Total Tput (tok/s) | Tput/GPU (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) |
|-----:|--------:|--------------------:|-------------------:|-----------------:|---------------:|--------------:|---------------:|--------------:|
| 4 | 40 | 289.3 | 2601.0 | **325.1** | 1099.5 | 3677.2 | 12.37 | 16.61 |
| 8 | 80 | 587.7 | 5228.2 | **653.5** | 578.5 | 3148.4 | 12.77 | 14.47 |
| 16 | 160 | 854.5 | 7711.2 | **963.9** | 781.4 | 4886.1 | 17.37 | 20.65 |
| 32 | 320 | 1114.1 | 9962.7 | **1245.3** | 1344.7 | 9565.3 | 26.67 | 33.11 |
| 64 | 640 | 1569.9 | 14150.6 | **1768.8** | 1803.6 | 17304.9 | 38.15 | 48.28 |

> **Tput/GPU** = total_token_throughput / tp_size (InferenceX 标准指标)

## 性能分析

### ISL=1024, OSL=1024 (等长输入输出)
- 吞吐量随 CONC 近乎线性增长（CONC 4→64 约 6.9x）
- TPOT 从 10.3ms (CONC=4) 增长到 24.5ms (CONC=64)，约 2.4x
- CONC=64 时 total throughput 接近 5000 tok/s

### ISL=1024, OSL=8192 (长输出 decode-heavy)
- 输出吞吐 CONC=64 达 2785 tok/s，但 total tput/GPU 较低（391.6）因输出占比大
- TPOT 非常稳定：9.9ms (CONC=4) → 22.3ms (CONC=64)
- Decode-heavy 场景下 TPOT 主导性能，TTFT 影响较小

### ISL=8192, OSL=1024 (长输入 prefill-heavy)
- Total tput/GPU 最高：CONC=64 达 1768.8 tok/s（因为 input tokens 占比大）
- Output tput 相对较低（prefill 阶段消耗大量计算）
- TTFT 较高（CONC=64 平均 1.8s, P99=17.3s），符合长输入预期
- TPOT 在高并发下增长显著：12.4ms → 38.2ms

### 跨场景对比（CONC=64）

| 场景 | Output Tput | Total Tput/GPU | Mean TPOT | 特点 |
|------|------------:|---------------:|----------:|------|
| 1k/1k | 2485.9 | 621.6 | 24.5ms | 均衡 |
| 1k/8k | 2785.0 | 391.6 | 22.3ms | Decode-heavy，output 吞吐高 |
| 8k/1k | 1569.9 | 1768.8 | 38.2ms | Prefill-heavy，total tput/GPU 高 |

## 日期

- 2026-03-15: ISL=1024/OSL=1024 基准测试
- 2026-03-16: ISL=1024/OSL=8192 + ISL=8192/OSL=1024 基准测试
- 机器: lekang-gpu-qswo7
