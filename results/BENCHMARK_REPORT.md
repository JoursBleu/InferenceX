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
    --cuda-graph-max-bs 64
```

## Benchmark 方法

使用 InferenceX `run_all_conc.sh` 原生方法：
- `--request-rate inf` (infinite, 尽快发送)
- `--max-concurrency CONC` (限制并发数)
- `--num-warmups 2*CONC` (预热请求)
- `--random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.8`
- `--num-prompts CONC*10`

## 结果汇总

| CONC | Prompts | Output Tput (tok/s) | Total Tput (tok/s) | Tput/GPU (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) |
|-----:|--------:|--------------------:|-------------------:|-----------------:|---------------:|--------------:|---------------:|--------------:|
| 4 | 40 | 359.7 | 722.9 | **90.4** | 478.6 | 3136.5 | 10.3 | 11.7 |
| 8 | 80 | 696.2 | 1387.3 | **173.4** | 285.1 | 1733.5 | 10.9 | 12.6 |
| 16 | 160 | 1087.5 | 2186.6 | **273.3** | 321.5 | 1951.1 | 14.1 | 15.7 |
| 32 | 320 | 1585.5 | 3166.0 | **395.8** | 390.0 | 2367.1 | 19.3 | 22.4 |
| 64 | 640 | 2485.9 | 4972.9 | **621.6** | 622.4 | 4757.3 | 24.5 | 26.9 |

> **Tput/GPU** = total_token_throughput / tp_size (InferenceX 标准指标, 见 `utils/process_result.py:110`)

## 性能趋势

- 吞吐量随 CONC 增加近乎线性增长（CONC 4→64 约 6.9x）
- TPOT 从 10.3ms (CONC=4) 增长到 24.5ms (CONC=64)，约 2.4x
- TTFT P99 随 CONC 增长较快，从 3.1s 到 4.8s
- CONC=64 时 total throughput 接近 5000 tok/s

## 日期

2026-03-15，机器 lekang-gpu-dl810
