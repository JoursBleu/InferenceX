#!/bin/bash
# InferenceX Benchmark Sweep - 在已运行的sglang容器中执行
# 使用InferenceX原生 run_benchmark_serving 函数
#
# 前提: sglang 服务器已在容器内 port 8888 运行
# 用法: bash run_sweep.sh <container_name>

set -e

CONTAINER="${1:-sglang-bench}"

docker exec $CONTAINER bash -c '
set -e
source /workspace/benchmarks/benchmark_lib.sh

MODEL=/models/DeepSeek-R1-0528
TP=8
ISL=1024
OSL=1024
RANDOM_RANGE_RATIO=0.8
PORT=8888

echo "====== SERVER READY, starting benchmark sweep ======"

for CONC in 1 2 4 8 16 32 64; do
    echo ""
    echo "====== Running CONC=$CONC at $(date +%H:%M:%S) ======"
    RESULT_FILENAME="dsr1_fp8_mi355x_sglang_tp8_isl${ISL}_osl${OSL}_conc${CONC}"
    NUM_PROMPTS=$((CONC * 10))
    if [ $NUM_PROMPTS -lt 50 ]; then NUM_PROMPTS=50; fi

    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir /workspace/

    echo "====== CONC=$CONC done at $(date +%H:%M:%S) ======"
done

echo ""
echo "====== ALL BENCHMARKS COMPLETE ======"
echo "====== RESULTS SUMMARY ======"
for CONC in 1 2 4 8 16 32 64; do
    RESULT_FILE="/workspace/dsr1_fp8_mi355x_sglang_tp8_isl${ISL}_osl${OSL}_conc${CONC}.json"
    if [ -f "$RESULT_FILE" ]; then
        python3 -c "
import json
with open(\"$RESULT_FILE\") as f:
    d = json.load(f)
tput_per_gpu = d[\"total_token_throughput\"] / $TP
print(f\"CONC={d.get(\"max_concurrency\", \"$CONC\"):>3}  output_tput={d[\"output_throughput\"]:8.1f}  total_tput={d[\"total_token_throughput\"]:8.1f}  tput/gpu={tput_per_gpu:8.1f}  mean_ttft={d[\"mean_ttft_ms\"]:8.1f}ms  mean_tpot={d[\"mean_tpot_ms\"]:6.1f}ms  p99_ttft={d[\"p99_ttft_ms\"]:8.1f}ms\")
" 2>/dev/null || echo "CONC=$CONC: parse error"
    else
        echo "CONC=$CONC: no result file"
    fi
done
'
