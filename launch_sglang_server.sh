#!/bin/bash
# 启动 sglang 服务器的 Docker 容器
# 用法: bash launch_sglang_server.sh [container_name] [extra_args...]
#
# 示例:
#   bash launch_sglang_server.sh sglang-bench
#   bash launch_sglang_server.sh sglang-opt --weight-loader-disable-mmap

set -ex

CONTAINER_NAME="${1:-sglang-bench}"
shift 2>/dev/null || true
EXTRA_ARGS="$@"

DOCKER_IMAGE="lmsysorg/sglang:v0.5.9-rocm700-mi35x"
MODEL_DIR="/root/codes/models"
INFERENCEX_DIR="/root/codes/InferenceX"

# 清理旧容器
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# 环境变量
ENV_VARS="-e SGLANG_USE_AITER=1 -e RCCL_MSCCL_ENABLE=0 -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4"

# 启动容器 - 运行 InferenceX 原生脚本
docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    $ENV_VARS \
    -v "$MODEL_DIR":/models \
    -v "$INFERENCEX_DIR":/workspace \
    "$DOCKER_IMAGE" \
    bash -c "
        cd /workspace && \
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
            $EXTRA_ARGS
    "

echo "Container $CONTAINER_NAME started. Logs: docker logs -f $CONTAINER_NAME"
echo "Health check: docker exec $CONTAINER_NAME curl -sf http://localhost:8888/health"
