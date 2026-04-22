#!/usr/bin/env bash

set -euo pipefail
set -x

# Single-instance rollout debug runner matching the teacher defaults in repo-root run.sh.
# By default, this starts a standalone vLLM OpenAI-compatible server for LiteLLM.

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-111}"
export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-examples/train/mini_swe_agent/litellm.json}"

DATA_DIR="${DATA_DIR:-/mnt/xujiawang/skyrl/examples/swe/swe_gym_subset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/xujiawang/skyrl/examples/swe}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.parquet}"
MINISWE_TRAJ_DIR="${MINISWE_TRAJ_DIR:-$OUTPUT_ROOT/trajs/mini_swe_single_debug}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-900}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$MINISWE_TRAJ_DIR/vllm_${MODEL_NAME//\//_}_${VLLM_PORT}.log}"
START_VLLM="${START_VLLM:-1}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://$VLLM_HOST:$VLLM_PORT/v1}"

TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://4.155.72.80:10189}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt-5.2}"
TEACHER_OUTPUT_TOKEN_PENALTY_COEF="${TEACHER_OUTPUT_TOKEN_PENALTY_COEF:-0.05}"

mkdir -p "$MINISWE_TRAJ_DIR"

health_url() {
  printf 'http://%s:%s/health' "$VLLM_HOST" "$VLLM_PORT"
}

server_is_ready() {
  python - "$1" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_vllm() {
  local url
  url="$(health_url)"

  for _ in $(seq 1 "$VLLM_STARTUP_TIMEOUT_S"); do
    if server_is_ready "$url"; then
      return 0
    fi
    if [ "${VLLM_PID:-}" != "" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done

  return 1
}

cleanup_vllm() {
  if [ "${VLLM_PID:-}" = "" ]; then
    return 0
  fi
  if [ "$KEEP_VLLM_ALIVE" = "1" ]; then
    echo "Leaving vLLM server running with pid $VLLM_PID"
    return 0
  fi

  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}

trap cleanup_vllm EXIT

if [ "$START_VLLM" = "1" ]; then
  if server_is_ready "$(health_url)"; then
    echo "Using existing vLLM server at $(health_url)"
  else
    uv run --isolated --extra fsdp --extra miniswe \
      vllm serve "$MODEL_NAME" \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --dtype "$VLLM_DTYPE" \
      --enable-auto-tool-choice \
      --tool-call-parser "$VLLM_TOOL_CALL_PARSER" \
      >"$VLLM_LOG_FILE" 2>&1 &
    VLLM_PID=$!

    if ! wait_for_vllm; then
      echo "vLLM did not become healthy. Last log lines from $VLLM_LOG_FILE:" >&2
      tail -200 "$VLLM_LOG_FILE" >&2 || true
      exit 1
    fi
  fi
else
  echo "START_VLLM=0; expecting existing rollout server at $OPENAI_BASE_URL"
fi

uv run --isolated --extra miniswe \
  -m examples.train.mini_swe_agent.demo_single_rollout \
  --train-data "$TRAIN_DATA" \
  --model-name "$MODEL_NAME" \
  --miniswe-config-path "examples/train/mini_swe_agent/swebench.yaml" \
  --traj-dir "$MINISWE_TRAJ_DIR" \
  --teacher-enabled \
  --teacher-base-url "$TEACHER_BASE_URL" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-output-token-penalty-coef "$TEACHER_OUTPUT_TOKEN_PENALTY_COEF" \
  "$@"
