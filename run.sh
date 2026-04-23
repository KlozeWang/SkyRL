#!/usr/bin/env bash

set -euo pipefail

export TEACHER_API_KEY="${TEACHER_API_KEY:-111}"

bash examples/train/mini_swe_agent/run_mini_swe_8B_8xa100.sh \
  generator.teacher.enabled=true \
  generator.teacher.base_url="${TEACHER_BASE_URL:-http://4.155.72.80:10189}" \
  generator.teacher.model="${TEACHER_MODEL:-gpt-5.2}" \
  generator.teacher.tool_env_mode="${TEACHER_TOOL_ENV_MODE:-shared}" \
  generator.teacher.max_turns="${TEACHER_MAX_TURNS:-20}" \
  generator.teacher.output_token_penalty_coef="${TEACHER_OUTPUT_TOKEN_PENALTY_COEF:-0.01}" \
  "$@"
