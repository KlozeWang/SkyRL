#!/usr/bin/env bash

set -u

log() {
  printf '\n==> %s\n' "$*"
}

run() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'

  "$@" || {
    status=$?
    printf 'warning: command exited with status %s\n' "$status" >&2
    return 0
  }
}

kill_matching_processes() {
  local label="$1"
  local pattern="$2"

  if ! command -v pgrep >/dev/null 2>&1 || ! command -v pkill >/dev/null 2>&1; then
    printf 'warning: pgrep/pkill not found; skipping %s process cleanup\n' "$label" >&2
    return 0
  fi

  if ! pgrep -f "$pattern" >/dev/null 2>&1; then
    printf 'no %s processes found\n' "$label"
    return 0
  fi

  printf 'terminating %s processes matching: %s\n' "$label" "$pattern"
  pkill -TERM -f "$pattern" >/dev/null 2>&1 || true
  sleep 3

  if pgrep -f "$pattern" >/dev/null 2>&1; then
    printf 'force killing remaining %s processes\n' "$label"
    pkill -KILL -f "$pattern" >/dev/null 2>&1 || true
  fi
}

cleanup_ray() {
  log "Stopping Ray"

  if command -v ray >/dev/null 2>&1; then
    run ray stop --force
  else
    printf 'ray command not found; using process cleanup only\n'
  fi

  kill_matching_processes "Ray" \
    '(^|/)(raylet|gcs_server|plasma_store|dashboard|log_monitor|monitor.py)( |$)|ray::|ray[._-]worker|python.*ray'
}

cleanup_vllm() {
  log "Stopping vLLM"
  kill_matching_processes "vLLM" 'vllm|VLLM'
}

cleanup_podman() {
  log "Removing Podman containers"

  if ! command -v podman >/dev/null 2>&1; then
    printf 'podman command not found; skipping container cleanup\n'
    return 0
  fi

  local containers
  containers="$(podman ps -aq 2>/dev/null || true)"

  if [ -z "$containers" ]; then
    printf 'no Podman containers found\n'
    return 0
  fi

  # shellcheck disable=SC2086
  run podman stop -t 5 $containers
  # shellcheck disable=SC2086
  run podman rm -f $containers
}

main() {
  cleanup_ray
  cleanup_vllm
  cleanup_podman

  log "Cleanup complete"
}

main "$@"
