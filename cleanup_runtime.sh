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

run_bounded() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    run timeout "$seconds" "$@"
    return 0
  fi

  printf 'warning: timeout command not found; running without bound: %s\n' "$*" >&2
  run "$@"
}

run_capture_bounded() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout "$seconds" "$@"
    return $?
  fi

  "$@"
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

kill_exact_processes() {
  local label="$1"
  shift

  if ! command -v pgrep >/dev/null 2>&1 || ! command -v pkill >/dev/null 2>&1; then
    printf 'warning: pgrep/pkill not found; skipping %s process cleanup\n' "$label" >&2
    return 0
  fi

  local name
  local found=0
  for name in "$@"; do
    if pgrep -x "$name" >/dev/null 2>&1; then
      found=1
      break
    fi
  done

  if [ "$found" = 0 ]; then
    printf 'no %s processes found\n' "$label"
    return 0
  fi

  printf 'terminating %s processes by exact name\n' "$label"
  for name in "$@"; do
    pkill -TERM -x "$name" >/dev/null 2>&1 || true
  done
  sleep 2

  for name in "$@"; do
    pkill -KILL -x "$name" >/dev/null 2>&1 || true
  done
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
  containers="$(timeout 10s podman ps -aq 2>/dev/null || true)"

  if [ -z "$containers" ]; then
    printf 'no Podman containers found before helper cleanup\n'
  else
    # shellcheck disable=SC2086
    run_bounded 30s podman stop -t 5 $containers
    # shellcheck disable=SC2086
    run_bounded 60s podman rm -f $containers

    local container
    for container in $containers; do
      remove_podman_container "$container"
    done
  fi

  kill_exact_processes "Podman helper" \
    podman \
    conmon \
    crun \
    slirp4netns \
    fuse-overlayfs \
    netavark \
    aardvark-dns

  containers="$(timeout 10s podman ps -aq 2>/dev/null || true)"

  if [ -z "$containers" ]; then
    printf 'no Podman containers found after helper cleanup\n'
    verify_podman_cleanup
    return 0
  fi

  for container in $containers; do
    remove_podman_container "$container"
  done

  kill_exact_processes "remaining Podman helper" \
    podman \
    conmon \
    crun \
    slirp4netns \
    fuse-overlayfs \
    netavark \
    aardvark-dns

  verify_podman_cleanup
}

remove_podman_container() {
  local container="$1"

  if run_capture_bounded 20s podman rm -f "$container"; then
    return 0
  fi

  printf 'podman rm failed for %s; killing inspected payload pid and retrying\n' "$container" >&2
  run_bounded 10s podman kill --signal KILL "$container"
  kill_podman_container_pid "$container"
  run_bounded 20s podman rm -f "$container"
}

kill_podman_container_pid() {
  local container="$1"
  local pid

  pid="$(run_capture_bounded 10s podman inspect --format '{{.State.Pid}}' "$container" 2>/dev/null || true)"
  if [ -z "$pid" ] || [ "$pid" = "0" ]; then
    return 0
  fi

  printf 'killing container payload pid=%s for %s\n' "$pid" "$container" >&2
  kill -KILL "-$pid" >/dev/null 2>&1 || true
  kill -KILL "$pid" >/dev/null 2>&1 || true
  sleep 1
}

verify_podman_cleanup() {
  log "Verifying Podman cleanup"

  if ! command -v podman >/dev/null 2>&1; then
    return 0
  fi

  printf 'Podman containers:\n'
  timeout 10s podman ps -a 2>/dev/null || true

  if command -v pgrep >/dev/null 2>&1; then
    printf 'Podman-related processes:\n'
    pgrep -a -x podman 2>/dev/null || true
    pgrep -a -x conmon 2>/dev/null || true
    pgrep -a -x crun 2>/dev/null || true
    pgrep -a -x slirp4netns 2>/dev/null || true
    pgrep -a -x fuse-overlayfs 2>/dev/null || true
    pgrep -a -x netavark 2>/dev/null || true
    pgrep -a -x aardvark-dns 2>/dev/null || true
  fi
}

main() {
  cleanup_ray
  cleanup_vllm
  cleanup_podman

  log "Cleanup complete"
}

main "$@"
