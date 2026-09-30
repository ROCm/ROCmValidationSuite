#!/usr/bin/env bash
# Reclaim runner disk before relocatable package builds.
# Does not uninstall SDKs or other host packages.
#
# Best-effort: a missing tool must not fail the caller.
# build-relocatable-packages.yml gates later build jobs on this step.

set -u

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    # -n: never block on a password prompt.
    sudo -n "$@"
  else
    echo "::warning::not root and sudo is unavailable; skipped: $*"
    return 0
  fi
}

report_disk() {
  local label="$1"
  echo "=== Disk ${label} ==="
  df -h / || df -h || true
}

report_disk "before cleanup"

# Download cache only. Does not remove installed packages.
if command -v apt-get >/dev/null 2>&1; then
  echo "apt-get clean"
  as_root apt-get clean \
    || echo "::warning::apt-get clean did not succeed"
else
  echo "skip: apt-get not available"
fi

# Drop unused Docker images. A daemon that is down, or a user who cannot
# talk to it, must not fail the job.
if command -v docker >/dev/null 2>&1; then
  echo "pruning unused Docker images"
  if ! docker image prune --all --force; then
    as_root docker image prune --all --force \
      || echo "::warning::docker image prune did not succeed"
  fi
else
  echo "skip: docker not installed"
fi

report_disk "after cleanup"
exit 0
