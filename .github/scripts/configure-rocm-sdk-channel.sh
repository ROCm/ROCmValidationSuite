#!/usr/bin/env bash
# Configure ROCm SDK channel + ROCM_VERSION for build_packages_local.sh (CI).
# Uses POSIX case for release/* refs (safe under dash/sh and bash).
set -euo pipefail

GITHUB_ENV="${GITHUB_ENV:-/dev/null}"

NIGHTLY_BASE="${ROCM_SDK_NIGHTLY_BASE_URL:-}"
NIGHTLY_IDX="${ROCM_SDK_NIGHTLY_INDEX_URL:-}"
[ -z "$NIGHTLY_BASE" ] && NIGHTLY_BASE='https://rocm.nightlies.amd.com/tarball-multi-arch'
[ -z "$NIGHTLY_IDX" ] && NIGHTLY_IDX='https://rocm.nightlies.amd.com/tarball-multi-arch/'

REL_URL="${ROCM_SDK_RELEASE_URL:-}"
[ -z "$REL_URL" ] && REL_URL='https://repo.amd.com/rocm/tarball/'
REL_BASE="${ROCM_SDK_RELEASE_BASE_URL:-}"
[ -z "$REL_BASE" ] && REL_BASE="${REL_URL%/}"

EVENT="${GITHUB_EVENT_NAME:-}"
REF="${GITHUB_REF:-}"
IN_VER="${INPUT_ROCM_VERSION:-}"
VAR_VER="${VAR_ROCM_VERSION:-}"
IN_GPU="${INPUT_GPU_FAMILY:-}"

release_build() {
  {
    echo "ROCM_SDK_CHANNEL=release"
    echo "ROCM_SDK_RELEASE_URL=$REL_URL"
    echo "ROCM_SDK_BASE_URL=$REL_BASE"
    echo "ROCM_SDK_RELEASE_BASE_URL=$REL_BASE"
    echo "ROCM_SDK_INDEX_URL="
  } >> "$GITHUB_ENV"
}

nightly_build() {
  {
    echo "ROCM_SDK_CHANNEL=nightly"
    echo "ROCM_SDK_RELEASE_URL="
    echo "ROCM_SDK_BASE_URL=$NIGHTLY_BASE"
    echo "ROCM_SDK_INDEX_URL=$NIGHTLY_IDX"
  } >> "$GITHUB_ENV"
}

format_build() {
  {
    echo "ROCM_SDK_CHANNEL=auto"
    echo "ROCM_SDK_RELEASE_URL=$REL_URL"
    echo "ROCM_SDK_INDEX_URL=$NIGHTLY_IDX"
    echo "ROCM_SDK_BASE_URL=$NIGHTLY_BASE"
    echo "ROCM_SDK_NIGHTLY_BASE_URL=$NIGHTLY_BASE"
    echo "ROCM_SDK_NIGHTLY_INDEX_URL=$NIGHTLY_IDX"
    echo "ROCM_SDK_RELEASE_BASE_URL=$REL_BASE"
  } >> "$GITHUB_ENV"
}

# ROCM_VERSION: schedule clears; dispatch input; push/PR use repo variable when set.
if [ "$EVENT" = "schedule" ]; then
  echo "ROCM_VERSION=" >> "$GITHUB_ENV"
  echo "Scheduled run: build script will auto-fetch latest ROCm version"
elif [ -n "$IN_VER" ]; then
  echo "ROCM_VERSION=$IN_VER" >> "$GITHUB_ENV"
  echo "Using workflow_dispatch ROCm version: $IN_VER"
elif [ -n "$VAR_VER" ]; then
  echo "ROCM_VERSION=$VAR_VER" >> "$GITHUB_ENV"
  echo "Using repository variable ROCM_VERSION: $VAR_VER"
fi

if [ -n "$IN_GPU" ]; then
  echo "GPU_FAMILY=$IN_GPU" >> "$GITHUB_ENV"
fi

if [ "$EVENT" = "schedule" ]; then
  nightly_build
  echo "ROCm SDK channel: nightly (scheduled — latest nightly)"
elif [ "$EVENT" = "pull_request" ]; then
  release_build
  echo "ROCm SDK channel: release (pull request — latest X.Y.Z)"
elif [ "$EVENT" = "push" ]; then
  case "$REF" in
    refs/heads/master|refs/heads/main|refs/heads/release/*)
      release_build
      echo "ROCm SDK channel: release (push to main or release/* — includes merge)"
      ;;
    *)
      nightly_build
      echo "ROCm SDK channel: nightly (push to feature branch)"
      ;;
  esac
elif [ "$EVENT" = "workflow_dispatch" ]; then
  if [ -n "$IN_VER" ] || [ -n "$VAR_VER" ]; then
    format_build
    echo "ROCm SDK: manual — tarball base chosen by version format (input or vars.ROCM_VERSION)"
  else
    nightly_build
    echo "ROCm SDK channel: nightly (manual, no version pin)"
  fi
else
  nightly_build
  echo "ROCm SDK channel: nightly (fallback)"
fi
