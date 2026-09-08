#!/usr/bin/env bash
# Build (and tag) the RHEL 9 ROCm+RVS runtime image from AMD nightly yum repos.
#
# Examples:
#   ./build-rocm-image.sh --channel nightly
#   ./build-rocm-image.sh --gpu-target multiarch
#   ./build-rocm-image.sh --gpu-target gfx942
#   ./build-rocm-image.sh --resolve-only
#   ./build-rocm-image.sh --from-tarball <ignored>   # accepted for rvs_nightly_docker.sh
#
# Nightly ROCm core listings are date-stamped:
#   https://nightly.repo.amd.com/rocm/core/packages/rhel9/<YYYYMMDD-id>/x86_64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_REPO="${RVS_NIGHTLY_DOCKER_IMAGE:-rvs-nightly-rocm-rhel9:latest}"
IMAGE_REPO="${IMAGE_REPO%%:*}"

CHANNEL="nightly"
GPU_TARGET="${GPU_TARGET:-multiarch}"
IMAGE_TAG=""
FROM_TARBALL=""
RESOLVE_ONLY=false
ROCM_VERSION="${ROCM_VERSION:-}"
ROCM_REPO_BASEURL="${ROCM_REPO_BASEURL:-}"
RVS_REPO_BASEURL="${RVS_REPO_BASEURL:-${RVS_NIGHTLY_RHEL9_RVS_REPO_BASEURL:-}}"
ROCM_GPG_KEY="${ROCM_GPG_KEY:-${RVS_NIGHTLY_RHEL9_GPG_KEY:-https://stable.repo.amd.com/rocm/gpg/packages.gpg}}"
ROCM_NIGHTLY_INDEX="${ROCM_NIGHTLY_INDEX:-${RVS_NIGHTLY_RHEL9_ROCM_REPO_INDEX:-https://nightly.repo.amd.com/rocm/core/packages/rhel9/}}"
RVS_NIGHTLY_REPO_DEFAULT="https://nightly.repo.amd.com/rocm/extras/rvs/packages/rhel9/x86_64"
RVS_STABLE_REPO_DEFAULT="https://stable.repo.amd.com/rocm/extras/rvs/packages/rhel9/x86_64"
ROCM_PACKAGE="${ROCM_PACKAGE:-}"
RVS_PACKAGE="${RVS_PACKAGE:-}"
ROCM_MAJOR=""
ROCM_SNAPSHOT=""
RVS_REPO_OVERRIDE=false

usage() {
  sed -n '2,12p' "$0"
  exit 1
}

fetch_url() {
  wget -q -O - "$1" 2>/dev/null || curl -fsSL --max-time 60 --retry 2 --retry-delay 2 "$1"
}

repo_has_metadata() {
  local base="${1%/}"
  curl -fsSL -o /dev/null --max-time 20 --retry 1 "${base}/repodata/repomd.xml" 2>/dev/null \
    || wget -q -O /dev/null --timeout=20 "${base}/repodata/repomd.xml" 2>/dev/null
}

resolve_rvs_repo() {
  if [ -n "$RVS_REPO_BASEURL" ]; then
    if repo_has_metadata "$RVS_REPO_BASEURL"; then
      return 0
    fi
    if [ "$RVS_REPO_OVERRIDE" = true ]; then
      echo "::error::RVS yum repo has no repodata: ${RVS_REPO_BASEURL}" >&2
      exit 1
    fi
    echo "::warning::RVS repo ${RVS_REPO_BASEURL} has no repodata; trying defaults" >&2
    RVS_REPO_BASEURL=""
  fi
  if repo_has_metadata "$RVS_NIGHTLY_REPO_DEFAULT"; then
    RVS_REPO_BASEURL="$RVS_NIGHTLY_REPO_DEFAULT"
    echo "::notice::Using nightly RVS extras repo"
    return 0
  fi
  if repo_has_metadata "$RVS_STABLE_REPO_DEFAULT"; then
    RVS_REPO_BASEURL="$RVS_STABLE_REPO_DEFAULT"
    echo "::warning::Nightly RVS extras yum is unpublished; using stable extras ${RVS_STABLE_REPO_DEFAULT}" >&2
    return 0
  fi
  echo "::error::No working RVS yum repo (tried nightly extras and ${RVS_STABLE_REPO_DEFAULT})" >&2
  exit 1
}

latest_rocm_snapshot() {
  local html
  html="$(fetch_url "$ROCM_NIGHTLY_INDEX")"
  printf '%s' "$html" | grep -oE '[0-9]{8}-[0-9]+' | sort -u | tail -n 1
}

gpu_target_is_multiarch() {
  case "${GPU_TARGET}" in
    multiarch|all|"") return 0 ;;
    *) return 1 ;;
  esac
}

latest_rocm_package_from_listing() {
  local listing pkg
  listing="$(fetch_url "${ROCM_REPO_BASEURL%/}/")"
  if gpu_target_is_multiarch; then
    # ISA-less meta: amdrocm10.1-10.1.0~DATE.x86_64.rpm (all GPU ISAs).
    pkg="$(printf '%s' "$listing" \
      | grep -oE 'amdrocm[0-9]+\.[0-9]+-[0-9]+\.[0-9]+\.[0-9]+~[0-9A-Za-z._~+-]+\.x86_64\.rpm' \
      | sed 's/-[0-9][0-9]*\.[0-9].*//' \
      | sort -uV | tail -n 1 || true)"
  else
    pkg="$(printf '%s' "$listing" \
      | grep -oE "amdrocm[0-9.]+-${GPU_TARGET}-[0-9A-Za-z._~+-]+\.x86_64\.rpm" \
      | sed 's/-[0-9][0-9A-Za-z._~+-]*\.x86_64\.rpm$//' \
      | sort -uV | tail -n 1 || true)"
  fi
  printf '%s\n' "$pkg"
}

emit_github() {
  if [ -z "${GITHUB_OUTPUT:-}" ]; then
    return 0
  fi
  {
    echo "rocm_snapshot=${ROCM_SNAPSHOT}"
    echo "rocm_version=${ROCM_VERSION}"
    echo "rocm_package=${ROCM_PACKAGE}"
    echo "rocm_major=${ROCM_MAJOR}"
    echo "rocm_repo_baseurl=${ROCM_REPO_BASEURL}"
    echo "rvs_repo_baseurl=${RVS_REPO_BASEURL}"
    echo "rvs_package=${RVS_PACKAGE}"
    echo "gpu_target=${GPU_TARGET}"
    echo "rocm_install_path=${ROCM_INSTALL_PATH:-/opt/rocm}"
    echo "tarball_name=amdrocm${ROCM_MAJOR}-rvs-nightly-rhel9"
    echo "tarball_url=${ROCM_REPO_BASEURL}"
  } >> "$GITHUB_OUTPUT"
}

resolve_nightly_repos() {
  if [ "$CHANNEL" != "nightly" ]; then
    echo "::error::RHEL 9 docker tests only support --channel nightly (got ${CHANNEL})" >&2
    exit 1
  fi

  if [ -z "$ROCM_REPO_BASEURL" ]; then
    ROCM_SNAPSHOT="$(latest_rocm_snapshot)"
    if [ -z "$ROCM_SNAPSHOT" ]; then
      echo "::error::Could not find a nightly ROCm snapshot under ${ROCM_NIGHTLY_INDEX}" >&2
      exit 1
    fi
    ROCM_REPO_BASEURL="${ROCM_NIGHTLY_INDEX%/}/${ROCM_SNAPSHOT}/x86_64"
  else
    ROCM_SNAPSHOT="$(printf '%s' "$ROCM_REPO_BASEURL" | grep -oE '[0-9]{8}-[0-9]+' | tail -n 1 || true)"
  fi

  if [ -z "$ROCM_PACKAGE" ]; then
    ROCM_PACKAGE="$(latest_rocm_package_from_listing)"
  fi
  if [ -z "$ROCM_PACKAGE" ]; then
    echo "::warning::Could not scrape package name from ${ROCM_REPO_BASEURL}; docker build will install amdrocm*-${GPU_TARGET}" >&2
    ROCM_PACKAGE=""
    ROCM_MAJOR="${ROCM_MAJOR:-10}"
    ROCM_VERSION="${ROCM_VERSION:-${ROCM_SNAPSHOT:-nightly}}"
  else
    if [[ "$ROCM_PACKAGE" =~ ^amdrocm([0-9]+) ]]; then
      ROCM_MAJOR="${BASH_REMATCH[1]}"
    else
      echo "::error::Cannot parse ROCm major from package ${ROCM_PACKAGE}" >&2
      exit 1
    fi
    ROCM_VERSION="${ROCM_VERSION:-${ROCM_SNAPSHOT:-$ROCM_PACKAGE}}"
  fi

  if [ -z "$RVS_PACKAGE" ]; then
    RVS_PACKAGE="amdrocm${ROCM_MAJOR}-rvs"
  fi

  resolve_rvs_repo

  if [[ "${ROCM_PACKAGE}" =~ ^amdrocm([0-9]+\.[0-9]+) ]]; then
    ROCM_INSTALL_PATH="${ROCM_INSTALL_PATH:-/opt/rocm/core-${BASH_REMATCH[1]}}"
  else
    ROCM_INSTALL_PATH="${ROCM_INSTALL_PATH:-/opt/rocm}"
  fi

  echo "Resolved RHEL 9 nightly repos"
  echo "  ROCm snapshot : ${ROCM_SNAPSHOT:-n/a}"
  echo "  ROCm repo     : ${ROCM_REPO_BASEURL}"
  echo "  ROCm package  : ${ROCM_PACKAGE:-amdrocm*-${GPU_TARGET}}"
  echo "  ROCm major    : ${ROCM_MAJOR}"
  echo "  RVS repo      : ${RVS_REPO_BASEURL}"
  echo "  RVS package   : ${RVS_PACKAGE}"
  echo "  GPU target    : ${GPU_TARGET}"
  echo "  ROCm path     : ${ROCM_INSTALL_PATH:-/opt/rocm}"
  emit_github
}

while [ $# -gt 0 ]; do
  case "$1" in
    --rocm-version) ROCM_VERSION="$2"; shift 2 ;;
    --from-tarball) FROM_TARBALL="$2"; shift 2 ;;
    --gpu-target)   GPU_TARGET="$2"; shift 2 ;;
    --channel)      CHANNEL="$2"; shift 2 ;;
    --tag)          IMAGE_TAG="$2"; shift 2 ;;
    --rocm-repo)    ROCM_REPO_BASEURL="$2"; shift 2 ;;
    --rvs-repo)     RVS_REPO_BASEURL="$2"; RVS_REPO_OVERRIDE=true; shift 2 ;;
    --rocm-package) ROCM_PACKAGE="$2"; shift 2 ;;
    --rvs-package)  RVS_PACKAGE="$2"; shift 2 ;;
    --resolve-only) RESOLVE_ONLY=true; shift ;;
    --fallback-latest-sdk) shift ;;
    -h|--help)      usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [ -n "$FROM_TARBALL" ]; then
  echo "::notice::--from-tarball ${FROM_TARBALL} ignored; RHEL 9 image uses nightly dnf repos"
fi

resolve_nightly_repos

if [ "$RESOLVE_ONLY" = true ]; then
  exit 0
fi

IMAGE_TAG="${IMAGE_TAG:-${IMAGE_REPO}:${ROCM_VERSION}}"

echo "Building docker image ${IMAGE_TAG}"
echo "  Base OS       : RHEL 9 (redhat/ubi9)"
echo "  ROCm version  : ${ROCM_VERSION}"

ROCM_INSTALL_PATH="${ROCM_INSTALL_PATH:-/opt/rocm}"

build_args=(
  --build-arg "ROCM_VERSION=${ROCM_VERSION}"
  --build-arg "ROCM_REPO_BASEURL=${ROCM_REPO_BASEURL}"
  --build-arg "RVS_REPO_BASEURL=${RVS_REPO_BASEURL}"
  --build-arg "ROCM_GPG_KEY=${ROCM_GPG_KEY}"
  --build-arg "GPU_TARGET=${GPU_TARGET}"
  --build-arg "ROCM_INSTALL_PATH=${ROCM_INSTALL_PATH}"
)
if [ -n "$ROCM_PACKAGE" ]; then
  build_args+=(--build-arg "ROCM_PACKAGE=${ROCM_PACKAGE}")
fi
if [ -n "$RVS_PACKAGE" ]; then
  build_args+=(--build-arg "RVS_PACKAGE=${RVS_PACKAGE}")
fi

docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${build_args[@]}" \
  -t "${IMAGE_TAG}" \
  "${SCRIPT_DIR}"

docker tag "${IMAGE_TAG}" "${IMAGE_REPO}:latest"
echo "::notice::Tagged ${IMAGE_TAG} and ${IMAGE_REPO}:latest"
