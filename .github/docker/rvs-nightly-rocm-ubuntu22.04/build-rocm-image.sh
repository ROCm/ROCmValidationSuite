#!/usr/bin/env bash
# Build (and tag) the ROCm runtime docker image for RVS nightly docker tests (Ubuntu 22.04).
# Uses the same multiarch / release tarball URLs as build_packages_local.sh.
#
# Examples:
#   ./build-rocm-image.sh --channel nightly --gpu-family multiarch
#   ./build-rocm-image.sh --rocm-version 10.1.0a20260819 --gpu-family multiarch
#   ./build-rocm-image.sh --from-tarball amdrocm10-rvs-1.7.10-r1001.20260819-Linux.tar.gz
#   ./build-rocm-image.sh --channel nightly --tag rvs-nightly-rocm-ubuntu22.04:latest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_REPO="${RVS_NIGHTLY_DOCKER_IMAGE:-rvs-nightly-rocm-ubuntu22.04:latest}"
IMAGE_REPO="${IMAGE_REPO%%:*}"

ROCM_VERSION=""
GPU_FAMILY="${GPU_FAMILY:-multiarch}"
CHANNEL="nightly"
IMAGE_TAG=""
ROCM_SDK_BASE_URL=""
FROM_TARBALL=""

NIGHTLY_INDEX="${ROCM_SDK_NIGHTLY_INDEX_URL:-https://rocm.nightlies.amd.com/tarball-multi-arch/}"
NIGHTLY_BASE="${ROCM_SDK_NIGHTLY_BASE_URL:-https://rocm.nightlies.amd.com/tarball-multi-arch}"
RELEASE_LIST="${ROCM_SDK_RELEASE_URL:-https://repo.amd.com/rocm/tarball/}"
RELEASE_BASE="${ROCM_SDK_RELEASE_BASE_URL:-https://repo.amd.com/rocm/tarball}"

usage() {
  sed -n '2,9p' "$0"
  exit 1
}

resolve_sdk_base() {
  local ver="$1"
  if echo "$ver" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+a[0-9]+'; then
    ROCM_SDK_BASE_URL="$NIGHTLY_BASE"
  elif echo "$ver" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    ROCM_SDK_BASE_URL="$RELEASE_BASE"
  else
    echo "::error::Unrecognized ROCm version format: $ver" >&2
    exit 1
  fi
}

fetch_latest_version() {
  local mode="$1"
  local listing_tmp versions
  listing_tmp="$(mktemp)"
  if [ "$mode" = "nightly" ]; then
    wget -q -O "$listing_tmp" "$NIGHTLY_INDEX"
    versions=$(grep -oE "therock-dist-linux-${GPU_FAMILY}-[0-9]+\.[0-9]+\.[0-9]+a[0-9]+" "$listing_tmp" \
      | sed "s|^therock-dist-linux-${GPU_FAMILY}-||" | sort -V | tail -1)
  else
    wget -q -O "$listing_tmp" "$RELEASE_LIST"
    versions=$(grep -oE "therock-dist-linux-${GPU_FAMILY}-[0-9]+\.[0-9]+\.[0-9]+" "$listing_tmp" \
      | sed "s|^therock-dist-linux-${GPU_FAMILY}-||" | sort -V | tail -1)
  fi
  rm -f "$listing_tmp"
  if [ -z "$versions" ]; then
    echo "::error::No SDK version found for ${GPU_FAMILY} (${mode})" >&2
    exit 1
  fi
  ROCM_VERSION="$versions"
}

resolve_rocm_from_tarball() {
  local name="$1"
  local base="${name##*/}"
  local major minor build_date exact listing_tmp sdk_file prefix

  if [[ "$base" != *-Linux.tar.gz ]]; then
    echo "::error::--from-tarball requires a *-Linux.tar.gz relocatable tarball; got: ${base}" >&2
    exit 1
  fi

  if [[ "$base" =~ -r([0-9]{2})([0-9]{2})\.([0-9]{8})-Linux\.tar\.gz$ ]]; then
    major=$((10#${BASH_REMATCH[1]}))
    minor=$((10#${BASH_REMATCH[2]}))
    build_date="${BASH_REMATCH[3]}"
  else
    echo "::error::Cannot parse ROCm version from tar tarball (expected ...-rMMmm.yyyymmdd-Linux.tar.gz): ${base}" >&2
    exit 1
  fi

  if [ "$CHANNEL" = "release" ]; then
    prefix="${major}.${minor}."
    listing_tmp="$(mktemp)"
    wget -q -O "$listing_tmp" "$RELEASE_LIST"
    ROCM_VERSION=$(grep -oE "therock-dist-linux-${GPU_FAMILY}-${prefix}[0-9]+" "$listing_tmp" \
      | sed "s|^therock-dist-linux-${GPU_FAMILY}-||" | sort -V | tail -1)
    rm -f "$listing_tmp"
    sdk_file="therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz"
    resolve_sdk_base "$ROCM_VERSION"
  else
    exact="${major}.${minor}.0a${build_date}"
    sdk_file="therock-dist-linux-${GPU_FAMILY}-${exact}.tar.gz"
    listing_tmp="$(mktemp)"
    wget -q -O "$listing_tmp" "$NIGHTLY_INDEX"
    if ! grep -qF "$sdk_file" "$listing_tmp"; then
      rm -f "$listing_tmp"
      echo "::error::No multiarch SDK ${exact} for tar ${base} (missing ${sdk_file} on ${NIGHTLY_INDEX})" >&2
      exit 1
    fi
    rm -f "$listing_tmp"
    ROCM_VERSION="$exact"
    ROCM_SDK_BASE_URL="$NIGHTLY_BASE"
  fi

  if [ -z "$ROCM_VERSION" ]; then
    echo "::error::No ${CHANNEL} SDK found for ROCm ${major}.${minor} (${GPU_FAMILY}) from tar ${base}" >&2
    exit 1
  fi
  echo "Resolved ROCm ${ROCM_VERSION} from tar tarball ${base} (r$(printf '%02d%02d' "$major" "$minor").${build_date})"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --rocm-version) ROCM_VERSION="$2"; shift 2 ;;
    --from-tarball) FROM_TARBALL="$2"; shift 2 ;;
    --gpu-family)   GPU_FAMILY="$2"; shift 2 ;;
    --channel)      CHANNEL="$2"; shift 2 ;;
    --tag)          IMAGE_TAG="$2"; shift 2 ;;
    -h|--help)      usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [ -n "$FROM_TARBALL" ]; then
  resolve_rocm_from_tarball "$(basename "$FROM_TARBALL")"
elif [ -z "$ROCM_VERSION" ]; then
  fetch_latest_version "$CHANNEL"
fi

resolve_sdk_base "$ROCM_VERSION"
IMAGE_TAG="${IMAGE_TAG:-${IMAGE_REPO}:${ROCM_VERSION}}"

echo "Building docker image ${IMAGE_TAG}"
echo "  ROCm version : ${ROCM_VERSION}"
echo "  GPU family   : ${GPU_FAMILY}"
echo "  SDK base URL : ${ROCM_SDK_BASE_URL}"
echo "  Base OS      : Ubuntu 22.04"

ROCM_INSTALL_PATH="${ROCM_INSTALL_PATH:-/opt/rocm/install}"

docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "ROCM_VERSION=${ROCM_VERSION}" \
  --build-arg "GPU_FAMILY=${GPU_FAMILY}" \
  --build-arg "ROCM_SDK_BASE_URL=${ROCM_SDK_BASE_URL}" \
  --build-arg "ROCM_INSTALL_PATH=${ROCM_INSTALL_PATH}" \
  -t "${IMAGE_TAG}" \
  "${SCRIPT_DIR}"

docker tag "${IMAGE_TAG}" "${IMAGE_REPO}:latest"
echo "::notice::Tagged ${IMAGE_TAG} and ${IMAGE_REPO}:latest"
