#!/usr/bin/env bash
# Shared builder for nightly ROCm runtime images.
# Tarball SDK (Ubuntu / manylinux) or yum (RHEL 8 / RHEL 9) is chosen from --context:
#   install-rvs.sh present  -> yum (rhel8 / rhel9 from the directory name)
#   otherwise               -> TheRock SDK tarball
#
# Examples:
#   ./build-rocm-sdk-image.sh --context .github/docker/rvs-nightly-rocm-ubuntu22.04 --from-tarball amdrocm10-rvs-...-Linux.tar.gz
#   ./build-rocm-sdk-image.sh --context .github/docker/rvs-nightly-rocm-rhel8 --channel nightly
#   ./build-rocm-sdk-image.sh --context .github/docker/rvs-nightly-rocm-rhel9 --resolve-only
#
# Nightly listing default: https://nightly.repo.amd.com/rocm/core/tarball/
# Override with ROCM_SDK_NIGHTLY_BASE_URL / ROCM_SDK_NIGHTLY_INDEX_URL.

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT=""
IMAGE_REPO="${RVS_NIGHTLY_DOCKER_IMAGE:-}"

ROCM_VERSION="${ROCM_VERSION:-}"
GPU_FAMILY="${GPU_FAMILY:-multiarch}"
CHANNEL="nightly"
IMAGE_TAG=""
ROCM_SDK_BASE_URL=""
FROM_TARBALL=""
FALLBACK_LATEST_SDK="${RVS_DOCKER_SDK_FALLBACK_LATEST:-true}"
GPU_TARGET="${GPU_TARGET:-multiarch}"
RESOLVE_ONLY=false
ROCM_REPO_BASEURL="${ROCM_REPO_BASEURL:-}"
RVS_REPO_BASEURL="${RVS_REPO_BASEURL:-}"
ROCM_GPG_KEY="${ROCM_GPG_KEY:-}"
ROCM_PACKAGE="${ROCM_PACKAGE:-}"
RVS_PACKAGE="${RVS_PACKAGE:-}"
ROCM_MAJOR=""
ROCM_SNAPSHOT=""
RVS_REPO_OVERRIDE=false
RHEL_DIST=""
ROCM_NIGHTLY_INDEX=""
RVS_NIGHTLY_REPO_DEFAULT=""
RVS_STABLE_REPO_DEFAULT=""

# Override with ROCM_SDK_NIGHTLY_BASE_URL (and optional ROCM_SDK_NIGHTLY_INDEX_URL).
if [ -n "${ROCM_SDK_NIGHTLY_BASE_URL:-}" ]; then
  NIGHTLY_BASE="${ROCM_SDK_NIGHTLY_BASE_URL%/}"
  NIGHTLY_INDEX="${ROCM_SDK_NIGHTLY_INDEX_URL:-${NIGHTLY_BASE}/}"
elif [ -n "${ROCM_SDK_NIGHTLY_INDEX_URL:-}" ]; then
  NIGHTLY_INDEX="${ROCM_SDK_NIGHTLY_INDEX_URL}"
  NIGHTLY_BASE="${NIGHTLY_INDEX%/}"
else
  NIGHTLY_INDEX="https://nightly.repo.amd.com/rocm/core/tarball/"
  NIGHTLY_BASE="https://nightly.repo.amd.com/rocm/core/tarball"
fi
RELEASE_LIST="${ROCM_SDK_RELEASE_URL:-https://repo.amd.com/rocm/tarball/}"
RELEASE_BASE="${ROCM_SDK_RELEASE_BASE_URL:-https://repo.amd.com/rocm/tarball}"

usage() {
  sed -n '2,12p' "$0"
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

fallback_latest_sdk_enabled() {
  case "${FALLBACK_LATEST_SDK}" in
    true|1|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

fetch_latest_nightly_sdk_for_line() {
  local listing="$1"
  local major="$2"
  local minor="$3"
  local prefix="${major}.${minor}.0a"
  grep -oE "therock-dist-linux-${GPU_FAMILY}-${prefix}[0-9]+" "$listing" \
    | sed "s|^therock-dist-linux-${GPU_FAMILY}-||" | sort -V | tail -1
}

fetch_latest_nightly_sdk_for_major() {
  local listing="$1"
  local major="$2"
  grep -oE "therock-dist-linux-${GPU_FAMILY}-${major}\.[0-9]+\.[0-9]+a[0-9]+" "$listing" \
    | sed "s|^therock-dist-linux-${GPU_FAMILY}-||" | sort -V | tail -1
}

fetch_latest_nightly_sdk_any() {
  local listing="$1"
  grep -oE "therock-dist-linux-${GPU_FAMILY}-[0-9]+\.[0-9]+\.[0-9]+a[0-9]+" "$listing" \
    | sed "s|^therock-dist-linux-${GPU_FAMILY}-||" | sort -V | tail -1
}

# Same major.minor, else same major (10.0 missing -> newest 10.x), else newest nightly on the listing.
pick_fallback_nightly_sdk() {
  local listing="$1" major="$2" minor="$3" exact="$4" ver
  ver="$(fetch_latest_nightly_sdk_for_line "$listing" "$major" "$minor")"
  if [ -n "$ver" ]; then
    echo "::warning::SDK ${exact} missing on ${NIGHTLY_INDEX}; using latest ${major}.${minor}.0a* ${ver}" >&2
    printf '%s\n' "$ver"
    return 0
  fi
  ver="$(fetch_latest_nightly_sdk_for_major "$listing" "$major")"
  if [ -n "$ver" ]; then
    echo "::warning::No ${major}.${minor}.0a* SDK for ${exact}; using latest ROCm ${major}.x ${ver}" >&2
    printf '%s\n' "$ver"
    return 0
  fi
  ver="$(fetch_latest_nightly_sdk_any "$listing")"
  if [ -n "$ver" ]; then
    echo "::warning::No ROCm ${major}.x SDK for ${exact}; using latest nightly ${ver}" >&2
    printf '%s\n' "$ver"
    return 0
  fi
  return 1
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
    if grep -qF "$sdk_file" "$listing_tmp"; then
      ROCM_VERSION="$exact"
      ROCM_SDK_BASE_URL="$NIGHTLY_BASE"
    elif fallback_latest_sdk_enabled; then
      if ! ROCM_VERSION="$(pick_fallback_nightly_sdk "$listing_tmp" "$major" "$minor" "$exact")"; then
        rm -f "$listing_tmp"
        echo "::error::No ${GPU_FAMILY} SDK ${exact} for tar ${base} (missing ${sdk_file} on ${NIGHTLY_INDEX}) and no nightly build on that listing" >&2
        exit 1
      fi
      ROCM_SDK_BASE_URL="$NIGHTLY_BASE"
    else
      rm -f "$listing_tmp"
      echo "::error::No ${GPU_FAMILY} SDK ${exact} for tar ${base} (missing ${sdk_file} on ${NIGHTLY_INDEX}). Pass --fallback-latest-sdk or set RVS_DOCKER_SDK_FALLBACK_LATEST=true to use the latest nightly build." >&2
      exit 1
    fi
    rm -f "$listing_tmp"
  fi

  if [ -z "$ROCM_VERSION" ]; then
    echo "::error::No ${CHANNEL} SDK found for ROCm ${major}.${minor} (${GPU_FAMILY}) from tar ${base}" >&2
    exit 1
  fi
  echo "Resolved ROCm ${ROCM_VERSION} from tar tarball ${base} (r$(printf '%02d%02d' "$major" "$minor").${build_date})"
}

yum_context() {
  [ -f "${CONTEXT}/install-rvs.sh" ]
}

init_rhel_dist() {
  case "$(basename "$CONTEXT")" in
    *rhel8*) RHEL_DIST=rhel8 ;;
    *rhel9*) RHEL_DIST=rhel9 ;;
    *)
      echo "::error::Yum image context ${CONTEXT} must be an *rhel8* or *rhel9* docker directory" >&2
      exit 1
      ;;
  esac
  local idx_var rvs_var gpg_var
  idx_var="RVS_NIGHTLY_$(printf '%s' "$RHEL_DIST" | tr '[:lower:]' '[:upper:]')_ROCM_REPO_INDEX"
  rvs_var="RVS_NIGHTLY_$(printf '%s' "$RHEL_DIST" | tr '[:lower:]' '[:upper:]')_RVS_REPO_BASEURL"
  gpg_var="RVS_NIGHTLY_$(printf '%s' "$RHEL_DIST" | tr '[:lower:]' '[:upper:]')_GPG_KEY"
  if [ -z "$ROCM_NIGHTLY_INDEX" ]; then
    ROCM_NIGHTLY_INDEX="${!idx_var:-https://nightly.repo.amd.com/rocm/core/packages/${RHEL_DIST}/}"
  fi
  if [ -z "$RVS_REPO_BASEURL" ]; then
    RVS_REPO_BASEURL="${!rvs_var:-}"
  fi
  if [ -z "$ROCM_GPG_KEY" ]; then
    ROCM_GPG_KEY="${!gpg_var:-https://stable.repo.amd.com/rocm/gpg/packages.gpg}"
  fi
  RVS_NIGHTLY_REPO_DEFAULT="https://nightly.repo.amd.com/rocm/extras/rvs/packages/${RHEL_DIST}/x86_64"
  RVS_STABLE_REPO_DEFAULT="https://stable.repo.amd.com/rocm/extras/rvs/packages/${RHEL_DIST}/x86_64"
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

emit_github_yum() {
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
    echo "tarball_name=amdrocm${ROCM_MAJOR}-rvs-nightly-${RHEL_DIST}"
    echo "tarball_url=${ROCM_REPO_BASEURL}"
  } >> "$GITHUB_OUTPUT"
}

resolve_nightly_repos() {
  if [ "$CHANNEL" != "nightly" ]; then
    echo "::error::RHEL docker tests only support --channel nightly (got ${CHANNEL})" >&2
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

  echo "Resolved ${RHEL_DIST} nightly repos"
  echo "  ROCm snapshot : ${ROCM_SNAPSHOT:-n/a}"
  echo "  ROCm repo     : ${ROCM_REPO_BASEURL}"
  echo "  ROCm package  : ${ROCM_PACKAGE:-amdrocm*-${GPU_TARGET}}"
  echo "  ROCm major    : ${ROCM_MAJOR}"
  echo "  RVS repo      : ${RVS_REPO_BASEURL}"
  echo "  RVS package   : ${RVS_PACKAGE}"
  echo "  GPU target    : ${GPU_TARGET}"
  echo "  ROCm path     : ${ROCM_INSTALL_PATH:-/opt/rocm}"
  emit_github_yum
}

build_yum_image() {
  local build_args
  IMAGE_TAG="${IMAGE_TAG:-${IMAGE_REPO}:${ROCM_VERSION}}"
  echo "Building docker image ${IMAGE_TAG}"
  echo "  Context      : ${CONTEXT}"
  echo "  Base image   : $(awk '/^FROM / { print $2; exit }' "${CONTEXT}/Dockerfile")"
  echo "  ROCm version : ${ROCM_VERSION}"
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
  docker build -f "${CONTEXT}/Dockerfile" "${build_args[@]}" -t "${IMAGE_TAG}" "${CONTEXT}"
  docker tag "${IMAGE_TAG}" "${IMAGE_REPO}:latest"
  echo "::notice::Tagged ${IMAGE_TAG} and ${IMAGE_REPO}:latest"
}

build_tarball_image() {
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
  echo "  Context      : ${CONTEXT}"
  echo "  Base image   : $(awk '/^FROM / { print $2; exit }' "${CONTEXT}/Dockerfile")"
  ROCM_INSTALL_PATH="${ROCM_INSTALL_PATH:-/opt/rocm/install}"
  docker build \
    -f "${CONTEXT}/Dockerfile" \
    --build-arg "ROCM_VERSION=${ROCM_VERSION}" \
    --build-arg "GPU_FAMILY=${GPU_FAMILY}" \
    --build-arg "ROCM_SDK_BASE_URL=${ROCM_SDK_BASE_URL}" \
    --build-arg "ROCM_INSTALL_PATH=${ROCM_INSTALL_PATH}" \
    -t "${IMAGE_TAG}" \
    "${CONTEXT}"
  docker tag "${IMAGE_TAG}" "${IMAGE_REPO}:latest"
  echo "::notice::Tagged ${IMAGE_TAG} and ${IMAGE_REPO}:latest"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --context)      CONTEXT="$2"; shift 2 ;;
    --rocm-version) ROCM_VERSION="$2"; shift 2 ;;
    --from-tarball) FROM_TARBALL="$2"; shift 2 ;;
    --gpu-family)   GPU_FAMILY="$2"; GPU_TARGET="$2"; shift 2 ;;
    --gpu-target)   GPU_TARGET="$2"; GPU_FAMILY="$2"; shift 2 ;;
    --channel)      CHANNEL="$2"; shift 2 ;;
    --tag)          IMAGE_TAG="$2"; shift 2 ;;
    --rocm-repo)    ROCM_REPO_BASEURL="$2"; shift 2 ;;
    --rvs-repo)     RVS_REPO_BASEURL="$2"; RVS_REPO_OVERRIDE=true; shift 2 ;;
    --rocm-package) ROCM_PACKAGE="$2"; shift 2 ;;
    --rvs-package)  RVS_PACKAGE="$2"; shift 2 ;;
    --resolve-only) RESOLVE_ONLY=true; shift ;;
    --fallback-latest-sdk) FALLBACK_LATEST_SDK=true; shift ;;
    -h|--help)      usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [ -z "$CONTEXT" ]; then
  if [ -f "${SELF_DIR}/Dockerfile" ]; then
    CONTEXT="$SELF_DIR"
  else
    echo "::error::Pass --context <docker-dir> (directory with Dockerfile), or place this script next to the Dockerfile." >&2
    exit 1
  fi
fi
CONTEXT="$(cd "$CONTEXT" && pwd)"
if [ ! -f "${CONTEXT}/Dockerfile" ]; then
  echo "::error::No Dockerfile in context: ${CONTEXT}" >&2
  exit 1
fi

if [ -z "$IMAGE_REPO" ]; then
  IMAGE_REPO="$(basename "$CONTEXT"):latest"
fi
IMAGE_REPO="${IMAGE_REPO%%:*}"

if yum_context; then
  init_rhel_dist
  if [ -n "$FROM_TARBALL" ]; then
    echo "::notice::--from-tarball ${FROM_TARBALL} ignored; ${RHEL_DIST} image uses nightly dnf repos"
  fi
  resolve_nightly_repos
  if [ "$RESOLVE_ONLY" = true ]; then
    exit 0
  fi
  build_yum_image
else
  if [ "$RESOLVE_ONLY" = true ]; then
    echo "::error::--resolve-only is only supported for RHEL yum images" >&2
    exit 1
  fi
  build_tarball_image
fi
