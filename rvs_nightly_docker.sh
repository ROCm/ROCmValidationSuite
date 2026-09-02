#!/usr/bin/env bash
################################################################################
# RVS nightly tests inside a ROCm-matched docker container.
#
# Default (RVS_DOCKER_ON_TARGET=true): build the ROCm image on the GPU target
# (RVS_DOCKER_BUILD_ON_TARGET=true) or deliver via scp/registry, then docker run via SSH.
# Set RVS_DOCKER_ON_TARGET=false to run docker locally on the build host instead.
#
# Image delivery (RVS_DOCKER_TRANSFER_MODE):
#   auto            build-on-target if RVS_DOCKER_BUILD_ON_TARGET=true, else registry
#                   when RVS_DOCKER_REGISTRY is set, else scp (default path)
#   scp             docker save | (pigz|gzip) | scp | docker load
#   registry        docker push on build host, docker pull on target
#   build-on-target docker build on the GPU node (no cross-host image transfer)
#
# Set RVS_DOCKER_SKIP_IF_PRESENT=false to force re-transfer/re-build.
################################################################################

set -euo pipefail

DOCKER_IMAGE="${RVS_NIGHTLY_DOCKER_IMAGE:-rvs-nightly-rocm:latest}"
DOCKER_IMAGE_ARCHIVE="${RVS_DOCKER_IMAGE_ARCHIVE:-rvs-nightly-rocm-image.tar.gz}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_BUILD_DIR="${RVS_DOCKER_BUILD_DIR:-${REPO_ROOT}/.github/docker/rvs-nightly-rocm}"
if [[ "$DOCKER_BUILD_DIR" != /* ]]; then
  DOCKER_BUILD_DIR="${REPO_ROOT}/${DOCKER_BUILD_DIR#./}"
fi

usage() {
  cat <<'EOF'
Usage: rvs_nightly_docker.sh <command>

Image delivery (build host → target):
  ensure-image-on-target     Skip, registry pull, scp load, or build on target (preferred)
  transfer-image-to-target   Alias for ensure-image-on-target
  build-image-on-target      tar docker build context, scp to target, docker build
  verify-image-on-target     Confirm image exists on target

In-container steps (on target when RVS_DOCKER_ON_TARGET=true):
  pull-image                 Verify image on build host (local) before scp/registry push
  verify-rocm                rocminfo + amd-smi inside container
  install-rvs                Extract RVS tarball under /opt/rocm/extras-<N>
  run-level4                 rvs -r 4 inside container
  capture-versions           Write rvs_version and target_rocm_version to GITHUB_OUTPUT
  run-pipeline               verify-rocm → install-rvs → run-level4

Environment:
  RVS_DOCKER_TRANSFER_MODE   auto | scp | registry | build-on-target (default: auto)
  RVS_DOCKER_BUILD_ON_TARGET true (default) prefers build-on-target in auto mode
  RVS_DOCKER_REGISTRY        e.g. ghcr.io/org/rvs-nightly-rocm (enables registry mode)
  RVS_DOCKER_REGISTRY_USER   optional registry login user
  RVS_DOCKER_REGISTRY_PASSWORD optional registry login password
  RVS_DOCKER_ROCM_VERSION    expected ROCm SDK version (for skip-if-present matching)
  RVS_DOCKER_SKIP_IF_PRESENT true (default) skips transfer when target already has image
  RVS_DOCKER_BUILD_DIR         docker build context (default: .github/docker/rvs-nightly-rocm)
  RVS_DOCKER_SDK_FALLBACK_LATEST  when true, use latest same-line SDK if exact date missing on CDN
EOF
}

docker_build_fallback_args() {
  case "${RVS_DOCKER_SDK_FALLBACK_LATEST:-false}" in
    true|1|yes|YES) printf '%s' ' --fallback-latest-sdk' ;;
    *) printf '%s' '' ;;
  esac
}

require_env() {
  local n="$1"
  if [ -z "${!n:-}" ]; then
    echo "::error::Required environment variable $n is not set" >&2
    exit 1
  fi
}

use_target_docker() {
  case "${RVS_DOCKER_ON_TARGET:-true}" in
    0|false|False|no|NO) return 1 ;;
  esac
  if [ -n "${TARGET_NODE:-}" ] && [ "$TARGET_NODE" != "localhost" ] && [ "$TARGET_NODE" != "127.0.0.1" ]; then
    return 0
  fi
  if [ -z "${TARGET_NODE:-}" ] || [ "$TARGET_NODE" = "localhost" ] || [ "$TARGET_NODE" = "127.0.0.1" ]; then
    if [ "${RVS_DOCKER_ON_TARGET:-true}" = "true" ] || [ "${RVS_DOCKER_ON_TARGET:-}" = "1" ]; then
      echo "::error::RVS_DOCKER_ON_TARGET is enabled but TARGET_NODE is not set (use secrets.RVS_TARGET_NODE)." >&2
      exit 1
    fi
  fi
  return 1
}

rvs_in_image() {
  case "${RVS_DOCKER_RVS_IN_IMAGE:-false}" in
    1|true|True|yes|YES) return 0 ;;
  esac
  return 1
}

container_rocm_path() {
  printf '%s\n' "${TARGET_ROCM_PATH:-/opt/rocm/install}"
}

require_ssh_config() {
  if [ -z "${SSH_CONFIG_FILE:-}" ]; then
    local ssh_env_file="${RUNNER_TEMP:-/tmp}/rvs_ssh_env.sh"
    if [ -f "$ssh_env_file" ]; then
      # shellcheck source=/dev/null
      source "$ssh_env_file"
    fi
  fi
  require_env SSH_CONFIG_FILE
  if [ ! -f "$SSH_CONFIG_FILE" ]; then
    echo "::error::SSH config missing at $SSH_CONFIG_FILE — run: ./rvs_nightly_test.sh setup-ssh" >&2
    exit 1
  fi
}

phase_start() {
  PHASE_LABEL="$1"
  PHASE_START_TS=$(date +%s)
  echo "::group::${PHASE_LABEL}"
  echo "[$(date -u +%FT%TZ)] START ${PHASE_LABEL}"
}

phase_end() {
  local elapsed=$(( $(date +%s) - PHASE_START_TS ))
  echo "[$(date -u +%FT%TZ)] END ${PHASE_LABEL} (${elapsed}s)"
  echo "::notice::${PHASE_LABEL} completed in ${elapsed}s"
  echo "::endgroup::"
}

compress_pipe() {
  if command -v pigz >/dev/null 2>&1; then
    echo "::notice::Compressing with pigz"
    pigz -1
  else
    echo "::notice::Compressing with gzip (install pigz for faster compression)"
    gzip -1
  fi
}

decompress_cmd() {
  if command -v pigz >/dev/null 2>&1; then
    pigz -dc
  else
    gzip -dc
  fi
}

rocm_version_from_tarball() {
  local base="${1##*/}"
  if [[ "$base" =~ -r([0-9]{2})([0-9]{2})\.([0-9]{8})-Linux\.tar\.gz$ ]]; then
    printf '%d.%d.0a%s\n' "$((10#${BASH_REMATCH[1]}))" "$((10#${BASH_REMATCH[2]}))" "${BASH_REMATCH[3]}"
  fi
}

expected_rocm_version() {
  if [ -n "${RVS_DOCKER_ROCM_VERSION:-}" ]; then
    printf '%s\n' "$RVS_DOCKER_ROCM_VERSION"
    return 0
  fi
  if [ -n "${TARBALL_NAME:-}" ]; then
    rocm_version_from_tarball "$TARBALL_NAME"
    return 0
  fi
  return 1
}

image_rocm_version_on_target() {
  require_ssh_config
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "docker image inspect '${DOCKER_IMAGE}' --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null" \
    | sed -n 's/^ROCM_VERSION=//p' | head -1
}

image_rocm_version_local() {
  docker image inspect "${DOCKER_IMAGE}" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
    | sed -n 's/^ROCM_VERSION=//p' | head -1
}

build_host_image_matches_expected() {
  local expected actual
  expected="$(expected_rocm_version 2>/dev/null || true)"
  if [ -z "$expected" ]; then
    return 0
  fi
  actual="$(image_rocm_version_local 2>/dev/null || true)"
  [ "$actual" = "$expected" ]
}

verify_gzip_archive() {
  local archive="$1"
  local label="${2:-archive}"
  if [ ! -s "$archive" ]; then
    echo "::error::${label} is empty: ${archive}" >&2
    exit 1
  fi
  if command -v pigz >/dev/null 2>&1; then
    pigz -t "$archive"
  else
    gzip -t "$archive"
  fi
  echo "::notice::${label} integrity OK ($(stat -c%s "$archive" 2>/dev/null || stat -f%z "$archive") bytes)"
}

verify_remote_file_size() {
  local local_path="$1"
  local remote_path="$2"
  local local_size remote_size
  local_size="$(stat -c%s "$local_path" 2>/dev/null || stat -f%z "$local_path")"
  remote_size="$(ssh -q -F "$SSH_CONFIG_FILE" rvs-target "stat -c%s '${remote_path}' 2>/dev/null || stat -f%z '${remote_path}'")"
  if [ "$local_size" != "$remote_size" ]; then
    echo "::error::Remote file size mismatch for ${remote_path}: local=${local_size} remote=${remote_size}" >&2
    exit 1
  fi
  echo "::notice::Remote file size matches local (${local_size} bytes)"
}

cmd_build_image_on_build_host() {
  if ! rvs_in_image; then
    require_env TARBALL_NAME
  fi
  check_docker_local
  local build_script="${DOCKER_BUILD_DIR}/build-rocm-image.sh"
  if [ ! -f "$build_script" ]; then
    echo "::error::Docker build script not found: ${build_script}" >&2
    exit 1
  fi
  chmod +x "$build_script"
  local fallback_args=()
  case "${RVS_DOCKER_SDK_FALLBACK_LATEST:-false}" in
    true|1|yes|YES) fallback_args=(--fallback-latest-sdk) ;;
  esac
  phase_start "docker build on build host"
  if [ -n "${TARBALL_NAME:-}" ]; then
    RVS_NIGHTLY_DOCKER_IMAGE="${DOCKER_IMAGE}" "$build_script" --from-tarball "${TARBALL_NAME}" "${fallback_args[@]}"
  else
    RVS_NIGHTLY_DOCKER_IMAGE="${DOCKER_IMAGE}" "$build_script" --channel nightly "${fallback_args[@]}"
  fi
  phase_end "docker build on build host"
  if ! build_host_image_matches_expected; then
    echo "::warning::Build host image ROCM_VERSION=$(image_rocm_version_local || echo unknown) may differ from expected $(expected_rocm_version 2>/dev/null || echo unknown) (SDK fallback in use?)"
  fi
}

ensure_build_host_image() {
  check_docker_local
  if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1 && build_host_image_matches_expected; then
    echo "::notice::Build host has ${DOCKER_IMAGE} with matching ROCM_VERSION"
    return 0
  fi
  if [ -z "${TARBALL_NAME:-}" ] && ! rvs_in_image; then
    echo "::error::Build host image is missing or stale and TARBALL_NAME is unset — cannot rebuild." >&2
    echo "Re-run with build_docker_image=true, build_on_target=true, or set TARBALL_NAME." >&2
    exit 1
  fi
  local expected actual
  expected="$(expected_rocm_version 2>/dev/null || true)"
  actual="$(image_rocm_version_local 2>/dev/null || true)"
  echo "::notice::Build host image ROCM_VERSION=${actual:-missing}; need ${expected:-unknown} — rebuilding on build host"
  cmd_build_image_on_build_host
}

resolve_transfer_mode() {
  case "${RVS_DOCKER_TRANSFER_MODE:-auto}" in
    scp|registry|build-on-target)
      printf '%s\n' "${RVS_DOCKER_TRANSFER_MODE}"
      ;;
    auto)
      case "${RVS_DOCKER_BUILD_ON_TARGET:-true}" in
        0|false|False|no|NO)
          if [ -n "${RVS_DOCKER_REGISTRY:-}" ]; then
            printf '%s\n' registry
          elif [ -n "${TARBALL_NAME:-}" ]; then
            # Prefer build-on-target over scp when the tarball is known — scp of
            # multi-GB images is slow and fragile; build-on-target sends only context.
            printf '%s\n' build-on-target
          else
            printf '%s\n' scp
          fi
          ;;
        *)
          printf '%s\n' build-on-target
          ;;
      esac
      ;;
    *)
      echo "::error::Invalid RVS_DOCKER_TRANSFER_MODE: ${RVS_DOCKER_TRANSFER_MODE}" >&2
      exit 1
      ;;
  esac
}

registry_image_ref() {
  local version="${1:-}"
  local base="${RVS_DOCKER_REGISTRY%/}"
  if [ -n "$version" ]; then
    printf '%s:%s\n' "$base" "$version"
  else
    printf '%s\n' "$base"
  fi
}

docker_registry_login() {
  local where="$1"
  if [ -z "${RVS_DOCKER_REGISTRY_USER:-}" ] || [ -z "${RVS_DOCKER_REGISTRY_PASSWORD:-}" ]; then
    echo "::notice::No registry credentials set — assuming public pull/push for ${where}"
    return 0
  fi
  local registry_host="${RVS_DOCKER_REGISTRY%%/*}"
  case "$where" in
    local)
      echo "${RVS_DOCKER_REGISTRY_PASSWORD}" | docker login -u "${RVS_DOCKER_REGISTRY_USER}" \
        --password-stdin "$registry_host"
      ;;
    target)
      ssh -q -F "$SSH_CONFIG_FILE" rvs-target bash -s <<REMOTE
echo '${RVS_DOCKER_REGISTRY_PASSWORD}' | docker login -u '${RVS_DOCKER_REGISTRY_USER}' --password-stdin '${registry_host}'
REMOTE
      ;;
    *)
      echo "::error::docker_registry_login: unknown location ${where}" >&2
      exit 1
      ;;
  esac
}

skip_if_present_enabled() {
  case "${RVS_DOCKER_SKIP_IF_PRESENT:-true}" in
    0|false|False|no|NO) return 1 ;;
  esac
  return 0
}

image_ready_on_target() {
  require_ssh_config
  check_docker_on_target
  if ! ssh -q -F "$SSH_CONFIG_FILE" rvs-target "docker image inspect '${DOCKER_IMAGE}' >/dev/null 2>&1"; then
    return 1
  fi
  local expected actual
  expected="$(expected_rocm_version 2>/dev/null || true)"
  if [ -z "$expected" ]; then
    return 0
  fi
  actual="$(image_rocm_version_on_target || true)"
  if [ "$actual" = "$expected" ]; then
    echo "::notice::Target already has ${DOCKER_IMAGE} (ROCM_VERSION=${expected}) — skipping delivery"
    return 0
  fi
  echo "Target image ROCM_VERSION=${actual:-unknown}; need ${expected} — delivery required"
  return 1
}

target_rvs_install_dir() {
  require_env REMOTE_WORK_DIR
  require_env ROCM_MAJOR
  printf '%s/extras-%s' "${REMOTE_WORK_DIR%/}" "$ROCM_MAJOR"
}

docker_gpu_opts() {
  echo --ipc=host --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined
}

check_docker_local() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  echo "::error::Cannot access Docker on the build host (/var/run/docker.sock). Add the runner user to the 'docker' group and restart the runner." >&2
  exit 1
}

check_docker_on_target() {
  require_ssh_config
  if ssh -q -F "$SSH_CONFIG_FILE" rvs-target 'docker info >/dev/null 2>&1'; then
    return 0
  fi
  echo "::error::Cannot access Docker on the target node. Ensure the SSH user is in the 'docker' group on the target." >&2
  exit 1
}

docker_run_local() {
  check_docker_local
  local -a install_mount=()
  local rocm_path
  rocm_path="$(container_rocm_path)"
  if ! rvs_in_image && [ -n "${INSTALL_DIR:-}" ] && [ -n "${REMOTE_WORK_DIR:-}" ] && [ -n "${ROCM_MAJOR:-}" ]; then
    local host_install
    host_install="$(target_rvs_install_dir)"
    mkdir -p "$host_install"
    install_mount=(-v "${host_install}:${INSTALL_DIR}")
  fi
  if ! rvs_in_image && [ -n "${TARBALL_NAME:-}" ] && [ -f "${REPO_ROOT}/pkg/${TARBALL_NAME}" ]; then
    install_mount+=(-v "${REPO_ROOT}/pkg/${TARBALL_NAME}:/pkg/${TARBALL_NAME}:ro")
  fi
  # shellcheck disable=SC2046
  docker run --rm \
    $(docker_gpu_opts) \
    "${install_mount[@]}" \
    -v "${REPO_ROOT}/reports:/reports" \
    -w /workspace \
    -e "ROCM_PATH=${rocm_path}" \
    -e "TARGET_ROCM_PATH=${rocm_path}" \
    "$DOCKER_IMAGE" \
    bash -lc "$1"
}

target_docker_run() {
  require_ssh_config
  require_env REMOTE_WORK_DIR
  check_docker_on_target
  local inner_cmd="$1"
  local gpu_opts install_vol pkg_path pkg_mount remote_dirs escaped
  gpu_opts=$(docker_gpu_opts)
  install_vol=""
  pkg_path=""
  pkg_mount=""
  remote_dirs="'${REMOTE_WORK_DIR}/reports' '${REMOTE_WORK_DIR}/workspace' '${REMOTE_WORK_DIR}/pkg'"
  local rocm_path
  rocm_path="$(container_rocm_path)"
  if ! rvs_in_image && [ -n "${INSTALL_DIR:-}" ] && [ -n "${ROCM_MAJOR:-}" ]; then
    install_vol="-v ${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}:${INSTALL_DIR}"
    # Pre-create so docker does not make a root-owned extras dir.
    remote_dirs="${remote_dirs} '${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}'"
  fi
  if ! rvs_in_image && [ -n "${TARBALL_NAME:-}" ]; then
    pkg_path="${REMOTE_WORK_DIR}/pkg/${TARBALL_NAME}"
    pkg_mount="-v ${pkg_path}:/pkg/${TARBALL_NAME}:ro"
  fi
  escaped=$(printf '%q' "$inner_cmd")
  # Only bind-mount the tarball when the file already exists on the target.
  # If the path is missing, docker would create a root-owned *directory* there,
  # and a later scp to that path fails with Permission denied.
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target bash -s <<REMOTE
set -euo pipefail
mkdir -p ${remote_dirs}
pkg_vol=""
if [ -n '${pkg_path}' ] && [ -f '${pkg_path}' ]; then
  pkg_vol='${pkg_mount}'
fi
docker run --rm ${gpu_opts} \
  ${install_vol} \
  \${pkg_vol} \
  -v '${REMOTE_WORK_DIR}/reports:/reports' \
  -v '${REMOTE_WORK_DIR}/workspace:/workspace' \
  -w /workspace \
  -e ROCM_PATH='${rocm_path}' \
  -e TARGET_ROCM_PATH='${rocm_path}' \
  '${DOCKER_IMAGE}' \
  bash -lc ${escaped}
REMOTE
}

docker_run() {
  if use_target_docker; then
    target_docker_run "$1"
  else
    docker_run_local "$1"
  fi
}

cmd_pull_image() {
  ensure_build_host_image
}

cmd_build_image_on_target() {
  require_ssh_config
  require_env REMOTE_WORK_DIR
  if ! rvs_in_image; then
    require_env TARBALL_NAME
  fi
  check_docker_on_target

  local remote_build_dir="${REMOTE_WORK_DIR}/docker-build"
  local archive_name="docker-build-context.tar.gz"
  local local_archive="${REPO_ROOT}/pkg/${archive_name}"
  local remote_archive="${remote_build_dir}/${archive_name}"

  mkdir -p "${REPO_ROOT}/pkg"

  phase_start "Sync docker build context to target"
  tar -C "${DOCKER_BUILD_DIR}" -czf "${local_archive}" .
  ls -lh "${local_archive}"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "mkdir -p '${remote_build_dir}'"
  scp -q -F "$SSH_CONFIG_FILE" "${local_archive}" \
    "rvs-target:${remote_archive}"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "tar -xzf '${remote_archive}' -C '${remote_build_dir}' && rm -f '${remote_archive}'"
  phase_end "Sync docker build context to target"

  phase_start "docker build on GPU target"
  local fallback_args build_args=""
  fallback_args="$(docker_build_fallback_args)"
  if [ -n "${TARBALL_NAME:-}" ]; then
    build_args="--from-tarball '${TARBALL_NAME}'${fallback_args}"
  else
    build_args="--channel nightly${fallback_args}"
  fi
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target bash -s <<REMOTE
set -euo pipefail
chmod +x '${remote_build_dir}/build-rocm-image.sh'
RVS_NIGHTLY_DOCKER_IMAGE='${DOCKER_IMAGE}' \
ROCM_REPO_BASEURL='${ROCM_REPO_BASEURL:-}' \
RVS_REPO_BASEURL='${RVS_REPO_BASEURL:-}' \
ROCM_PACKAGE='${ROCM_PACKAGE:-}' \
RVS_PACKAGE='${RVS_PACKAGE:-}' \
GPU_TARGET='${GPU_TARGET:-gfx942}' \
ROCM_VERSION='${RVS_DOCKER_ROCM_VERSION:-}' \
'${remote_build_dir}/build-rocm-image.sh' ${build_args}
REMOTE
  phase_end "docker build on GPU target"
  cmd_verify_image_on_target
}

cmd_transfer_via_registry() {
  require_ssh_config
  require_env REMOTE_WORK_DIR
  require_env RVS_DOCKER_REGISTRY

  local version ref
  version="$(expected_rocm_version 2>/dev/null || true)"
  ref="$(registry_image_ref "$version")"

  ensure_build_host_image
  docker_registry_login local

  phase_start "docker tag + push (${ref})"
  docker tag "${DOCKER_IMAGE}" "${ref}"
  docker push "${ref}"
  if [ "$ref" != "${DOCKER_IMAGE}" ]; then
    docker tag "${DOCKER_IMAGE}" "rvs-nightly-rocm:latest"
  fi
  phase_end "docker tag + push (${ref})"

  phase_start "docker pull on target (${ref})"
  docker_registry_login target
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "docker pull '${ref}'"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "docker tag '${ref}' '${DOCKER_IMAGE}'"
  phase_end "docker pull on target (${ref})"
  cmd_verify_image_on_target
}

cmd_transfer_via_scp() {
  require_ssh_config
  require_env REMOTE_WORK_DIR
  ensure_build_host_image

  local archive="${REPO_ROOT}/pkg/${DOCKER_IMAGE_ARCHIVE}"
  local remote_archive="${REMOTE_WORK_DIR}/docker/${DOCKER_IMAGE_ARCHIVE}"
  mkdir -p "${REPO_ROOT}/pkg"

  phase_start "docker save + compress (${DOCKER_IMAGE})"
  rm -f "${archive}"
  docker save "${DOCKER_IMAGE}" | compress_pipe > "${archive}"
  ls -lh "${archive}"
  verify_gzip_archive "${archive}" "local image archive"
  phase_end "docker save + compress (${DOCKER_IMAGE})"

  phase_start "scp image archive to target"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "mkdir -p '${REMOTE_WORK_DIR}/docker'"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "rm -f '${remote_archive}' '${remote_archive}.partial'"
  scp -q -F "$SSH_CONFIG_FILE" "${archive}" \
    "rvs-target:${remote_archive}.partial"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "mv -f '${remote_archive}.partial' '${remote_archive}'"
  verify_remote_file_size "${archive}" "${remote_archive}"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target bash -s <<REMOTE
set -euo pipefail
if command -v pigz >/dev/null 2>&1; then
  pigz -t '${remote_archive}'
else
  gzip -t '${remote_archive}'
fi
REMOTE
  phase_end "scp image archive to target"

  phase_start "docker load on target"
  local decompress
  decompress="$(decompress_cmd)"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "${decompress} '${remote_archive}' | docker load"
  phase_end "docker load on target"
  cmd_verify_image_on_target
}

cmd_download_rvs_tarball() {
  require_env TARBALL_URL
  require_env TARBALL_NAME
  require_ssh_config
  require_env REMOTE_WORK_DIR

  local local_pkg="${REPO_ROOT}/pkg/${TARBALL_NAME}"
  mkdir -p "${REPO_ROOT}/pkg"

  phase_start "Download RVS tarball on orchestrator"
  echo "  ${TARBALL_URL}"
  curl -fL --max-time 600 -o "${local_pkg}" "${TARBALL_URL}"
  file "${local_pkg}" || true
  ls -lh "${local_pkg}"
  phase_end "Download RVS tarball on orchestrator"

  phase_start "scp RVS tarball to target"
  echo "  local_pkg=${local_pkg}"
  echo "  REMOTE_WORK_DIR=${REMOTE_WORK_DIR}"
  echo "  remote_dest=${REMOTE_WORK_DIR}/pkg/${TARBALL_NAME}"
  # Clear a leftover root-owned dir/file from a prior docker bind-mount miss,
  # then stage the tarball as a normal user-owned file.
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target bash -s <<REMOTE
set -euo pipefail
mkdir -p '${REMOTE_WORK_DIR}/pkg'
dest='${REMOTE_WORK_DIR}/pkg/${TARBALL_NAME}'
if [ -e "\$dest" ] && [ ! -f "\$dest" ]; then
  rm -rf "\$dest" 2>/dev/null || sudo -n rm -rf "\$dest"
fi
REMOTE
  scp -q -F "$SSH_CONFIG_FILE" "${local_pkg}" \
    "rvs-target:${REMOTE_WORK_DIR}/pkg/${TARBALL_NAME}"
  phase_end "scp RVS tarball to target"
}

cmd_ensure_image_on_target() {
  require_ssh_config
  require_env REMOTE_WORK_DIR

  local mode requested
  requested="${RVS_DOCKER_TRANSFER_MODE:-auto}"
  mode="$(resolve_transfer_mode)"
  if [ "$requested" = "auto" ] && [ "$mode" = "build-on-target" ] && \
     case "${RVS_DOCKER_BUILD_ON_TARGET:-true}" in 0|false|False|no|NO) true ;; *) false ;; esac; then
    echo "::notice::auto mode: using build-on-target (TARBALL_NAME set; avoids fragile multi-GB scp). Set RVS_DOCKER_TRANSFER_MODE=scp to force scp."
  fi
  echo "Image delivery mode: ${mode}"

  if skip_if_present_enabled && image_ready_on_target; then
    return 0
  fi

  case "$mode" in
    build-on-target)
      if [ -z "${TARBALL_NAME:-}" ] && ! rvs_in_image; then
        echo "::error::build-on-target requires TARBALL_NAME" >&2
        exit 1
      fi
      cmd_build_image_on_target
      ;;
    registry)
      cmd_transfer_via_registry
      ;;
    scp)
      cmd_transfer_via_scp
      ;;
  esac
}

cmd_verify_image_on_target() {
  require_ssh_config
  check_docker_on_target
  if ! ssh -q -F "$SSH_CONFIG_FILE" rvs-target "docker image inspect '${DOCKER_IMAGE}' >/dev/null 2>&1"; then
    echo "::error::Image ${DOCKER_IMAGE} not found on target after delivery." >&2
    exit 1
  fi
  local expected actual
  expected="$(expected_rocm_version 2>/dev/null || true)"
  actual="$(image_rocm_version_on_target || true)"
  if [ -n "$expected" ] && [ -n "$actual" ] && [ "$actual" != "$expected" ]; then
    echo "::warning::Target image ROCM_VERSION=${actual} (expected ${expected}) — SDK fallback or stale image may apply"
  fi
  echo "::notice::Image ${DOCKER_IMAGE} present on target node (ROCM_VERSION=${actual:-unknown})."
}

cmd_verify_rocm() {
  docker_run '
    source /etc/profile.d/rocm-env.sh
    echo "=== rocminfo ==="
    rocminfo
    echo "=== amd-smi version ==="
    amd-smi version 2>/dev/null || amdsmi version 2>/dev/null || echo "amd-smi not present"
  '
}

cmd_install_rvs() {
  require_env ROCM_MAJOR
  require_env INSTALL_DIR
  require_env RVS_BIN
  require_env REMOTE_WORK_DIR
  if rvs_in_image; then
    docker_run "
      source /etc/profile.d/rocm-env.sh
      set -euo pipefail
      RVS_BIN=${RVS_BIN}
      if [ ! -x \"\${RVS_BIN}\" ]; then
        RVS_BIN=\$(command -v rvs || true)
      fi
      if [ -z \"\${RVS_BIN}\" ] || [ ! -x \"\${RVS_BIN}\" ]; then
        echo \"::error::rvs binary missing in image (expected ${RVS_BIN})\" >&2
        exit 1
      fi
      echo \"::notice::RVS already installed in image at \${RVS_BIN}\"
      \"\${RVS_BIN}\" --version || true
    "
    return 0
  fi
  require_env TARBALL_NAME
  require_env TARBALL_URL

  if use_target_docker; then
    cmd_download_rvs_tarball
  else
    local local_pkg="${REPO_ROOT}/pkg/${TARBALL_NAME}"
    mkdir -p "${REPO_ROOT}/pkg"
    if [ ! -f "${local_pkg}" ]; then
      phase_start "Download RVS tarball locally"
      curl -fL --max-time 600 -o "${local_pkg}" "${TARBALL_URL}"
      phase_end "Download RVS tarball locally"
    fi
  fi

  docker_run "
    source /etc/profile.d/rocm-env.sh
    set -euo pipefail
    PKG=/pkg/${TARBALL_NAME}
    INSTALL_DIR=${INSTALL_DIR}
    RVS_BIN=${RVS_BIN}
    if [ ! -f \"\${PKG}\" ]; then
      echo \"::error::RVS tarball not mounted at \${PKG} — run download-rvs-tarball first\" >&2
      exit 1
    fi
    echo \"Installing RVS from pre-staged tarball \${PKG}...\"
    mkdir -p \"\${INSTALL_DIR}\"
    tar -xzf \"\${PKG}\" -C \"\${INSTALL_DIR}\"
    export LD_LIBRARY_PATH=\"\${INSTALL_DIR}/lib:\${LD_LIBRARY_PATH}\"
    if [ ! -x \"\${RVS_BIN}\" ]; then
      echo \"::error::rvs binary missing at \${RVS_BIN}\" >&2
      exit 1
    fi
    echo \"::notice::Installed RVS at \${RVS_BIN}\"
    ldd \"\${RVS_BIN}\" | grep 'not found' && exit 1 || true
    \"\${RVS_BIN}\" --version
  "
}

cmd_run_level4() {
  require_env INSTALL_DIR
  require_env RVS_BIN
  require_env REMOTE_WORK_DIR
  require_env ROCM_MAJOR

  if use_target_docker && ! rvs_in_image; then
    require_ssh_config
    if ! ssh -q -F "$SSH_CONFIG_FILE" rvs-target "test -x '${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}/bin/rvs'"; then
      echo "::error::RVS not installed on target at ${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}/bin/rvs — run install-rvs first." >&2
      exit 1
    fi
  elif ! rvs_in_image; then
    local host_install
    host_install="$(target_rvs_install_dir)"
    if [ ! -x "${host_install}/bin/rvs" ]; then
      echo "::error::RVS not installed at ${host_install}/bin/rvs — run install-rvs first." >&2
      exit 1
    fi
  fi

  mkdir -p "${REPO_ROOT}/reports"
  local start end rc
  start="$(date -u +%FT%TZ)"

  set +e
  docker_run "
    source /etc/profile.d/rocm-env.sh
    set -euo pipefail
    RVS_BIN=${RVS_BIN}
    if [ ! -x \"\${RVS_BIN}\" ]; then
      RVS_BIN=\$(command -v rvs)
    fi
    export LD_LIBRARY_PATH=\"${INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}\"
    mkdir -p /reports
    \"\${RVS_BIN}\" -r 4 2>&1 | tee /reports/rvs_level_4.log
    exit \${PIPESTATUS[0]}
  "
  rc=$?
  set -e
  end="$(date -u +%FT%TZ)"

  if use_target_docker; then
    mkdir -p "${REPO_ROOT}/reports"
    scp -q -F "$SSH_CONFIG_FILE" "rvs-target:${REMOTE_WORK_DIR}/reports/rvs_level_4.log" \
      "${REPO_ROOT}/reports/" 2>/dev/null || echo "::warning::Could not copy rvs_level_4.log from target."
  fi

  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "rc=$rc" >> "$GITHUB_OUTPUT"
    echo "start=$start" >> "$GITHUB_OUTPUT"
    echo "end=$end" >> "$GITHUB_OUTPUT"
  fi
  return 0
}

cmd_capture_versions() {
  require_env RVS_BIN
  require_env INSTALL_DIR
  require_env REMOTE_WORK_DIR
  require_env ROCM_MAJOR

  local rvs_version target_rocm_version
  rvs_version=$(docker_run "
    source /etc/profile.d/rocm-env.sh
    export LD_LIBRARY_PATH=\"${INSTALL_DIR}/lib:\${LD_LIBRARY_PATH}\"
    ${RVS_BIN} --version 2>/dev/null | head -1
  " | tail -1)
  target_rocm_version=$(docker_run "
    source /etc/profile.d/rocm-env.sh
    cat \"\${TARGET_ROCM_PATH}/.info/version\" 2>/dev/null \
      || cat \"\${TARGET_ROCM_PATH}/share/doc/rocm-core/version\" 2>/dev/null \
      || cat /opt/rocm/install/.info/version 2>/dev/null \
      || cat /opt/rocm/.info/version 2>/dev/null \
      || echo unknown
  " | tail -1)

  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "rvs_version=${rvs_version:-unknown}" >> "$GITHUB_OUTPUT"
    echo "target_rocm_version=${target_rocm_version:-unknown}" >> "$GITHUB_OUTPUT"
  fi
}

cmd_run_pipeline() {
  require_env REMOTE_WORK_DIR
  cmd_verify_rocm
  cmd_install_rvs
  cmd_run_level4
}

main() {
  [ $# -ge 1 ] || { usage; exit 1; }
  case "$1" in
    pull-image)               cmd_pull_image ;;
    ensure-image-on-target)   cmd_ensure_image_on_target ;;
    transfer-image-to-target) cmd_ensure_image_on_target ;;
    build-image-on-target)    cmd_build_image_on_target ;;
    verify-image-on-target)   cmd_verify_image_on_target ;;
    download-rvs-tarball)     cmd_download_rvs_tarball ;;
    verify-rocm)              cmd_verify_rocm ;;
    install-rvs)              cmd_install_rvs ;;
    run-level4)               cmd_run_level4 ;;
    capture-versions)         cmd_capture_versions ;;
    run-pipeline)             cmd_run_pipeline ;;
    -h|--help)                usage ;;
    *) echo "Unknown command: $1" >&2; usage; exit 1 ;;
  esac
}

main "$@"
