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
DOCKER_BUILD_DIR="${REPO_ROOT}/.github/docker/rvs-nightly-rocm"

usage() {
  cat <<'EOF'
Usage: rvs_nightly_docker.sh <command>

Image delivery (build host → target):
  ensure-image-on-target     Skip, registry pull, scp load, or build on target (preferred)
  transfer-image-to-target   Alias for ensure-image-on-target
<<<<<<< Updated upstream
  build-image-on-target      tar docker build context, scp to target, docker build
=======
  build-image-on-target      git fetch on GPU target + docker build (no scp of build context)
>>>>>>> Stashed changes
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
EOF
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
  if [ -n "${INSTALL_DIR:-}" ] && [ -n "${REMOTE_WORK_DIR:-}" ] && [ -n "${ROCM_MAJOR:-}" ]; then
    local host_install
    host_install="$(target_rvs_install_dir)"
    mkdir -p "$host_install"
    install_mount=(-v "${host_install}:${INSTALL_DIR}")
  fi
  if [ -n "${TARBALL_NAME:-}" ] && [ -f "${REPO_ROOT}/pkg/${TARBALL_NAME}" ]; then
    install_mount+=(-v "${REPO_ROOT}/pkg/${TARBALL_NAME}:/pkg/${TARBALL_NAME}:ro")
  fi
  # shellcheck disable=SC2046
  docker run --rm \
    $(docker_gpu_opts) \
    "${install_mount[@]}" \
    -v "${REPO_ROOT}/reports:/reports" \
    -w /workspace \
    -e ROCM_PATH=/opt/rocm/install \
    -e TARGET_ROCM_PATH=/opt/rocm/install \
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
  if [ -n "${INSTALL_DIR:-}" ] && [ -n "${ROCM_MAJOR:-}" ]; then
    install_vol="-v ${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}:${INSTALL_DIR}"
    # Pre-create so docker does not make a root-owned extras dir.
    remote_dirs="${remote_dirs} '${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}'"
  fi
  if [ -n "${TARBALL_NAME:-}" ]; then
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
  -e ROCM_PATH=/opt/rocm/install \
  -e TARGET_ROCM_PATH=/opt/rocm/install \
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
  check_docker_local
  if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "::notice::Image ${DOCKER_IMAGE} present on build host."
    return 0
  fi
  echo "::error::Docker image ${DOCKER_IMAGE} not found on build host. Run:" >&2
  echo "  ./.github/docker/rvs-nightly-rocm/setup-on-runner.sh --from-tarball <tarball-name>" >&2
  echo "Or re-run the workflow with build_docker_image=true (or build-on-target mode)." >&2
  exit 1
}

cmd_build_image_on_target() {
  require_ssh_config
  require_env REMOTE_WORK_DIR
  require_env TARBALL_NAME
  check_docker_on_target

  local remote_build_dir="${REMOTE_WORK_DIR}/docker-build"
<<<<<<< Updated upstream
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
=======
  local clone_dir="${remote_build_dir}/repo"
  local build_script=".github/docker/rvs-nightly-rocm/build-rocm-image.sh"
>>>>>>> Stashed changes

  if [ -n "${GITHUB_REPOSITORY:-}" ] && [ -n "${GITHUB_SHA:-}" ]; then
    local host repo_url sha
    host="${GITHUB_SERVER_URL:-https://github.com}"
    host="${host#https://}"
    host="${host#http://}"
    if [ -n "${GITHUB_TOKEN:-}" ]; then
      repo_url="https://x-access-token:${GITHUB_TOKEN}@${host}/${GITHUB_REPOSITORY}.git"
    else
      repo_url="https://${host}/${GITHUB_REPOSITORY}.git"
    fi
    sha="${GITHUB_SHA}"

    phase_start "Checkout repo and docker build on GPU target"
    ssh -q -F "$SSH_CONFIG_FILE" rvs-target bash -s <<REMOTE
set -euo pipefail
if ! command -v git >/dev/null 2>&1; then
  echo "::error::git is required on the target node for build-on-target" >&2
  exit 1
fi
rm -rf '${clone_dir}'
mkdir -p '${clone_dir}'
git -C '${clone_dir}' init -q
git -C '${clone_dir}' remote add origin '${repo_url}'
git -C '${clone_dir}' fetch --depth 1 origin '${sha}'
git -C '${clone_dir}' checkout -q FETCH_HEAD
chmod +x '${clone_dir}/${build_script}'
'${clone_dir}/${build_script}' --from-tarball '${TARBALL_NAME}'
REMOTE
    phase_end "Checkout repo and docker build on GPU target"
  else
    phase_start "Sync docker build context to target"
    ssh -q -F "$SSH_CONFIG_FILE" rvs-target "mkdir -p '${remote_build_dir}'"
    tar -C "${DOCKER_BUILD_DIR}" -cf - . \
      | ssh -q -F "$SSH_CONFIG_FILE" rvs-target "tar -C '${remote_build_dir}' -xf -"
    phase_end "Sync docker build context to target"

    phase_start "docker build on GPU target"
    ssh -q -F "$SSH_CONFIG_FILE" rvs-target bash -s <<REMOTE
set -euo pipefail
chmod +x '${remote_build_dir}/build-rocm-image.sh'
'${remote_build_dir}/build-rocm-image.sh' --from-tarball '${TARBALL_NAME}'
REMOTE
    phase_end "docker build on GPU target"
  fi

  cmd_verify_image_on_target
}

cmd_transfer_via_registry() {
  require_ssh_config
  require_env REMOTE_WORK_DIR
  require_env RVS_DOCKER_REGISTRY

  local version ref
  version="$(expected_rocm_version 2>/dev/null || true)"
  ref="$(registry_image_ref "$version")"

  cmd_pull_image
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
  cmd_pull_image

  local archive="${REPO_ROOT}/pkg/${DOCKER_IMAGE_ARCHIVE}"
  mkdir -p "${REPO_ROOT}/pkg"

  phase_start "docker save + compress (${DOCKER_IMAGE})"
  docker save "${DOCKER_IMAGE}" | compress_pipe > "${archive}"
  ls -lh "${archive}"
  phase_end "docker save + compress (${DOCKER_IMAGE})"

  phase_start "scp image archive to target"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "mkdir -p '${REMOTE_WORK_DIR}/docker'"
  scp -q -F "$SSH_CONFIG_FILE" "${archive}" \
    "rvs-target:${REMOTE_WORK_DIR}/docker/${DOCKER_IMAGE_ARCHIVE}"
  phase_end "scp image archive to target"

  phase_start "docker load on target"
  local decompress
  decompress="$(decompress_cmd)"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "${decompress} '${REMOTE_WORK_DIR}/docker/${DOCKER_IMAGE_ARCHIVE}' | docker load"
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

  local mode
  mode="$(resolve_transfer_mode)"
  echo "Image delivery mode: ${mode}"

  if skip_if_present_enabled && image_ready_on_target; then
    return 0
  fi

  case "$mode" in
    build-on-target)
      if [ -z "${TARBALL_NAME:-}" ]; then
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
  if ssh -q -F "$SSH_CONFIG_FILE" rvs-target "docker image inspect '${DOCKER_IMAGE}' >/dev/null 2>&1"; then
    echo "::notice::Image ${DOCKER_IMAGE} present on target node."
    return 0
  fi
  echo "::error::Image ${DOCKER_IMAGE} not found on target after delivery." >&2
  exit 1
}

cmd_verify_rocm() {
  docker_run '
    source /etc/profile.d/rocm-env.sh
    echo "=== rocminfo ==="
    rocminfo
    echo "=== amd-smi version ==="
    amd-smi version
  '
}

cmd_install_rvs() {
  require_env TARBALL_NAME
  require_env TARBALL_URL
  require_env ROCM_MAJOR
  require_env INSTALL_DIR
  require_env RVS_BIN
  require_env REMOTE_WORK_DIR

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

  if use_target_docker; then
    require_ssh_config
    if ! ssh -q -F "$SSH_CONFIG_FILE" rvs-target "test -x '${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}/bin/rvs'"; then
      echo "::error::RVS not installed on target at ${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}/bin/rvs — run install-rvs first." >&2
      exit 1
    fi
  else
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
    export LD_LIBRARY_PATH=\"${INSTALL_DIR}/lib:\${LD_LIBRARY_PATH}\"
    mkdir -p /reports
    ${RVS_BIN} -r 4 2>&1 | tee /reports/rvs_level_4.log
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
    cat /opt/rocm/install/.info/version 2>/dev/null \
      || cat /opt/rocm/install/share/doc/rocm-core/version 2>/dev/null \
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
