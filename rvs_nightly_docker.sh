#!/usr/bin/env bash
################################################################################
# RVS nightly tests inside a ROCm-matched docker container.
#
# Default (RVS_DOCKER_ON_TARGET=true): build host saves the image,
# scp + docker load on the GPU target, then docker run on the target via SSH.
# Set RVS_DOCKER_ON_TARGET=false to run docker locally on the build host instead.
################################################################################

set -euo pipefail

DOCKER_IMAGE="${RVS_NIGHTLY_DOCKER_IMAGE:-rvs-nightly-rocm:latest}"
DOCKER_IMAGE_ARCHIVE="${RVS_DOCKER_IMAGE_ARCHIVE:-rvs-nightly-rocm-image.tar.gz}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: rvs_nightly_docker.sh <command>

Image transfer (build host → target via scp + docker load):
  transfer-image-to-target   docker save | gzip, scp, docker load on target
  verify-image-on-target     Confirm image exists on target after load

In-container steps (on target when RVS_DOCKER_ON_TARGET=true):
  pull-image                 Verify image on build host (local) before transfer
  verify-rocm                rocminfo + amd-smi inside container
  install-rvs                Extract RVS tarball under /opt/rocm/extras-<N>
  run-level4                 rvs -r 4 inside container
  capture-versions           Write rvs_version and target_rocm_version to GITHUB_OUTPUT
  run-pipeline               verify-rocm → install-rvs → run-level4
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
  local gpu_opts install_vol escaped
  gpu_opts=$(docker_gpu_opts)
  install_vol=""
  if [ -n "${INSTALL_DIR:-}" ] && [ -n "${ROCM_MAJOR:-}" ]; then
    install_vol="-v ${REMOTE_WORK_DIR}/extras-${ROCM_MAJOR}:${INSTALL_DIR}"
  fi
  escaped=$(printf '%q' "$inner_cmd")
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target bash -s <<REMOTE
set -euo pipefail
mkdir -p '${REMOTE_WORK_DIR}/reports' '${REMOTE_WORK_DIR}/workspace'
docker run --rm ${gpu_opts} \
  ${install_vol} \
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
  echo "Or re-run the workflow with build_docker_image=true." >&2
  exit 1
}

cmd_transfer_image_to_target() {
  require_ssh_config
  require_env REMOTE_WORK_DIR
  cmd_pull_image
  local archive="${REPO_ROOT}/pkg/${DOCKER_IMAGE_ARCHIVE}"
  mkdir -p "${REPO_ROOT}/pkg"
  echo "Saving ${DOCKER_IMAGE} to ${archive} ..."
  docker save "${DOCKER_IMAGE}" | gzip -1 > "${archive}"
  ls -lh "${archive}"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "mkdir -p '${REMOTE_WORK_DIR}/docker'"
  echo "Copying image archive to target:${REMOTE_WORK_DIR}/docker/ ..."
  scp -q -F "$SSH_CONFIG_FILE" "${archive}" \
    "rvs-target:${REMOTE_WORK_DIR}/docker/${DOCKER_IMAGE_ARCHIVE}"
  echo "Loading image on target ..."
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "gzip -dc '${REMOTE_WORK_DIR}/docker/${DOCKER_IMAGE_ARCHIVE}' | docker load"
  cmd_verify_image_on_target
}

cmd_verify_image_on_target() {
  require_ssh_config
  check_docker_on_target
  if ssh -q -F "$SSH_CONFIG_FILE" rvs-target "docker image inspect '${DOCKER_IMAGE}' >/dev/null 2>&1"; then
    echo "::notice::Image ${DOCKER_IMAGE} present on target node."
    return 0
  fi
  echo "::error::Image ${DOCKER_IMAGE} not found on target after transfer." >&2
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

  local url_escaped
  url_escaped=$(printf '%q' "$TARBALL_URL")

  docker_run "
    source /etc/profile.d/rocm-env.sh
    set -euo pipefail
    PKG=/tmp/${TARBALL_NAME}
    INSTALL_DIR=${INSTALL_DIR}
    RVS_BIN=${RVS_BIN}
    echo \"Downloading RVS tarball inside container...\"
    wget -q -O \"\${PKG}\" ${url_escaped}
    mkdir -p \"\${INSTALL_DIR}\"
    tar -xzf \"\${PKG}\" -C \"\${INSTALL_DIR}\"
    rm -f \"\${PKG}\"
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
    transfer-image-to-target) cmd_transfer_image_to_target ;;
    verify-image-on-target)   cmd_verify_image_on_target ;;
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
