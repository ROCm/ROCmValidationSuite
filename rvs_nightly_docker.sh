#!/usr/bin/env bash
################################################################################
# Run RVS nightly install + level 4 inside a ROCm-matched docker container on the
# self-hosted runner (GPU passthrough).
################################################################################

set -euo pipefail

DOCKER_IMAGE="${RVS_NIGHTLY_DOCKER_IMAGE:-rvs-nightly-rocm:latest}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: rvs_nightly_docker.sh <command>

Commands:
  pull-image          Ensure docker image is present locally
  verify-rocm         rocminfo + amd-smi inside container
  install-rvs         Extract RVS tarball under /opt/rocm/extras-<N> in container
  run-level4          rvs -r 4 inside container
  capture-versions    Write rvs_version and target_rocm_version to GITHUB_OUTPUT
  run-pipeline        verify-rocm → install-rvs → run-level4
EOF
}

require_env() {
  local n="$1"
  if [ -z "${!n:-}" ]; then
    echo "::error::Required environment variable $n is not set" >&2
    exit 1
  fi
}

host_rvs_install_dir() {
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

cmd_pull_image() {
  if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "::notice::Image ${DOCKER_IMAGE} already present locally."
    return 0
  fi
  echo "Pulling ${DOCKER_IMAGE} ..."
  docker pull "$DOCKER_IMAGE"
}

docker_run() {
  local -a install_mount=()
  if [ -n "${INSTALL_DIR:-}" ] && [ -n "${REMOTE_WORK_DIR:-}" ] && [ -n "${ROCM_MAJOR:-}" ]; then
    local host_install
    host_install="$(host_rvs_install_dir)"
    mkdir -p "$host_install"
    install_mount=(-v "${host_install}:${INSTALL_DIR}")
  fi
  # shellcheck disable=SC2046
  docker run --rm \
    $(docker_gpu_opts) \
    "${install_mount[@]}" \
    -v "${REPO_ROOT}:/workspace" \
    -v "${REPO_ROOT}/pkg:/pkg:ro" \
    -v "${REPO_ROOT}/reports:/reports" \
    -w /workspace \
    -e ROCM_PATH=/opt/rocm/install \
    -e TARGET_ROCM_PATH=/opt/rocm/install \
    "$DOCKER_IMAGE" \
    bash -lc "$1"
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
  require_env ROCM_MAJOR
  require_env INSTALL_DIR
  require_env RVS_BIN
  require_env REMOTE_WORK_DIR

  local pkg_host="${REPO_ROOT}/pkg/${TARBALL_NAME}"
  if [ ! -f "$pkg_host" ]; then
    echo "::error::RVS tarball not found at ${pkg_host}" >&2
    exit 1
  fi

  docker_run "
    source /etc/profile.d/rocm-env.sh
    set -euo pipefail
    PKG=/pkg/${TARBALL_NAME}
    INSTALL_DIR=${INSTALL_DIR}
    RVS_BIN=${RVS_BIN}
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

  local host_install
  host_install="$(host_rvs_install_dir)"
  if [ ! -x "${host_install}/bin/rvs" ]; then
    echo "::error::RVS not installed at ${host_install}/bin/rvs — run install-rvs first (same REMOTE_WORK_DIR=${REMOTE_WORK_DIR})" >&2
    exit 1
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
    pull-image)       cmd_pull_image ;;
    verify-rocm)      cmd_verify_rocm ;;
    install-rvs)      cmd_install_rvs ;;
    run-level4)       cmd_run_level4 ;;
    capture-versions) cmd_capture_versions ;;
    run-pipeline)     cmd_run_pipeline ;;
    -h|--help)        usage ;;
    *) echo "Unknown command: $1" >&2; usage; exit 1 ;;
  esac
}

main "$@"
