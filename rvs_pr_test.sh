#!/bin/bash
################################################################################
# Remote RVS PR-package install + test driver (used by rvs-pr-tests.yml).
# Like rvs_nightly_test.sh but defaults to /tmp/rvs-pr-* staging; primary artifact
# is the amdrocm*-rvs-*-Linux.tar.gz relocatable tarball (optional .deb still supported).
################################################################################

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: rvs_pr_test.sh <command>

Commands (workflow order):
  resolve-package-url Discover latest Linux .tar.gz from rvs/ root, from a .../manylinux_*/ listing dir, or a direct .tar.gz / .deb URL;
                      writes tarball_url + tarball_name to GITHUB_OUTPUT / GITHUB_ENV when set
  peek-latest-artifact-key Print one-line artifact key (build/pr/tgz, tgz:basename, or deb:basename) for cache compare; stdout only
  validate-config     Fail fast; emit prepare job outputs to GITHUB_OUTPUT
  setup-ssh           Write SSH key/config under RUNNER_TEMP and verify connectivity
  verify-rocm         Remote rocminfo + amd-smi on TARGET_ROCM_PATH
  download-tarball    curl package (.deb or .tar.gz) to ./pkg/ on the orchestrator only
  copy-to-target      scp package to REMOTE_WORK_DIR/pkg on target
  install-rvs         Install .deb via dpkg on target, or extract .tar.gz under INSTALL_DIR
  verify-rvs-binary   Remote ldd check on RVS_BIN
  run-level4          Run rvs -r 4; write rc/start/end to GITHUB_OUTPUT if set
  collect-logs        scp remote logs to ./reports/
  capture-versions    Write rvs_version and target_rocm_version to GITHUB_OUTPUT
  build-report        Write ./reports/SUMMARY.md from env + prior outputs
  cleanup-remote      rm -rf REMOTE_WORK_DIR on target
  cleanup-local-ssh   Remove workflow-scoped key/config files
EOF
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "::error::Required environment variable $name is not set" >&2
    exit 1
  fi
}

ld_path_export() {
  echo "${INSTALL_DIR}/lib:${TARGET_ROCM_PATH}/lib/rocm_sysdeps/lib:${TARGET_ROCM_PATH}/lib/llvm/lib:${TARGET_ROCM_PATH}/lib"
}

# --- CDN rvs/ root → latest Linux tarball (orchestrator only; curl never on GPU target) ---

cdn_curl_listing() {
  local url="$1"
  curl -sSL --max-time 120 --retry 2 --retry-delay 2 \
    -A 'Mozilla/5.0 (compatible; RVS-PR-Tests/1.0)' "$url" 2>/dev/null || true
}

# Largest purely-numeric directory name from an S3/CF HTML index (href="1234/").
cdn_latest_numeric_dir() {
  local html="$1"
  printf '%s' "$html" | grep -oE 'href="[0-9]+/"' | sed 's/href="//;s/\/"//' | sort -n | tail -n 1 || true
}

# First subdirectory (non-parent) containing an amdrocm*-rvs-*-Linux.tar.gz link; prefer manylinux_*.
cdn_find_tgz_listing_url() {
  local pr_page_html="$1"
  local pr_base="$2"
  local line dir html pick
  pick=""
  while IFS= read -r line; do
    dir="${line%/}"
    [[ -z "$dir" || "$dir" == '..' ]] && continue
    html="$(cdn_curl_listing "${pr_base}${dir}/")"
    if ! printf '%s' "$html" | grep -qE 'amdrocm[0-9]+-rvs-[0-9A-Za-z._\-]+-Linux\.tar\.gz'; then
      continue
    fi
    if [[ "$dir" == manylinux* ]]; then
      pick="$dir"
      break
    fi
    if [ -z "$pick" ]; then
      pick="$dir"
    fi
  done < <(printf '%s' "$pr_page_html" | grep -oE 'href="[^./][^"]*/"' | sed 's/href="//;s/"$//' | grep -v '^\.\./$' || true)

  if [ -z "$pick" ]; then
    echo ""
    return 0
  fi
  printf '%s' "${pr_base}${pick}/"
}

# Args: internal rvs root URL without trailing slash (not a direct .tar.gz / .deb file).
# Prints: build<TAB>pr<TAB>tgz_name<TAB>tgz_https_url to stdout; errors to stderr; exit 1 on failure.
cdn_resolve_latest_merge_tgz_metadata() {
  local root="$1"
  local base="${root}/"
  local idx html build merge_base merge_html pr pr_page tgz_url tgz_dir_html tgz_name

  idx="$(cdn_curl_listing "$base")"
  if [ -z "$idx" ]; then
    echo "::error::Empty listing from ${base} (check URL and orchestrator HTTPS egress)." >&2
    return 1
  fi
  build="$(cdn_latest_numeric_dir "$idx")"
  if [ -z "$build" ]; then
    echo "::error::No numeric build directory under ${base}" >&2
    return 1
  fi
  html="$(cdn_curl_listing "${base}${build}/")"
  if ! printf '%s' "$html" | grep -q 'href="merge/"'; then
    echo "::error::Build ${build} has no merge/ directory (expected rvs/<id>/merge/...)." >&2
    return 1
  fi
  merge_base="${base}${build}/merge/"
  merge_html="$(cdn_curl_listing "$merge_base")"
  pr="$(cdn_latest_numeric_dir "$merge_html")"
  if [ -z "$pr" ]; then
    echo "::error::No numeric PR directory under ${merge_base}" >&2
    return 1
  fi
  pr_page="$(cdn_curl_listing "${merge_base}${pr}/")"
  tgz_url="$(cdn_find_tgz_listing_url "$pr_page" "${merge_base}${pr}/")"
  if [ -z "$tgz_url" ]; then
    echo "::error::No subdirectory under ${merge_base}${pr}/ listing amdrocm*-rvs-*-Linux.tar.gz." >&2
    return 1
  fi
  tgz_dir_html="$(cdn_curl_listing "$tgz_url")"
  tgz_name="$(printf '%s' "$tgz_dir_html" | grep -oE 'amdrocm[0-9]+-rvs-[0-9A-Za-z._\-]+-Linux\.tar\.gz' 2>/dev/null | sort -uV | tail -n 1 || true)"
  if [ -z "$tgz_name" ]; then
    echo "::error::Could not parse tarball filename under ${tgz_url}" >&2
    return 1
  fi
  printf '%s\t%s\t%s\t%s' "$build" "$pr" "$tgz_name" "${tgz_url}${tgz_name}"
}

# HTTPS directory listing ending in .../manylinux_*/ (e.g. PR build upload path).
# Prints: tarball_https_url<TAB>basename to stdout; errors to stderr; exit 1 on failure.
cdn_resolve_tgz_from_manylinux_dir() {
  local root="$1"
  root="${root%/}"
  local base="${root}/"
  local html tgz_name

  html="$(cdn_curl_listing "$base")"
  if [ -z "$html" ]; then
    echo "::error::Empty listing from ${base} (check URL and orchestrator HTTPS egress)." >&2
    return 1
  fi
  tgz_name="$(printf '%s' "$html" | grep -oE 'amdrocm[0-9]+-rvs-[0-9A-Za-z._\-]+-Linux\.tar\.gz' 2>/dev/null | sort -uV | tail -n 1 || true)"
  if [ -z "$tgz_name" ]; then
    echo "::error::No amdrocm*-rvs-*-Linux.tar.gz link under ${base}" >&2
    return 1
  fi
  printf '%s\t%s' "${base}${tgz_name}" "$tgz_name"
}

cmd_peek_latest_artifact_key() {
  local root="${PR_PACKAGE_ROOT_URL:-}"
  if [ -z "$root" ]; then
    root="${1:-}"
  fi
  if [ -z "${root:-}" ]; then
    echo "::error::Set PR_PACKAGE_ROOT_URL (HTTPS …/rvs index root or direct .tar.gz / .deb URL)." >&2
    exit 1
  fi
  root="${root%/}"

  if [[ "$root" == *.deb ]]; then
    printf 'deb:%s\n' "$(basename "${root%%\?*}")"
    return 0
  fi

  if [[ "$root" == *-Linux.tar.gz ]] || [[ "$root" == *.tar.gz ]]; then
    printf 'tgz:%s\n' "$(basename "${root%%\?*}")"
    return 0
  fi

  local meta
  if ! meta="$(cdn_resolve_latest_merge_tgz_metadata "$root")"; then
    exit 1
  fi
  local build pr tgz_name _url
  IFS=$'\t' read -r build pr tgz_name _url <<<"$meta"
  printf '%s/%s/%s\n' "$build" "$pr" "$tgz_name"
}

cmd_resolve_package_url() {
  local root="${PR_PACKAGE_ROOT_URL:-}"
  if [ -z "$root" ]; then
    root="${1:-}"
  fi
  if [ -z "${root:-}" ]; then
    echo "::error::Set PR_PACKAGE_ROOT_URL (HTTPS …/rvs index root or direct .tar.gz / .deb URL)." >&2
    exit 1
  fi

  root="${root%/}"
  local URL NAME

  # Direct HTTPS URL to relocatable Linux tarball (same artifact as nightly).
  if [[ "$root" == *-Linux.tar.gz ]] || [[ "$root" == *.tar.gz ]]; then
    URL="$root"
    NAME="$(basename "${URL%%\?*}")"
    echo "::notice::Using direct Linux tarball URL (basename: ${NAME})"
    if ! curl -fsSIL --max-time 120 -A 'Mozilla/5.0 (compatible; RVS-PR-Tests/1.0)' -o /dev/null "$URL"; then
      echo "::error::Tarball URL is not retrievable: ${URL}" >&2
      exit 1
    fi
  elif [[ "$root" == *.deb ]]; then
    URL="$root"
    NAME="$(basename "${URL%%\?*}")"
    echo "::notice::Using direct .deb URL (basename: ${NAME})"
    if ! curl -fsSIL --max-time 120 -A 'Mozilla/5.0 (compatible; RVS-PR-Tests/1.0)' -o /dev/null "$URL"; then
      echo "::error::Deb URL is not retrievable: ${URL}" >&2
      exit 1
    fi
  elif printf '%s' "$root" | grep -qE '/manylinux[^/]*$'; then
    local meta_ml
    echo "::notice::Resolving Linux tarball under manylinux listing: ${root}/"
    if ! meta_ml="$(cdn_resolve_tgz_from_manylinux_dir "$root")"; then
      exit 1
    fi
    IFS=$'\t' read -r URL NAME <<<"$meta_ml"
    echo "::notice::Resolved tarball: ${NAME}"
    if ! curl -fsSIL --max-time 120 -A 'Mozilla/5.0 (compatible; RVS-PR-Tests/1.0)' -o /dev/null "$URL"; then
      echo "::error::Resolved tarball URL is not retrievable: ${URL}" >&2
      exit 1
    fi
  else
    local base="${root}/"
    echo "::notice::Resolving latest Linux tarball under CDN root: ${base}"
    local meta build pr tgz_name tgz_url
    if ! meta="$(cdn_resolve_latest_merge_tgz_metadata "$root")"; then
      exit 1
    fi
    IFS=$'\t' read -r build pr tgz_name tgz_url <<<"$meta"
    echo "::notice::Latest build id: ${build}"
    echo "::notice::Latest merge PR directory: ${pr}"
    URL="$tgz_url"
    NAME="$tgz_name"
    echo "::notice::Resolved tarball: ${NAME}"
    if ! curl -fsSIL --max-time 120 -A 'Mozilla/5.0 (compatible; RVS-PR-Tests/1.0)' -o /dev/null "$URL"; then
      echo "::error::Resolved tarball URL is not retrievable: ${URL}" >&2
      exit 1
    fi
  fi

  if [[ "$NAME" != *.deb ]] && [[ "$NAME" != *-Linux.tar.gz ]]; then
    echo "::error::RVS PR Tests expected a *-Linux.tar.gz relocatable tarball or a .deb; got: ${NAME}" >&2
    exit 1
  fi

  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    # Write each output in its own multiline block so GitHub never merges tarball_url
    # body with the tarball_name line (which can leave tarball_name unset for job outputs).
    {
      echo "tarball_url<<RVS_PR_PACKAGE_URL"
      printf '%s\n' "$URL"
      echo "RVS_PR_PACKAGE_URL"
    } >> "$GITHUB_OUTPUT"
    {
      echo "tarball_name<<RVS_PR_PKG_NAME"
      printf '%s\n' "$NAME"
      echo "RVS_PR_PKG_NAME"
    } >> "$GITHUB_OUTPUT"
  fi

  if [ -n "${GITHUB_ENV:-}" ]; then
    {
      echo "TARBALL_URL<<RVS_PR_PACKAGE_URL"
      printf '%s\n' "$URL"
      echo "RVS_PR_PACKAGE_URL"
    } >> "$GITHUB_ENV"
    {
      echo "TARBALL_NAME<<RVS_PR_PKG_NAME"
      printf '%s\n' "$NAME"
      echo "RVS_PR_PKG_NAME"
    } >> "$GITHUB_ENV"
  fi

  echo "Package file : $NAME"
  echo "URL          : $URL"
}

cmd_validate_config() {
  require_env TARBALL_NAME
  require_env TARGET_NODE
  require_env TARGET_ROCM_PATH

  local input_remote="${INPUT_REMOTE_WORK_DIR:-}"
  local var_remote="${VAR_REMOTE_WORK_DIR:-}"
  local run_id="${GITHUB_RUN_ID:-local}"

  if [ -n "$input_remote" ]; then
    REMOTE_WORK_DIR="$input_remote"
  elif [ -n "$var_remote" ]; then
    REMOTE_WORK_DIR="$var_remote"
  else
    REMOTE_WORK_DIR="/tmp/rvs-pr-${run_id}"
  fi

  # amdrocm7-rvs-…-Linux.tar.gz or amdrocm7-rvs_….deb
  if [[ "$TARBALL_NAME" =~ ^amdrocm([0-9]+)- ]]; then
    ROCM_MAJOR="${BASH_REMATCH[1]}"
  else
    echo "::error::Cannot parse ROCm major version from package name: $TARBALL_NAME" >&2
    exit 1
  fi

  INSTALL_DIR="/opt/rocm/extras-${ROCM_MAJOR}"
  RVS_BIN="${INSTALL_DIR}/bin/rvs"

  # Do not print orchestrator or target host identity here (hostname/IP may be
  # workflow inputs and must not appear in CI logs). GitHub may still log runner
  # metadata in the job "Set up job" phase.
  echo "SSH target          : (omitted — use secrets RVS_TARGET_* / workflow inputs)"
  echo "Target ROCm path    : $TARGET_ROCM_PATH"
  echo "Remote work dir     : $REMOTE_WORK_DIR"
  echo "ROCm major          : $ROCM_MAJOR"
  echo "Expected RVS binary : $RVS_BIN"

  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    {
      echo "remote_work_dir=$REMOTE_WORK_DIR"
      echo "rocm_major=$ROCM_MAJOR"
      echo "install_dir=$INSTALL_DIR"
      echo "rvs_bin=$RVS_BIN"
    } >> "$GITHUB_OUTPUT"
  fi

  export REMOTE_WORK_DIR ROCM_MAJOR INSTALL_DIR RVS_BIN
}

ssh_state_paths() {
  local base="${RUNNER_TEMP:-/tmp}"
  SSH_KEY_FILE="${base}/rvs_target_key"
  SSH_CONFIG_FILE="${base}/rvs_target_ssh_config"
  mkdir -p "$base"
}

cmd_setup_ssh() {
  require_env TARGET_NODE
  require_env REMOTE_WORK_DIR

  ssh_state_paths

  if [ -n "${GITHUB_ENV:-}" ]; then
    {
      echo "SSH_KEY_FILE=$SSH_KEY_FILE"
      echo "SSH_CONFIG_FILE=$SSH_CONFIG_FILE"
    } >> "$GITHUB_ENV"
  fi

  if [ -z "${SSH_PRIVATE_KEY:-}" ]; then
    echo "::error::SSH_PRIVATE_KEY is not set; cannot SSH to target node." >&2
    exit 1
  fi

  install -m 600 /dev/null "$SSH_KEY_FILE"
  printf '%s\n' "$SSH_PRIVATE_KEY" > "$SSH_KEY_FILE"
  chmod 600 "$SSH_KEY_FILE"

  local known_hosts="${SSH_CONFIG_FILE}.known_hosts"
  : > "$known_hosts"
  chmod 600 "$known_hosts"
  ssh-keyscan -T 15 -H "$TARGET_NODE" >> "$known_hosts" 2>/dev/null || true

  cat > "$SSH_CONFIG_FILE" <<EOF
Host rvs-target
  HostName $TARGET_NODE
  User ${TARGET_USER:-}
  IdentityFile $SSH_KEY_FILE
  IdentitiesOnly yes
  BatchMode yes
  LogLevel ERROR
  StrictHostKeyChecking accept-new
  UserKnownHostsFile $known_hosts
  ServerAliveInterval 30
  ServerAliveCountMax 10
EOF
  chmod 600 "$SSH_CONFIG_FILE"

  echo "Verifying SSH connectivity to target node..."
  if ! ssh -q -F "$SSH_CONFIG_FILE" rvs-target 'true'; then
    echo "::error::SSH connectivity check failed. Verify secrets RVS_TARGET_NODE, RVS_TARGET_USER, RVS_TARGET_SSH_KEY and network access from this runner." >&2
    exit 1
  fi
  echo "::notice::SSH connectivity OK."

  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "mkdir -p '${REMOTE_WORK_DIR}/pkg' '${REMOTE_WORK_DIR}/reports'"
}

cmd_verify_rocm() {
  require_env SSH_CONFIG_FILE
  require_env TARGET_ROCM_PATH

  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "TARGET_ROCM_PATH='$TARGET_ROCM_PATH' bash -s" <<'REMOTE'
set -euo pipefail
echo "=== System ==="
# Omit nodename (uname -a prints hostname); -srvmo keeps kernel/OS/arch only.
uname -srvmo
echo
echo "=== Target ROCm path: ${TARGET_ROCM_PATH} ==="
if [ ! -d "${TARGET_ROCM_PATH}" ]; then
  echo "::error::${TARGET_ROCM_PATH} does not exist on the target node."
  exit 1
fi
echo "=== ${TARGET_ROCM_PATH}/bin/rocminfo ==="
"${TARGET_ROCM_PATH}/bin/rocminfo"
echo
echo "=== ${TARGET_ROCM_PATH}/bin/amd-smi version ==="
"${TARGET_ROCM_PATH}/bin/amd-smi" version
echo
echo "::notice::ROCm prerequisites OK on target node at ${TARGET_ROCM_PATH}"
REMOTE
}

cmd_download_tarball() {
  require_env TARBALL_URL
  require_env TARBALL_NAME
  mkdir -p ./pkg
  echo "Downloading RVS package on orchestrator runner (target node is not used for this fetch):"
  echo "  $TARBALL_URL"
  curl -fL --max-time 600 -o "./pkg/${TARBALL_NAME}" "${TARBALL_URL}"
  ls -la ./pkg/
  file "./pkg/${TARBALL_NAME}" || true
}

cmd_copy_to_target() {
  require_env SSH_CONFIG_FILE
  require_env TARBALL_NAME
  require_env REMOTE_WORK_DIR
  echo "Copying ./pkg/${TARBALL_NAME} -> rvs-target:${REMOTE_WORK_DIR}/pkg/"
  scp -q -F "$SSH_CONFIG_FILE" "./pkg/${TARBALL_NAME}" \
    "rvs-target:${REMOTE_WORK_DIR}/pkg/${TARBALL_NAME}"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "ls -la '${REMOTE_WORK_DIR}/pkg/'"
}

cmd_install_rvs() {
  require_env SSH_CONFIG_FILE
  require_env TARBALL_NAME
  require_env REMOTE_WORK_DIR
  require_env ROCM_MAJOR
  require_env INSTALL_DIR
  require_env RVS_BIN
  require_env TARGET_ROCM_PATH

  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "TARBALL_NAME='$TARBALL_NAME' REMOTE_WORK_DIR='$REMOTE_WORK_DIR' ROCM_MAJOR='$ROCM_MAJOR' INSTALL_DIR='$INSTALL_DIR' RVS_BIN='$RVS_BIN' TARGET_ROCM_PATH='$TARGET_ROCM_PATH' bash -s" <<'REMOTE'
set -euo pipefail
PKG="${REMOTE_WORK_DIR}/pkg/${TARBALL_NAME}"
echo "Target ROCm path            : ${TARGET_ROCM_PATH}"
echo "Detected ROCm major version : ${ROCM_MAJOR}"
echo "Install target              : ${INSTALL_DIR}"
echo "Expected RVS binary         : ${RVS_BIN}"
if [ ! -f "$PKG" ]; then
  echo "::error::Package file not found on target node at $PKG"
  exit 1
fi
if [[ "$TARBALL_NAME" == *.deb ]]; then
  echo "Installing .deb with dpkg (non-interactive)..."
  export DEBIAN_FRONTEND=noninteractive
  if sudo -n dpkg -i "$PKG"; then
    :
  else
    echo "::warning::dpkg -i had a non-zero exit; trying apt-get -f install then dpkg again"
    sudo -n apt-get update -qq || true
    sudo -n apt-get install -f -y -qq || true
    sudo -n dpkg -i "$PKG"
  fi
else
  probe="$INSTALL_DIR"
  while [ ! -d "$probe" ] && [ "$probe" != "/" ]; do probe=$(dirname "$probe"); done
  if [ -w "$probe" ]; then
    echo "Installing into $INSTALL_DIR without sudo (writable path)"
    mkdir -p "$INSTALL_DIR"
    tar -xzf "$PKG" -C "$INSTALL_DIR"
  else
    echo "Installing into $INSTALL_DIR via sudo -n (path not user-writable)"
    sudo -n mkdir -p "$INSTALL_DIR"
    sudo -n tar -xzf "$PKG" -C "$INSTALL_DIR"
  fi
fi
if [ ! -x "$RVS_BIN" ]; then
  echo "::error::rvs binary not found or not executable at $RVS_BIN after install"
  ls -la "$INSTALL_DIR/" || true
  ls -la "$INSTALL_DIR/bin/" || true
  exit 1
fi
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TARGET_ROCM_PATH}/lib/rocm_sysdeps/lib:${TARGET_ROCM_PATH}/lib/llvm/lib:${TARGET_ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
echo "Installed RVS at: $RVS_BIN"
"$RVS_BIN" --version || true
REMOTE
}

cmd_verify_rvs_binary() {
  require_env SSH_CONFIG_FILE
  require_env RVS_BIN
  require_env INSTALL_DIR
  require_env TARGET_ROCM_PATH

  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "RVS_BIN='$RVS_BIN' INSTALL_DIR='$INSTALL_DIR' TARGET_ROCM_PATH='$TARGET_ROCM_PATH' bash -s" <<'REMOTE'
set -euo pipefail
if [ ! -x "$RVS_BIN" ]; then
  echo "::error::$RVS_BIN was not produced by install on target."
  exit 1
fi
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TARGET_ROCM_PATH}/lib/rocm_sysdeps/lib:${TARGET_ROCM_PATH}/lib/llvm/lib:${TARGET_ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
echo "=== ldd $RVS_BIN ==="
LDD_OUTPUT=$(ldd "$RVS_BIN" 2>&1 || true)
echo "$LDD_OUTPUT"
if echo "$LDD_OUTPUT" | grep -q "not found"; then
  echo "::error::RVS binary has unresolved library dependencies on target (see above)."
  exit 1
fi
echo "::notice::RVS binary's library dependencies resolved OK on target."
REMOTE
}

cmd_run_level4() {
  require_env SSH_CONFIG_FILE
  require_env RVS_BIN
  require_env INSTALL_DIR
  require_env REMOTE_WORK_DIR
  require_env TARGET_ROCM_PATH

  set +e
  local start end rc
  start=$(date -u +%FT%TZ)
  echo "::group::RVS level 4 (${RVS_BIN} -r 4) against ${TARGET_ROCM_PATH}"
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
    "RVS_BIN='$RVS_BIN' INSTALL_DIR='$INSTALL_DIR' REMOTE_WORK_DIR='$REMOTE_WORK_DIR' TARGET_ROCM_PATH='$TARGET_ROCM_PATH' bash -s" <<'REMOTE'
set +e
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TARGET_ROCM_PATH}/lib/rocm_sysdeps/lib:${TARGET_ROCM_PATH}/lib/llvm/lib:${TARGET_ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${REMOTE_WORK_DIR}/reports"
"$RVS_BIN" -r 4 2>&1 | tee "${REMOTE_WORK_DIR}/reports/rvs_level_4.log"
RC=${PIPESTATUS[0]}
echo "remote_rc=$RC"
exit $RC
REMOTE
  rc=$?
  echo "::endgroup::"
  end=$(date -u +%FT%TZ)

  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "rc=$rc" >> "$GITHUB_OUTPUT"
    echo "start=$start" >> "$GITHUB_OUTPUT"
    echo "end=$end" >> "$GITHUB_OUTPUT"
  fi
  # Caller workflow decides whether to fail the job on non-zero rc.
  return 0
}

cmd_collect_logs() {
  require_env SSH_CONFIG_FILE
  require_env REMOTE_WORK_DIR
  mkdir -p ./reports
  if [ ! -f "$SSH_CONFIG_FILE" ]; then
    echo "::warning::SSH config not present; skipping log collection."
    return 0
  fi
  echo "Copying logs from rvs-target:${REMOTE_WORK_DIR}/reports/ -> ./reports/"
  scp -q -F "$SSH_CONFIG_FILE" "rvs-target:${REMOTE_WORK_DIR}/reports/*.log" ./reports/ || \
    echo "::warning::No log files retrieved from target node."
  ls -la ./reports/ || true
}

cmd_capture_versions() {
  require_env SSH_CONFIG_FILE
  require_env RVS_BIN
  require_env INSTALL_DIR
  require_env TARGET_ROCM_PATH

  local rvs_version target_rocm_version
  rvs_version=$(
    ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
      "RVS_BIN='$RVS_BIN' INSTALL_DIR='$INSTALL_DIR' TARGET_ROCM_PATH='$TARGET_ROCM_PATH' bash -s" <<'REMOTE' 2>/dev/null | head -1 || echo "unknown"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TARGET_ROCM_PATH}/lib/rocm_sysdeps/lib:${TARGET_ROCM_PATH}/lib/llvm/lib:${TARGET_ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
"$RVS_BIN" --version 2>/dev/null
REMOTE
  )
  target_rocm_version=$(
    ssh -q -F "$SSH_CONFIG_FILE" rvs-target \
      "TARGET_ROCM_PATH='$TARGET_ROCM_PATH' bash -s" <<'REMOTE' 2>/dev/null || echo "unknown"
cat "${TARGET_ROCM_PATH}/.info/version" 2>/dev/null \
  || cat "${TARGET_ROCM_PATH}/share/doc/rocm-core/version" 2>/dev/null \
  || echo "unknown"
REMOTE
  )

  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "rvs_version=$rvs_version" >> "$GITHUB_OUTPUT"
    echo "target_rocm_version=$target_rocm_version" >> "$GITHUB_OUTPUT"
  fi
}

cmd_build_report() {
  # Job outputs sometimes drop tarball_name when tarball_url is multiline; recover from URL.
  if [ -z "${TARBALL_NAME:-}" ] && [ -n "${TARBALL_URL:-}" ]; then
    TARBALL_NAME="$(basename "${TARBALL_URL%%\?*}")"
    export TARBALL_NAME
  fi
  require_env TARBALL_NAME
  require_env TARGET_ROCM_PATH
  require_env REMOTE_WORK_DIR
  require_env RVS_BIN

  # TARBALL_URL is optional: the report job reads it from install-rvs-on-target outputs.
  # If that output was empty (e.g. legacy echo-to-GITHUB_OUTPUT with special URL chars),
  # still emit SUMMARY.md with the tarball filename.
  local tarball_url_display="${TARBALL_URL:-}"
  if [ -z "$tarball_url_display" ]; then
    tarball_url_display="_(unavailable — check install job resolve step / use heredoc GITHUB_OUTPUT for URLs with special characters)_"
  fi

  local rc4="${RVS_LEVEL4_RC:-0}"
  local start="${RVS_LEVEL4_START:-}"
  local end="${RVS_LEVEL4_END:-}"
  local rvs_version="${RVS_VERSION:-unknown}"
  local target_rocm_version="${TARGET_ROCM_VERSION:-unknown}"
  local run_id="${GITHUB_RUN_ID:-local}"
  local server_url="${GITHUB_SERVER_URL:-https://github.com}"
  local repository="${GITHUB_REPOSITORY:-local/repo}"
  local event_name="${GITHUB_EVENT_NAME:-local}"

  mkdir -p ./reports
  local s4 overall
  if [ "$rc4" -eq 0 ]; then
    s4=PASS
    overall=PASS
  else
    s4=FAIL
    overall=FAIL
  fi

  local report_title="${REPORT_TITLE:-# RVS PR Tests Report}"
  local pkg_label="Linux tarball"
  if [[ "${TARBALL_NAME}" == *.deb ]]; then
    pkg_label="Deb package"
  fi
  local artifact_hint="${REPORT_ARTIFACT_HINT:-rvs-pr-report-${run_id}}"

  local report=./reports/SUMMARY.md
  {
    echo "${report_title}"
    echo ""
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| Run | [\`${run_id}\`](${server_url}/${repository}/actions/runs/${run_id}) |"
    echo "| Trigger | \`${event_name}\` |"
    echo "| Target ROCm path | \`${TARGET_ROCM_PATH}\` (version \`${target_rocm_version}\`) |"
    echo "| Remote work dir | \`${REMOTE_WORK_DIR}\` |"
    echo "| ${pkg_label} | \`${TARBALL_NAME}\` |"
    echo "| Source URL | ${tarball_url_display} |"
    echo "| RVS version | \`${rvs_version}\` |"
    echo "| Overall result | **${overall}** |"
    echo ""
    echo "## Results"
    echo ""
    echo "| Test | Command | Result | Exit | Started (UTC) | Ended (UTC) |"
    echo "|---|---|:--:|---:|---|---|"
    echo "| Level 4 | \`${RVS_BIN} -r 4\` | ${s4} | ${rc4} | ${start} | ${end} |"
    echo ""
    echo "## Logs"
    echo ""
    echo "Full stdout/stderr from the level-4 run is attached to the artifact"
    echo "\`${artifact_hint}\`:"
    echo ""
    echo "- \`rvs_level_4.log\`"
    echo "- \`SUMMARY.md\` (this file)"
  } > "$report"

  cat "$report"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    cat "$report" >> "$GITHUB_STEP_SUMMARY"
  fi

  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "overall=$overall" >> "$GITHUB_OUTPUT"
    echo "rc4=$rc4" >> "$GITHUB_OUTPUT"
  fi
}

cmd_cleanup_remote() {
  set +e
  if [ -z "${REMOTE_WORK_DIR:-}" ] || [ -z "${SSH_CONFIG_FILE:-}" ] || [ ! -f "$SSH_CONFIG_FILE" ]; then
    return 0
  fi
  ssh -q -F "$SSH_CONFIG_FILE" rvs-target "rm -rf '${REMOTE_WORK_DIR}'" || \
    echo "::warning::Failed to clean up ${REMOTE_WORK_DIR} on target node."
}

cmd_cleanup_local_ssh() {
  set +e
  if [ -n "${SSH_KEY_FILE:-}" ]; then
    rm -f "$SSH_KEY_FILE"
  fi
  if [ -n "${SSH_CONFIG_FILE:-}" ]; then
    rm -f "$SSH_CONFIG_FILE" "${SSH_CONFIG_FILE}.known_hosts"
  fi
}

main() {
  if [ $# -lt 1 ]; then
    usage
    exit 1
  fi
  case "$1" in
    resolve-package-url) cmd_resolve_package_url "${2:-}" ;;
    peek-latest-artifact-key) cmd_peek_latest_artifact_key "${2:-}" ;;
    validate-config)     cmd_validate_config ;;
    setup-ssh)           cmd_setup_ssh ;;
    verify-rocm)         cmd_verify_rocm ;;
    download-tarball)    cmd_download_tarball ;;
    copy-to-target)      cmd_copy_to_target ;;
    install-rvs)         cmd_install_rvs ;;
    verify-rvs-binary)   cmd_verify_rvs_binary ;;
    run-level4)          cmd_run_level4 ;;
    collect-logs)        cmd_collect_logs ;;
    capture-versions)    cmd_capture_versions ;;
    build-report)        cmd_build_report ;;
    cleanup-remote)      cmd_cleanup_remote ;;
    cleanup-local-ssh)   cmd_cleanup_local_ssh ;;
    -h|--help)           usage ;;
    *)
      echo "::error::Unknown command: $1" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
