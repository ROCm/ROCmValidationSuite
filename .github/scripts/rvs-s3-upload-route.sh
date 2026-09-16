#!/bin/sh
# POSIX S3 path routing for RVS packages (ubuntu:22.04 container uses sh/dash).
# Usage:
#   rvs-s3-upload-route.sh upload-deb
#   rvs-s3-upload-route.sh upload-rpm-tar
#   rvs-s3-upload-route.sh deb-repo-prefix
#   rvs-s3-upload-route.sh rpm-repo-prefix
#   rvs-s3-upload-route.sh unsigned-deb-prefix
#   rvs-s3-upload-route.sh unsigned-rpm-prefix
#   rvs-s3-upload-route.sh unsigned-tar-prefix
#   rvs-s3-upload-route.sh upload-rpm-tar-unsigned
#   rvs-s3-upload-route.sh unsigned-upload-enabled
set -eu

BASE="rvs"
EVENT="${GITHUB_EVENT_NAME:-}"
REF="${GITHUB_REF:-}"
REF_NAME="${GITHUB_REF_NAME:-}"
RUN_NUMBER="${GITHUB_RUN_NUMBER:-0}"
BUCKET="${AWS_S3_BUCKET:-}"
GITHUB_OUTPUT="${GITHUB_OUTPUT:-/dev/null}"

RVS_SKIP_UPLOAD="${RVS_SKIP_UPLOAD:-false}"
RVS_IS_DEFAULT_BRANCH="${RVS_IS_DEFAULT_BRANCH:-false}"
RVS_BRANCH_PREFIX="${RVS_BRANCH_PREFIX:-}"
RVS_BUILD_REF_NAME="${RVS_BUILD_REF_NAME:-}"
RUNNER_SUFFIX="${RVS_RUNNER_SUFFIX:-ubuntu-22.04}"

rvs_is_release_ref() {
  case "$1" in
    refs/heads/release/*) return 0 ;;
  esac
  return 1
}

rvs_unsigned_upload_enabled() {
  [ "$EVENT" = "schedule" ] && [ "$RVS_IS_DEFAULT_BRANCH" = "true" ]
}

rvs_resolve_route() {
  RVS_S3_ROUTE="pr"
  RVS_S3_DEB_PREFIX=""
  RVS_S3_RPM_PREFIX=""
  RVS_S3_TAR_PREFIX=""
  RVS_S3_UNSIGNED_DEB_PREFIX=""
  RVS_S3_UNSIGNED_RPM_PREFIX=""
  RVS_S3_UNSIGNED_TAR_PREFIX=""
  RVS_APT_SUITE="rvs-nightly"
  RVS_S3_OUTPUT_PATHS=""

  if [ "$RVS_SKIP_UPLOAD" = "true" ]; then
    RVS_S3_ROUTE="skip"
    return 0
  fi

  if [ "$EVENT" = "schedule" ] && [ "$RVS_IS_DEFAULT_BRANCH" != "true" ]; then
    RVS_S3_ROUTE="scheduled_branch"
    RVS_S3_DEB_PREFIX="${RVS_BRANCH_PREFIX}/${RVS_BUILD_REF_NAME}/nightly/deb"
    RVS_S3_RPM_PREFIX="${RVS_BRANCH_PREFIX}/${RVS_BUILD_REF_NAME}/nightly/rpm"
    RVS_S3_TAR_PREFIX="${RVS_BRANCH_PREFIX}/${RVS_BUILD_REF_NAME}/nightly/tar"
    RVS_S3_OUTPUT_PATHS="Ubuntu DEB|${RVS_BRANCH_PREFIX}/${RVS_BUILD_REF_NAME}/nightly/deb||CentOS/RHEL RPM|${RVS_BRANCH_PREFIX}/${RVS_BUILD_REF_NAME}/nightly/rpm||CentOS/RHEL TGZ|${RVS_BRANCH_PREFIX}/${RVS_BUILD_REF_NAME}/nightly/tar"
    return 0
  fi

  if rvs_is_release_ref "$REF" && { [ "$EVENT" = "push" ] || [ "$EVENT" = "workflow_dispatch" ]; }; then
    RVS_S3_ROUTE="release"
    RVS_S3_DEB_PREFIX="release/${BASE}/deb"
    RVS_S3_RPM_PREFIX="release/${BASE}/rpm"
    RVS_S3_TAR_PREFIX="release/${BASE}/tar"
    RVS_APT_SUITE="rvs-release"
    RVS_S3_OUTPUT_PATHS="Ubuntu DEB|release/${BASE}/deb||CentOS/RHEL RPM|release/${BASE}/rpm||CentOS/RHEL TGZ|release/${BASE}/tar"
    return 0
  fi

  if [ "$EVENT" = "schedule" ] || [ "$EVENT" = "push" ] || [ "$EVENT" = "workflow_dispatch" ]; then
    RVS_S3_ROUTE="nightly"
    RVS_S3_DEB_PREFIX="nightly/${BASE}/deb"
    RVS_S3_RPM_PREFIX="nightly/${BASE}/rpm"
    RVS_S3_TAR_PREFIX="nightly/${BASE}/tar"
    RVS_APT_SUITE="rvs-nightly"
    RVS_S3_OUTPUT_PATHS="Ubuntu DEB|nightly/${BASE}/deb||CentOS/RHEL RPM|nightly/${BASE}/rpm||CentOS/RHEL TGZ|nightly/${BASE}/tar"
    if rvs_unsigned_upload_enabled; then
      RVS_S3_UNSIGNED_DEB_PREFIX="nightly/unsigned/deb"
      RVS_S3_UNSIGNED_RPM_PREFIX="nightly/unsigned/rpm"
      RVS_S3_UNSIGNED_TAR_PREFIX="nightly/unsigned/tar"
      RVS_S3_OUTPUT_PATHS="${RVS_S3_OUTPUT_PATHS}||Unsigned DEB|nightly/unsigned/deb||Unsigned RPM|nightly/unsigned/rpm||Unsigned TGZ|nightly/unsigned/tar"
    fi
    return 0
  fi

  RVS_S3_ROUTE="pr"
  RVS_S3_DEB_PREFIX="${BASE}/${REF_NAME}/${RUN_NUMBER}/${RUNNER_SUFFIX}"
  RVS_S3_RPM_PREFIX="${BASE}/${REF_NAME}/${RUN_NUMBER}/manylinux_2_28"
  RVS_S3_TAR_PREFIX="${RVS_S3_RPM_PREFIX}"
  RVS_S3_OUTPUT_PATHS="Ubuntu DEB|${RVS_S3_DEB_PREFIX}||CentOS/RHEL packages|${RVS_S3_RPM_PREFIX}"
}

cmd="${1:-}"
rvs_resolve_route

case "$cmd" in
  upload-deb)
    if [ -z "$BUCKET" ]; then
      echo "::warning::AWS_S3_BUCKET not set. Skipping S3 upload."
      exit 0
    fi
    if [ "$RVS_S3_ROUTE" = "skip" ]; then
      echo "Skipping S3 upload (scheduled release* branch)."
      exit 0
    fi
    case "$RVS_S3_ROUTE" in
      scheduled_branch)
        echo "Scheduled ACTIVE_BRANCHES build: uploading to ${RVS_S3_DEB_PREFIX}"
        ;;
      release)
        echo "Release branch build: uploading to ${RVS_S3_DEB_PREFIX}"
        ;;
      nightly)
        echo "Nightly/push build: uploading to ${RVS_S3_DEB_PREFIX}"
        ;;
      *)
        echo "Uploading to s3://${BUCKET}/${RVS_S3_DEB_PREFIX}/"
        ;;
    esac
    aws s3 cp ./build "s3://${BUCKET}/${RVS_S3_DEB_PREFIX}/" \
      --recursive --exclude "*" --include "amdrocm*-rvs*.deb" --no-progress
    echo "Listing s3://${BUCKET}/${RVS_S3_DEB_PREFIX}/"
    aws s3 ls "s3://${BUCKET}/${RVS_S3_DEB_PREFIX}/" --human-readable || true
    echo "bucket=${BUCKET}" >> "$GITHUB_OUTPUT"
    echo "paths=Ubuntu DEB|${RVS_S3_DEB_PREFIX}" >> "$GITHUB_OUTPUT"
    echo "Done."
    ;;
  upload-rpm-tar)
    if [ -z "$BUCKET" ]; then
      echo "::warning::AWS_S3_BUCKET not set. Skipping S3 upload."
      exit 0
    fi
    if [ "$RVS_S3_ROUTE" = "skip" ]; then
      echo "Skipping S3 upload (scheduled release* branch)."
      exit 0
    fi
    case "$RVS_S3_ROUTE" in
      scheduled_branch)
        echo "Scheduled ACTIVE_BRANCHES build: uploading to ${RVS_S3_RPM_PREFIX} and ${RVS_S3_TAR_PREFIX}"
        ;;
      release)
        echo "Release branch build: uploading to ${RVS_S3_RPM_PREFIX} and ${RVS_S3_TAR_PREFIX}"
        ;;
      nightly)
        echo "Nightly/push build: uploading to ${RVS_S3_RPM_PREFIX} and ${RVS_S3_TAR_PREFIX}"
        ;;
      *)
        echo "Uploading to s3://${BUCKET}/${RVS_S3_RPM_PREFIX}/"
        ;;
    esac
    if [ "$RVS_S3_ROUTE" = "pr" ]; then
      aws s3 cp ./build "s3://${BUCKET}/${RVS_S3_RPM_PREFIX}/" \
        --recursive --exclude "*" --include "amdrocm*-rvs*.rpm" --include "amdrocm*-rvs*.tar.gz" --no-progress
      echo "Listing s3://${BUCKET}/${RVS_S3_RPM_PREFIX}/"
      aws s3 ls "s3://${BUCKET}/${RVS_S3_RPM_PREFIX}/" --human-readable || true
      echo "bucket=${BUCKET}" >> "$GITHUB_OUTPUT"
      echo "paths=CentOS/RHEL packages|${RVS_S3_RPM_PREFIX}" >> "$GITHUB_OUTPUT"
    else
      aws s3 cp ./build "s3://${BUCKET}/${RVS_S3_RPM_PREFIX}/" \
        --recursive --exclude "*" --include "amdrocm*-rvs*.rpm" --no-progress
      aws s3 cp ./build "s3://${BUCKET}/${RVS_S3_TAR_PREFIX}/" \
        --recursive --exclude "*" --include "amdrocm*-rvs*.tar.gz" --no-progress
      echo "Listing s3://${BUCKET}/${RVS_S3_RPM_PREFIX}/"
      aws s3 ls "s3://${BUCKET}/${RVS_S3_RPM_PREFIX}/" --human-readable || true
      echo "Listing s3://${BUCKET}/${RVS_S3_TAR_PREFIX}/"
      aws s3 ls "s3://${BUCKET}/${RVS_S3_TAR_PREFIX}/" --human-readable || true
      echo "bucket=${BUCKET}" >> "$GITHUB_OUTPUT"
      echo "paths=CentOS/RHEL RPM|${RVS_S3_RPM_PREFIX}||CentOS/RHEL TGZ|${RVS_S3_TAR_PREFIX}" >> "$GITHUB_OUTPUT"
    fi
    echo "Done."
    ;;
  deb-repo-prefix)
    echo "$RVS_S3_DEB_PREFIX"
    echo "$RVS_APT_SUITE"
    ;;
  rpm-repo-prefix)
    echo "$RVS_S3_RPM_PREFIX"
    ;;
  unsigned-upload-enabled)
    if rvs_unsigned_upload_enabled; then
      echo "true"
    else
      echo "false"
    fi
    ;;
  unsigned-deb-prefix)
    if [ -z "$RVS_S3_UNSIGNED_DEB_PREFIX" ]; then
      echo "::error::Unsigned DEB upload is only enabled for scheduled default-branch builds." >&2
      exit 1
    fi
    echo "$RVS_S3_UNSIGNED_DEB_PREFIX"
    ;;
  unsigned-rpm-prefix)
    if [ -z "$RVS_S3_UNSIGNED_RPM_PREFIX" ]; then
      echo "::error::Unsigned RPM upload is only enabled for scheduled default-branch builds." >&2
      exit 1
    fi
    echo "$RVS_S3_UNSIGNED_RPM_PREFIX"
    ;;
  unsigned-tar-prefix)
    if [ -z "$RVS_S3_UNSIGNED_TAR_PREFIX" ]; then
      echo "::error::Unsigned TAR upload is only enabled for scheduled default-branch builds." >&2
      exit 1
    fi
    echo "$RVS_S3_UNSIGNED_TAR_PREFIX"
    ;;
  upload-rpm-tar-unsigned)
    if [ -z "$BUCKET" ]; then
      echo "::warning::AWS_S3_BUCKET not set. Skipping unsigned S3 upload."
      exit 0
    fi
    if [ -z "$RVS_S3_UNSIGNED_RPM_PREFIX" ] || [ -z "$RVS_S3_UNSIGNED_TAR_PREFIX" ]; then
      echo "Skipping unsigned S3 upload (not a scheduled default-branch build)."
      exit 0
    fi
    rpm_count=0
    for f in ./build/amdrocm*-rvs*.rpm; do
      [ -f "$f" ] || continue
      rpm_count=$((rpm_count + 1))
    done
    tar_count=0
    for f in ./build/amdrocm*-rvs*.tar.gz; do
      [ -f "$f" ] || continue
      tar_count=$((tar_count + 1))
      if [ ! -f "${f}.sha256" ]; then
        echo "::error::Missing SHA-256 sidecar for $(basename "$f"); run sha256sum before unsigned upload." >&2
        exit 1
      fi
    done
    if [ "$rpm_count" -lt 1 ]; then
      echo "::error::No amdrocm*-rvs*.rpm in ./build; refusing unsigned RPM sync --delete." >&2
      exit 1
    fi
    if [ "$tar_count" -lt 1 ]; then
      echo "::error::No amdrocm*-rvs*.tar.gz in ./build; refusing unsigned TAR sync --delete." >&2
      exit 1
    fi

    echo "Scheduled unsigned build: replacing ${RVS_S3_UNSIGNED_RPM_PREFIX} and ${RVS_S3_UNSIGNED_TAR_PREFIX}"
    RPM_STAGING=$(mktemp -d)
    TAR_STAGING=$(mktemp -d)
    for f in ./build/amdrocm*-rvs*.rpm; do
      [ -f "$f" ] || continue
      cp "$f" "$RPM_STAGING/"
    done
    for f in ./build/amdrocm*-rvs*.tar.gz ./build/amdrocm*-rvs*.tar.gz.sha256; do
      [ -f "$f" ] || continue
      cp "$f" "$TAR_STAGING/"
    done
    aws s3 sync "$RPM_STAGING/" "s3://${BUCKET}/${RVS_S3_UNSIGNED_RPM_PREFIX}/" \
      --delete --no-progress
    aws s3 sync "$TAR_STAGING/" "s3://${BUCKET}/${RVS_S3_UNSIGNED_TAR_PREFIX}/" \
      --delete --no-progress
    rm -rf "$RPM_STAGING" "$TAR_STAGING"
    echo "Listing s3://${BUCKET}/${RVS_S3_UNSIGNED_RPM_PREFIX}/"
    aws s3 ls "s3://${BUCKET}/${RVS_S3_UNSIGNED_RPM_PREFIX}/" --human-readable || true
    echo "Listing s3://${BUCKET}/${RVS_S3_UNSIGNED_TAR_PREFIX}/"
    aws s3 ls "s3://${BUCKET}/${RVS_S3_UNSIGNED_TAR_PREFIX}/" --human-readable || true
    echo "Done."
    ;;
  *)
    echo "Usage: $0 upload-deb|upload-rpm-tar|deb-repo-prefix|rpm-repo-prefix|unsigned-deb-prefix|unsigned-rpm-prefix|unsigned-tar-prefix|upload-rpm-tar-unsigned|unsigned-upload-enabled" >&2
    exit 1
    ;;
esac
