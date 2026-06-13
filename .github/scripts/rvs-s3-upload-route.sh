#!/bin/sh
# POSIX S3 path routing for RVS packages (ubuntu:22.04 container uses sh/dash).
# Usage:
#   rvs-s3-upload-route.sh upload-deb
#   rvs-s3-upload-route.sh upload-rpm-tar
#   rvs-s3-upload-route.sh deb-repo-prefix
#   rvs-s3-upload-route.sh rpm-repo-prefix
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

rvs_resolve_route() {
  RVS_S3_ROUTE="pr"
  RVS_S3_DEB_PREFIX=""
  RVS_S3_RPM_PREFIX=""
  RVS_S3_TAR_PREFIX=""
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
  *)
    echo "Usage: $0 upload-deb|upload-rpm-tar|deb-repo-prefix|rpm-repo-prefix" >&2
    exit 1
    ;;
esac
