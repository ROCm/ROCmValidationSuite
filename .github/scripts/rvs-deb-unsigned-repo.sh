#!/bin/sh
# Build or update nightly/unsigned/deb APT archive (dists/ + pool/) via reprepro.
# Usage: rvs-deb-unsigned-repo.sh [path-to-build-dir]
# Env: AWS_S3_BUCKET (required), optional RVS_UNSIGNED_DEB_PREFIX (default nightly/unsigned/deb)
set -eu

BUILD_DIR="${1:-./build}"
BUCKET="${AWS_S3_BUCKET:-}"
DEB_PREFIX="${RVS_UNSIGNED_DEB_PREFIX:-}"

if [ -z "$BUCKET" ]; then
  echo "::warning::AWS_S3_BUCKET not set. Skipping unsigned DEB repo update."
  exit 0
fi

if [ -z "$DEB_PREFIX" ]; then
  DEB_PREFIX=$(sh "$(dirname "$0")/rvs-s3-upload-route.sh" unsigned-deb-prefix)
fi

if ! command -v reprepro >/dev/null 2>&1; then
  echo "::error::reprepro is required for unsigned DEB archive layout." >&2
  exit 1
fi

DEBS=$(find "$BUILD_DIR" -maxdepth 1 -name 'amdrocm*-rvs*.deb' 2>/dev/null | sort)
if [ -z "$DEBS" ]; then
  echo "::warning::No amdrocm*-rvs*.deb in ${BUILD_DIR}; skipping unsigned DEB repo."
  exit 0
fi

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

mkdir -p "$STAGING/conf"
echo "Downloading existing unsigned DEB archive from s3://${BUCKET}/${DEB_PREFIX}/ ..."
aws s3 sync "s3://${BUCKET}/${DEB_PREFIX}/conf/" "$STAGING/conf/" --no-progress 2>/dev/null || true
aws s3 sync "s3://${BUCKET}/${DEB_PREFIX}/pool/" "$STAGING/pool/" --no-progress 2>/dev/null || true
aws s3 sync "s3://${BUCKET}/${DEB_PREFIX}/dists/" "$STAGING/dists/" --no-progress 2>/dev/null || true

if [ ! -f "$STAGING/conf/distributions" ]; then
  cat >"$STAGING/conf/distributions" <<'EOF'
Origin: ROCm Validation Suite
Label: stable
Suite: stable
Codename: stable
Architectures: amd64
Components: main
Description: RVS unsigned nightly
EOF
fi

for deb in $DEBS; do
  echo "Including $(basename "$deb") in suite stable ..."
  reprepro -b "$STAGING" includedeb stable "$deb"
done

echo "Uploading conf/, pool/, and dists/ to s3://${BUCKET}/${DEB_PREFIX}/ ..."
aws s3 sync "$STAGING/conf/" "s3://${BUCKET}/${DEB_PREFIX}/conf/" --no-progress
aws s3 sync "$STAGING/pool/" "s3://${BUCKET}/${DEB_PREFIX}/pool/" --no-progress
aws s3 sync "$STAGING/dists/" "s3://${BUCKET}/${DEB_PREFIX}/dists/" --no-progress

echo "=== Unsigned DEB archive updated at s3://${BUCKET}/${DEB_PREFIX}/ ==="
aws s3 ls "s3://${BUCKET}/${DEB_PREFIX}/dists/stable/" --human-readable 2>/dev/null || true
