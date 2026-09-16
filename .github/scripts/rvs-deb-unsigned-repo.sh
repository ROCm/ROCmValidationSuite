#!/bin/sh
# Build nightly/unsigned/deb APT archive (dists/ + pool/) via reprepro.
# Replaces the unsigned DEB tree each run (no merge with historical pool/).
# Usage: rvs-deb-unsigned-repo.sh [path-to-build-dir]
# Env: AWS_S3_BUCKET (required), optional RVS_UNSIGNED_DEB_PREFIX, GITHUB_RUN_ID, GITHUB_OUTPUT
set -eu

BUILD_DIR="${1:-./build}"
BUCKET="${AWS_S3_BUCKET:-}"
DEB_PREFIX="${RVS_UNSIGNED_DEB_PREFIX:-}"
META_OUT="${RVS_UNSIGNED_DEB_META_OUT:-${BUILD_DIR}/unsigned-deb-meta.json}"
RUN_ID="${GITHUB_RUN_ID:-}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)

if [ -z "$BUCKET" ]; then
  echo "::warning::AWS_S3_BUCKET not set. Skipping unsigned DEB repo update."
  exit 0
fi

if [ -z "$DEB_PREFIX" ]; then
  DEB_PREFIX=$(sh "${SCRIPT_DIR}/rvs-s3-upload-route.sh" unsigned-deb-prefix)
fi

if ! command -v reprepro >/dev/null 2>&1; then
  echo "Installing reprepro ..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends reprepro
fi

DEBS=$(find "$BUILD_DIR" -maxdepth 1 -name 'amdrocm*-rvs*.deb' 2>/dev/null | sort)
if [ -z "$DEBS" ]; then
  echo "::error::No amdrocm*-rvs*.deb in ${BUILD_DIR}; unsigned DEB publish requires a package." >&2
  exit 1
fi

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

mkdir -p "$STAGING/conf"
cat >"$STAGING/conf/distributions" <<'EOF'
Origin: ROCm Validation Suite
Label: stable
Suite: stable
Codename: stable
Architectures: amd64
Components: main
Description: RVS unsigned nightly
EOF

for deb in $DEBS; do
  echo "Including $(basename "$deb") ..."
  reprepro -b "$STAGING" includedeb stable "$deb"
done

if ! find "$STAGING/pool" -name '*.deb' 2>/dev/null | grep -q .; then
  echo "::error::reprepro produced no packages under pool/; aborting before S3 sync." >&2
  exit 1
fi

echo "Uploading conf/, pool/, and dists/ to s3://${BUCKET}/${DEB_PREFIX}/ (--delete stale objects) ..."
aws s3 sync "$STAGING/conf/" "s3://${BUCKET}/${DEB_PREFIX}/conf/" --delete --no-progress
aws s3 sync "$STAGING/pool/" "s3://${BUCKET}/${DEB_PREFIX}/pool/" --delete --no-progress
aws s3 sync "$STAGING/dists/" "s3://${BUCKET}/${DEB_PREFIX}/dists/" --delete --no-progress

echo "=== Unsigned DEB archive updated at s3://${BUCKET}/${DEB_PREFIX}/ ==="
aws s3 ls "s3://${BUCKET}/${DEB_PREFIX}/dists/stable/" --human-readable 2>/dev/null || true

# Meta for nightly/unsigned/latest.json (signing CI).
python3 <<PY
import hashlib
import json
import os
from pathlib import Path

staging = Path("${STAGING}")
deb_prefix = "${DEB_PREFIX}"
run_id = "${RUN_ID}"
packages = []
for deb_path in sorted(staging.glob("pool/**/*.deb")):
    rel = deb_path.relative_to(staging).as_posix()
    s3_key = f"{deb_prefix}/{rel}"
    h = hashlib.sha256()
    with open(deb_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    packages.append(
        {
            "filename": deb_path.name,
            "pool_key": rel,
            "s3_key": s3_key,
            "sha256": h.hexdigest(),
        }
    )

if not packages:
    import sys
    print("::error::No .deb files in pool/ after reprepro; aborting deb.json upload.", file=sys.stderr)
    sys.exit(1)

payload = {
    "github_run_id": run_id,
    "deb_prefix": deb_prefix,
    "packages": packages,
}
out = Path("${META_OUT}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out}")
PY

if [ -n "$RUN_ID" ]; then
  aws s3 cp "$META_OUT" "s3://${BUCKET}/nightly/unsigned/runs/${RUN_ID}/deb.json" --no-progress
fi
