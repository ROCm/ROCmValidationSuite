#!/bin/sh
# Accumulate nightly/unsigned/deb APT archive (dists/ + pool/) via reprepro.
# Merges this run's .deb into the existing S3 archive (no s3:DeleteObject / --delete).
# Usage: rvs-deb-unsigned-repo.sh [path-to-build-dir]
# Env: AWS_S3_BUCKET (required), optional RVS_UNSIGNED_DEB_PREFIX, GITHUB_RUN_ID
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
echo "Downloading existing unsigned DEB archive from s3://${BUCKET}/${DEB_PREFIX}/ ..."
aws s3 sync "s3://${BUCKET}/${DEB_PREFIX}/conf/" "$STAGING/conf/" --no-progress
aws s3 sync "s3://${BUCKET}/${DEB_PREFIX}/pool/" "$STAGING/pool/" --no-progress
aws s3 sync "s3://${BUCKET}/${DEB_PREFIX}/dists/" "$STAGING/dists/" --no-progress

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

# Track this run's pool keys for latest.json (not the entire historical pool).
NEW_META=$(mktemp)
trap 'rm -rf "$STAGING" "$NEW_META"' EXIT
: >"$NEW_META"

deb_count=0
for deb in $DEBS; do
  deb_count=$((deb_count + 1))
  pkg=$(dpkg-deb -f "$deb" Package)
  ver=$(dpkg-deb -f "$deb" Version)
  # Same Package+Version already in the archive: drop that version from the local
  # db/pool then re-include so PutObject can overwrite the pool object.
  # Does not require s3:DeleteObject (orphan keys with other filenames may remain).
  if reprepro -b "$STAGING" listfilter stable "Package (== ${pkg}), Version (== ${ver})" 2>/dev/null | grep -q .; then
    echo "Package ${pkg} ${ver} already in suite stable; removing that version from local archive before re-include ..."
    reprepro -b "$STAGING" -T deb removefilter stable "Package (== ${pkg}), Version (== ${ver})"
  fi
  echo "Including $(basename "$deb") ..."
  reprepro -b "$STAGING" includedeb stable "$deb"

  # Resolve the pool path reprepro recorded for this exact Package+Version from
  # its own Packages index. Using find is non-deterministic (undefined order) and
  # a glob fallback on ${pkg}_*.deb would match every historical version in pool/.
  PACKAGES_FILE="$STAGING/dists/stable/main/binary-amd64/Packages"
  pool_rel=$(awk -v pkg="$pkg" -v ver="$ver" '
    /^Package:/  { cur_pkg=$2; cur_ver=""; cur_fn="" }
    /^Version:/  { cur_ver=$2 }
    /^Filename:/ { cur_fn=$2 }
    /^$/         { if (cur_pkg==pkg && cur_ver==ver && cur_fn!="") { print cur_fn; cur_pkg=""; cur_ver=""; cur_fn="" } }
    END          { if (cur_pkg==pkg && cur_ver==ver && cur_fn!="") print cur_fn }
  ' "$PACKAGES_FILE")

  if [ -z "$pool_rel" ]; then
    echo "::error::Cannot resolve pool path for ${pkg} ${ver} from reprepro Packages index." >&2
    exit 1
  fi
  echo "$pool_rel" >>"$NEW_META"
done

if ! find "$STAGING/pool" -name '*.deb' 2>/dev/null | grep -q .; then
  echo "::error::reprepro produced no packages under pool/; aborting before S3 sync." >&2
  exit 1
fi

meta_count=$(wc -l < "$NEW_META")
if [ "$meta_count" -ne "$deb_count" ]; then
  echo "::error::Resolved ${meta_count} pool path(s) for ${deb_count} .deb file(s); counts must match." >&2
  exit 1
fi

echo "Uploading conf/, pool/, and dists/ to s3://${BUCKET}/${DEB_PREFIX}/ (accumulate; no --delete) ..."
aws s3 sync "$STAGING/conf/" "s3://${BUCKET}/${DEB_PREFIX}/conf/" --no-progress
aws s3 sync "$STAGING/pool/" "s3://${BUCKET}/${DEB_PREFIX}/pool/" --no-progress
aws s3 sync "$STAGING/dists/" "s3://${BUCKET}/${DEB_PREFIX}/dists/" --no-progress

echo "=== Unsigned DEB archive updated at s3://${BUCKET}/${DEB_PREFIX}/ ==="
aws s3 ls "s3://${BUCKET}/${DEB_PREFIX}/dists/stable/" --human-readable 2>/dev/null || true

# Meta for nightly/unsigned/latest.json: this run's packages only.
python3 <<PY
import hashlib
import json
from pathlib import Path

staging = Path("${STAGING}")
deb_prefix = "${DEB_PREFIX}"
run_id = "${RUN_ID}"
meta_list = Path("${NEW_META}")
packages = []
for line in meta_list.read_text(encoding="utf-8").splitlines():
    rel = line.strip()
    if not rel:
        continue
    deb_path = staging / rel
    if not deb_path.is_file():
        raise SystemExit(f"::error::Missing staged deb for meta: {rel}")
    h = hashlib.sha256()
    with open(deb_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    packages.append(
        {
            "filename": deb_path.name,
            "pool_key": rel,
            "s3_key": f"{deb_prefix}/{rel}",
            "sha256": h.hexdigest(),
        }
    )

if not packages:
    import sys
    print("::error::No packages for this-run deb.json; aborting.", file=sys.stderr)
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

if [ -z "$RUN_ID" ]; then
  echo "::error::GITHUB_RUN_ID unset; cannot upload deb.json for signing CI." >&2
  exit 1
fi
aws s3 cp "$META_OUT" "s3://${BUCKET}/nightly/unsigned/runs/${RUN_ID}/deb.json" --no-progress
