#!/bin/sh
# Merge per-run unsigned metadata and publish nightly/unsigned/latest.json for signing CI.
# Env: AWS_S3_BUCKET (required), GITHUB_RUN_ID, GITHUB_SHA, ROCM_VERSION (optional)
set -eu

BUCKET="${AWS_S3_BUCKET:-}"
RUN_ID="${GITHUB_RUN_ID:-}"
SHA="${GITHUB_SHA:-}"
ROCM_VERSION="${ROCM_VERSION:-}"

if [ -z "$BUCKET" ]; then
  echo "::warning::AWS_S3_BUCKET unset; skipping latest.json publish."
  exit 0
fi

if [ -z "$RUN_ID" ]; then
  echo "::error::GITHUB_RUN_ID unset; cannot publish latest.json." >&2
  exit 1
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

DEB_JSON="$WORKDIR/deb.json"
RPM_JSON="$WORKDIR/rpm-tar.json"

if ! aws s3 cp "s3://${BUCKET}/nightly/unsigned/runs/${RUN_ID}/deb.json" "$DEB_JSON" --no-progress; then
  echo "::error::Missing s3://${BUCKET}/nightly/unsigned/runs/${RUN_ID}/deb.json (unsigned DEB step)." >&2
  exit 1
fi

if ! aws s3 cp "s3://${BUCKET}/nightly/unsigned/runs/${RUN_ID}/rpm-tar.json" "$RPM_JSON" --no-progress; then
  echo "::error::Missing s3://${BUCKET}/nightly/unsigned/runs/${RUN_ID}/rpm-tar.json (unsigned RPM/TAR step)." >&2
  exit 1
fi

python3 <<PY
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

deb = json.loads(Path("${DEB_JSON}").read_text(encoding="utf-8"))
rpm_tar = json.loads(Path("${RPM_JSON}").read_text(encoding="utf-8"))

packages = deb.get("packages") or []
if not packages:
    print("::error::deb.json has no packages", file=sys.stderr)
    sys.exit(1)

rpm = rpm_tar.get("rpm")
tar = rpm_tar.get("tar")
if not rpm or not rpm.get("s3_key") or not rpm.get("sha256"):
    print("::error::rpm-tar.json missing rpm.s3_key or rpm.sha256", file=sys.stderr)
    sys.exit(1)
if not tar or not tar.get("s3_key") or not tar.get("sha256"):
    print("::error::rpm-tar.json missing tar.s3_key or tar.sha256", file=sys.stderr)
    sys.exit(1)
if not tar.get("sha256_sidecar_key"):
    print("::error::rpm-tar.json missing tar.sha256_sidecar_key", file=sys.stderr)
    sys.exit(1)

latest = {
    "github_run_id": "${RUN_ID}",
    "github_sha": "${SHA}",
    "rocm_version": os.environ.get("ROCM_VERSION") or None,
    "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "deb": deb,
    "rpm": rpm,
    "tar": tar,
}
out = Path("${WORKDIR}/latest.json")
out.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
print(out.read_text(encoding="utf-8"))

def _rewrite(key, old, new):
    if isinstance(key, str) and key.startswith(old):
        return new + key[len(old):]
    return key

signed = json.loads(json.dumps(latest))
deb = signed.get("deb") or {}
if deb.get("deb_prefix"):
    deb["deb_prefix"] = _rewrite(
        deb["deb_prefix"],
        "nightly/unsigned/packages/deb",
        "nightly/signed/packages/deb",
    )
for pkg in deb.get("packages") or []:
    if pkg.get("s3_key"):
        pkg["s3_key"] = _rewrite(
            pkg["s3_key"],
            "nightly/unsigned/packages/deb",
            "nightly/signed/packages/deb",
        )
rpm = signed.get("rpm") or {}
if rpm.get("s3_key"):
    rpm["s3_key"] = _rewrite(
        rpm["s3_key"],
        "nightly/unsigned/packages/rpm",
        "nightly/signed/packages/rpm",
    )
tar = signed.get("tar") or {}
if tar.get("s3_key"):
    tar["s3_key"] = _rewrite(tar["s3_key"], "nightly/unsigned/tarball", "nightly/signed/tarball")
if tar.get("sha256_sidecar_key"):
    tar["sha256_sidecar_key"] = _rewrite(
        tar["sha256_sidecar_key"],
        "nightly/unsigned/tarball",
        "nightly/signed/tarball",
    )
signed["metadata"] = [
    {"s3_key": "nightly/signed/packages/deb/dists/stable/main/binary-amd64/Packages"},
    {"s3_key": "nightly/signed/packages/rpm/x86_64/repodata/primary.xml.gz"},
]
signed_out = Path("${WORKDIR}/signed-latest.json")
signed_out.write_text(json.dumps(signed, indent=2) + "\n", encoding="utf-8")
print(signed_out.read_text(encoding="utf-8"))
PY

aws s3 cp "$WORKDIR/latest.json" "s3://${BUCKET}/nightly/unsigned/latest.json" --no-progress
echo "Published s3://${BUCKET}/nightly/unsigned/latest.json for run ${RUN_ID}"
aws s3 cp "$WORKDIR/signed-latest.json" "s3://${BUCKET}/nightly/signed/latest.json" --no-progress
echo "Published s3://${BUCKET}/nightly/signed/latest.json for run ${RUN_ID}"
