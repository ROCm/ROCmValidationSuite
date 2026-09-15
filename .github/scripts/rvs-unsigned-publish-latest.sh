#!/bin/sh
# Merge per-run unsigned metadata and publish nightly/unsigned/latest.json for signing CI.
# Env: AWS_S3_BUCKET (required), GITHUB_RUN_ID, GITHUB_SHA, ROCM_VERSION (optional)
set -eu

BUCKET="${AWS_S3_BUCKET:-}"
RUN_ID="${GITHUB_RUN_ID:-}"
SHA="${GITHUB_SHA:-}"
ROCM_VERSION="${ROCM_VERSION:-}"

if [ -z "$BUCKET" ] || [ -z "$RUN_ID" ]; then
  echo "::warning::AWS_S3_BUCKET or GITHUB_RUN_ID unset; skipping latest.json publish."
  exit 0
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

DEB_JSON="$WORKDIR/deb.json"
RPM_JSON="$WORKDIR/rpm-tar.json"

if ! aws s3 cp "s3://${BUCKET}/nightly/unsigned/runs/${RUN_ID}/deb.json" "$DEB_JSON" --no-progress 2>/dev/null; then
  echo "::notice::No unsigned DEB metadata for run ${RUN_ID}; skipping latest.json (expected when unsigned steps did not run)."
  exit 0
fi

if ! aws s3 cp "s3://${BUCKET}/nightly/unsigned/runs/${RUN_ID}/rpm-tar.json" "$RPM_JSON" --no-progress 2>/dev/null; then
  echo "::notice::No unsigned RPM/TAR metadata for run ${RUN_ID}; skipping latest.json."
  exit 0
fi

python3 <<PY
import json
import os
from datetime import datetime, timezone
from pathlib import Path

deb = json.loads(Path("${DEB_JSON}").read_text(encoding="utf-8"))
rpm_tar = json.loads(Path("${RPM_JSON}").read_text(encoding="utf-8"))

latest = {
    "github_run_id": "${RUN_ID}",
    "github_sha": "${SHA}",
    "rocm_version": os.environ.get("ROCM_VERSION") or None,
    "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "deb": deb,
    "rpm": rpm_tar.get("rpm"),
    "tar": rpm_tar.get("tar"),
}
out = Path("${WORKDIR}/latest.json")
out.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
print(out.read_text(encoding="utf-8"))
PY

aws s3 cp "$WORKDIR/latest.json" "s3://${BUCKET}/nightly/unsigned/latest.json" --no-progress
echo "Published s3://${BUCKET}/nightly/unsigned/latest.json for run ${RUN_ID}"
