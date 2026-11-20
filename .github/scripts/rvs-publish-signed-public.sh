#!/bin/sh
# Publish a complete signed RVS set to the public extras-rvs buckets, or copy
# one RC build to stable-staging. The public role comes only from AWS_PUB_ARN,
# which must contain the literal token <stream>.
# Usage: rvs-publish-signed-public.sh publish|promote-stable
set -eu

COMPONENT="v5/rocm/extras/rvs"
INTERNAL_BUCKET="${AWS_S3_BUCKET:-}"
INTERNAL_ROLE="${AWS_ROLE_ARN:-}"
PUB_ROLE_TEMPLATE="${AWS_PUB_ARN:-}"
RUN_NUMBER="${RUN_NUMBER:-}"
CMD="${1:-}"

case "$PUB_ROLE_TEMPLATE" in
  *"<stream>"*) ;;
  *)
    echo "::error::AWS_PUB_ARN must contain the <stream> token." >&2
    exit 1
    ;;
esac
if [ -z "${ACTIONS_ID_TOKEN_REQUEST_TOKEN:-}" ] || [ -z "${ACTIONS_ID_TOKEN_REQUEST_URL:-}" ]; then
  echo "::error::GitHub OIDC token request is not available." >&2
  exit 1
fi

public_bucket() {
  echo "therock-repo-amd-${1}-extras-rvs"
}

public_role() {
  printf '%s\n' "$PUB_ROLE_TEMPLATE" | sed "s/<stream>/${1}/g"
}

assume_role() {
  role_arn="$1"
  session="$2"
  token=$(curl -fsS -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
    "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=sts.amazonaws.com" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['value'])")
  creds=$(aws sts assume-role-with-web-identity \
    --role-arn "$role_arn" \
    --role-session-name "$session" \
    --web-identity-token "$token" \
    --query 'Credentials' --output json)
  AWS_ACCESS_KEY_ID=$(printf '%s' "$creds" | python3 -c "import sys,json; print(json.load(sys.stdin)['AccessKeyId'])")
  AWS_SECRET_ACCESS_KEY=$(printf '%s' "$creds" | python3 -c "import sys,json; print(json.load(sys.stdin)['SecretAccessKey'])")
  AWS_SESSION_TOKEN=$(printf '%s' "$creds" | python3 -c "import sys,json; print(json.load(sys.stdin)['SessionToken'])")
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
  export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
  unset creds token role_arn
}

object_exists() {
  aws s3api head-object --bucket "$1" --key "$2" >/dev/null 2>&1
}

collect_keys() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
keys = []

def add(value):
    if isinstance(value, str) and value and value not in keys:
        keys.append(value)

deb = doc.get("deb") or {}
for pkg in deb.get("packages") or []:
    add(pkg.get("s3_key"))
add((doc.get("rpm") or {}).get("s3_key"))
tar = doc.get("tar") or {}
add(tar.get("s3_key"))
add(tar.get("sha256_sidecar_key"))
for item in doc.get("metadata") or []:
    if isinstance(item, dict):
        add(item.get("s3_key"))
    else:
        add(item)
if not keys:
    print("::error::latest.json has no s3 keys", file=sys.stderr)
    sys.exit(1)
for key in keys:
    print(key)
PY
}

publish_stream() {
  prefix="$1"
  stream="$2"
  work="$3/${stream}"
  mkdir -p "$work"
  echo "Checking ${prefix} for stream ${stream}"
  assume_role "$INTERNAL_ROLE" "rvs-signed-read-${stream}"
  if ! aws s3 cp "s3://${INTERNAL_BUCKET}/${prefix}/latest.json" "${work}/latest.json" --no-progress; then
    echo "::warning::Missing s3://${INTERNAL_BUCKET}/${prefix}/latest.json; skipping ${stream}."
    return 0
  fi
  keys_file="${work}/keys.txt"
  collect_keys "${work}/latest.json" > "$keys_file"
  while IFS= read -r key; do
    [ -n "$key" ] || continue
    if ! object_exists "$INTERNAL_BUCKET" "$key"; then
      echo "::warning::Missing s3://${INTERNAL_BUCKET}/${key}; skipping ${stream}."
      return 0
    fi
  done < "$keys_file"

  stage="${work}/stage"
  while IFS= read -r key; do
    [ -n "$key" ] || continue
    case "$key" in
      "${prefix}/"*) rel=${key#"${prefix}/"} ;;
      *)
        echo "::error::${key} is outside ${prefix}" >&2
        return 1
        ;;
    esac
    dest="${stage}/${rel}"
    mkdir -p "$(dirname "$dest")"
    aws s3 cp "s3://${INTERNAL_BUCKET}/${key}" "$dest" --no-progress
  done < "$keys_file"

  dest_bucket=$(public_bucket "$stream")
  assume_role "$(public_role "$stream")" "rvs-signed-pub-${stream}"
  while IFS= read -r key; do
    [ -n "$key" ] || continue
    rel=${key#"${prefix}/"}
    dest_key="${COMPONENT}/${rel}"
    aws s3 cp "${stage}/${rel}" "s3://${dest_bucket}/${dest_key}" --no-progress
    if ! object_exists "$dest_bucket" "$dest_key"; then
      echo "::error::Public object missing after copy: s3://${dest_bucket}/${dest_key}" >&2
      return 1
    fi
  done < "$keys_file"
  echo "Published ${stream} from ${prefix}"
}

match_name() {
  fname="$1"
  kind="$2"
  case "$kind" in
    deb) case "$fname" in *"-${RUN_NUMBER}_"*) return 0 ;; esac ;;
    rpm) case "$fname" in *"-${RUN_NUMBER}."*) return 0 ;; esac ;;
    tar) case "$fname" in *"-${RUN_NUMBER}-Linux"*) return 0 ;; esac ;;
  esac
  return 1
}

promote_stable() {
  if [ -z "$RUN_NUMBER" ]; then
    echo "::error::run_number is required to copy RC to stable-staging." >&2
    exit 1
  fi
  rc_bucket=$(public_bucket rc)
  staging_bucket=$(public_bucket stable-staging)
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
  assume_role "$(public_role rc)" "rvs-rc-read"
  : > "${work}/selected.txt"
  copy_matched() {
    kind="$1"
    pattern="$2"
    aws s3 ls "s3://${rc_bucket}/${COMPONENT}/${pattern}" --recursive \
      | awk '{print $NF}' > "${work}/list.txt" || true
    found=0
    while IFS= read -r key; do
      [ -n "$key" ] || continue
      fname=$(basename "$key")
      case "$fname" in
        *.tar.gz.sha256) [ "$kind" = "sha" ] || continue ;;
        *.tar.gz) [ "$kind" = "tar" ] || continue ;;
        *.deb) [ "$kind" = "deb" ] || continue ;;
        *.rpm) [ "$kind" = "rpm" ] || continue ;;
        *) continue ;;
      esac
      check_kind="$kind"
      [ "$kind" = "sha" ] && check_kind="tar"
      match_name "$fname" "$check_kind" || continue
      echo "$key" >> "${work}/selected.txt"
      found=$((found + 1))
    done < "${work}/list.txt"
    if [ "$kind" != "sha" ] && [ "$found" -ne 1 ]; then
      echo "::error::Expected one ${kind} for run number ${RUN_NUMBER} in s3://${rc_bucket}/${COMPONENT}/${pattern}; found ${found}." >&2
      exit 1
    fi
    if [ "$kind" = "sha" ] && [ "$found" -lt 1 ]; then
      echo "::error::No sha256 sidecar for run number ${RUN_NUMBER}." >&2
      exit 1
    fi
  }
  copy_matched deb "packages/deb/"
  copy_matched rpm "packages/rpm/"
  copy_matched tar "tarball/"
  copy_matched sha "tarball/"

  mkdir -p "${work}/dists" "${work}/repodata"
  aws s3 sync "s3://${rc_bucket}/${COMPONENT}/packages/deb/dists/" "${work}/dists/" --no-progress
  aws s3 sync "s3://${rc_bucket}/${COMPONENT}/packages/rpm/" "${work}/repodata/" \
    --exclude "*" --include "*/repodata/*" --no-progress

  assume_role "$(public_role rc)" "rvs-rc-read-files"
  while IFS= read -r key; do
    [ -n "$key" ] || continue
    rel=${key#"${COMPONENT}/"}
    dest="${work}/files/${rel}"
    mkdir -p "$(dirname "$dest")"
    aws s3 cp "s3://${rc_bucket}/${key}" "$dest" --no-progress
  done < "${work}/selected.txt"

  assume_role "$(public_role stable-staging)" "rvs-stable-staging-write"
  while IFS= read -r key; do
    [ -n "$key" ] || continue
    rel=${key#"${COMPONENT}/"}
    aws s3 cp "${work}/files/${rel}" "s3://${staging_bucket}/${key}" --no-progress
    if ! object_exists "$staging_bucket" "$key"; then
      echo "::error::stable-staging object missing after copy: ${key}" >&2
      exit 1
    fi
  done < "${work}/selected.txt"
  if [ -d "${work}/dists" ] && [ -n "$(find "${work}/dists" -type f -print -quit)" ]; then
    aws s3 sync "${work}/dists/" "s3://${staging_bucket}/${COMPONENT}/packages/deb/dists/" --no-progress
  fi
  if [ -d "${work}/repodata" ] && [ -n "$(find "${work}/repodata" -type f -print -quit)" ]; then
    aws s3 sync "${work}/repodata/" "s3://${staging_bucket}/${COMPONENT}/packages/rpm/" --no-progress
  fi
  echo "Copied run ${RUN_NUMBER} from public RC to stable-staging"
}

case "$CMD" in
  publish)
    if [ -z "$INTERNAL_BUCKET" ] || [ -z "$INTERNAL_ROLE" ]; then
      echo "::error::AWS_S3_BUCKET and AWS_ROLE_ARN are required." >&2
      exit 1
    fi
    root=$(mktemp -d)
    trap 'rm -rf "$root"' EXIT
    failed=0
    publish_stream "nightly/signed" "nightly" "$root" || failed=1
    publish_stream "release/signed" "rc" "$root" || failed=1
    exit "$failed"
    ;;
  promote-stable)
    promote_stable
    ;;
  *)
    echo "Usage: $0 publish|promote-stable" >&2
    exit 1
    ;;
esac
