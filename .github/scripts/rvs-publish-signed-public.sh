#!/bin/sh
# Publish a complete signed RVS set to the public extras-rvs buckets, or copy
# one RC build to stable-staging. The public role comes only from AWS_PUB_ARN,
# which must contain the literal token <stream>.
#
# Whole-prefix sync: every run publishes the complete signed prefix
# (<stream>/signed/packages/ and <stream>/signed/tarball/), not only the files
# named in latest.json, so older packages that the signed deb/rpm index still lists
# are always public. latest.json and the run folders are never published. Only the
# difference is transferred: a file is copied when it is missing publicly, has a
# different size, or was modified in the signed prefix after the public copy was
# written (the rule of aws s3 sync). Nothing is ever deleted from any bucket.
#
# Signed content rule: signed files are only READ and copied unchanged with
# "aws s3 cp". The script never edits, recompresses, regenerates or re-signs a
# file. It assumes the signed prefix is the latest complete set and is not modified
# while the publisher runs. The only object it writes in the RVS bucket is
# <stream>/signed/latest.json, to set published=Yes after a successful publish
# (a conditional write, so a newer latest.json is never overwritten). A latest.json
# with published=Yes is skipped unless FORCE_PUBLISH=true. A build is also not
# published until the signed Packages / primary.xml.gz list its deb and rpm.
#
# Usage: rvs-publish-signed-public.sh publish|promote-stable
#   (internal) publish-stream <signed-prefix> <stream>
#
# "publish" runs each stream in its own child process (publish-stream), so errexit
# (set -e) is active inside the stream and a failed step cannot continue with the
# credentials of a previous role.
set -eu

COMPONENT="v5/rocm/extras/rvs"
INTERNAL_BUCKET="${AWS_S3_BUCKET:-}"
INTERNAL_ROLE="${AWS_ROLE_ARN:-}"
PUB_ROLE_TEMPLATE="${AWS_PUB_ARN:-}"
RUN_NUMBER="${RUN_NUMBER:-}"
CMD="${1:-}"

DEB_INDEX_REL="packages/deb/dists/stable/main/binary-amd64/Packages"
RPM_INDEX_REL="packages/rpm/x86_64/repodata/primary.xml.gz"
# Top-level index and signature files. They are uploaded last so a client never
# sees an index that points at files that are not public yet.
INDEX_LAST="packages/deb/dists/stable/Release
packages/deb/dists/stable/InRelease
packages/deb/dists/stable/Release.gpg
packages/rpm/x86_64/repodata/repomd.xml
packages/rpm/x86_64/repodata/repomd.xml.asc"

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

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

# Python helper: strict key validation, manifest parsing and index parsing.
# It only reads its inputs; it never modifies a signed file.
cat > "${TMP_ROOT}/helper.py" <<'PY'
import gzip
import json
import posixpath
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REQUIRED_METADATA = [
    "packages/deb/dists/stable/Release",
    "packages/deb/dists/stable/InRelease",
    "packages/deb/dists/stable/Release.gpg",
    "packages/deb/dists/stable/main/binary-amd64/Packages",
    "packages/deb/dists/stable/main/binary-amd64/Packages.gz",
    "packages/rpm/x86_64/repodata/repomd.xml",
    "packages/rpm/x86_64/repodata/repomd.xml.asc",
    "packages/rpm/x86_64/repodata/primary.xml.gz",
    "packages/rpm/x86_64/repodata/filelists.xml.gz",
    "packages/rpm/x86_64/repodata/other.xml.gz",
]
DEB_INDEX = "packages/deb/dists/stable/main/binary-amd64/Packages"
RPM_INDEX = "packages/rpm/x86_64/repodata/primary.xml.gz"


def fail(message):
    print(f"::error::{message}", file=sys.stderr)
    sys.exit(1)


def valid_key(key, prefixes):
    """A key is valid only if it is already normalized, has no ., .. or empty
    segment, no backslash, no control character and no leading slash, and starts
    with one of the allowed prefixes."""
    if not isinstance(key, str) or not key:
        return False
    if key.startswith("/") or "\\" in key or key != key.strip():
        return False
    if any(ord(ch) < 32 for ch in key):
        return False
    if any(part in ("", ".", "..") for part in key.split("/")):
        return False
    if posixpath.normpath(key) != key:
        return False
    return any(key.startswith(prefix) for prefix in prefixes)


def text(value):
    return value if isinstance(value, str) and value else None


def cmd_collect(latest_path, signed_prefix):
    doc = json.loads(Path(latest_path).read_text(encoding="utf-8"))
    signed_prefix = signed_prefix.rstrip("/")
    if not signed_prefix.endswith("/signed"):
        fail(f"Unexpected signed prefix {signed_prefix}")
    unsigned_prefix = signed_prefix[: -len("signed")] + "unsigned"
    allowed = (f"{signed_prefix}/packages/", f"{signed_prefix}/tarball/")
    keys = []
    bad = []

    def add(value):
        value = text(value)
        if value is None:
            return
        if value.startswith(unsigned_prefix + "/"):
            value = signed_prefix + value[len(unsigned_prefix):]
        if not valid_key(value, allowed):
            bad.append(value)
            return
        if value not in keys:
            keys.append(value)

    deb_packages = [
        text(pkg.get("s3_key"))
        for pkg in ((doc.get("deb") or {}).get("packages") or [])
        if isinstance(pkg, dict)
    ]
    deb_packages = [key for key in deb_packages if key]
    rpm_key = text((doc.get("rpm") or {}).get("s3_key"))
    tar = doc.get("tar") or {}
    tar_key = text(tar.get("s3_key"))
    sidecar_key = text(tar.get("sha256_sidecar_key"))

    missing = []
    if not deb_packages:
        missing.append("deb.packages[].s3_key")
    if not rpm_key:
        missing.append("rpm.s3_key")
    if not tar_key:
        missing.append("tar.s3_key")
    if not sidecar_key:
        missing.append("tar.sha256_sidecar_key")
    if missing:
        fail("latest.json is missing required manifest key(s): " + ", ".join(missing))

    for key in deb_packages:
        add(key)
    add(rpm_key)
    add(tar_key)
    add(sidecar_key)
    for item in doc.get("metadata") or []:
        add(item.get("s3_key") if isinstance(item, dict) else item)
    if bad:
        for value in bad:
            print(
                f"::error::{value!r} is not a valid key under {signed_prefix}/packages/ or {signed_prefix}/tarball/",
                file=sys.stderr,
            )
        sys.exit(1)

    for rel in REQUIRED_METADATA:
        add(f"{signed_prefix}/{rel}")
    for key in keys:
        print(key)


def index_key(base, value):
    value = value.strip()
    if not value or value.startswith("/") or "\\" in value:
        fail(f"Unsafe package path in index: {value!r}")
    norm = posixpath.normpath(value)
    if norm in (".", "..") or norm.startswith("../"):
        fail(f"Unsafe package path in index: {value!r}")
    key = f"{base}/{norm}"
    if not valid_key(key, (base + "/",)):
        fail(f"Unsafe package path in index: {value!r}")
    return key


def cmd_refs(index_dir):
    """Print the package paths (relative to the signed or public root) that the
    deb Packages and rpm primary.xml.gz in index_dir reference."""
    root_dir = Path(index_dir)
    packages_path = root_dir / DEB_INDEX
    try:
        packages_text = packages_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        fail(f"Cannot read {packages_path}: {exc}")
    refs = set()
    deb_found = 0
    for line in packages_text.splitlines():
        if line.startswith("Filename:"):
            refs.add(index_key("packages/deb", line.split(":", 1)[1]))
            deb_found += 1
    if not deb_found:
        fail("Packages lists no Filename entries")

    primary_path = root_dir / RPM_INDEX
    try:
        with gzip.open(primary_path, "rb") as handle:
            xml_root = ET.parse(handle).getroot()
    except (OSError, EOFError, ValueError, ET.ParseError) as exc:
        fail(f"Cannot parse {primary_path}: {exc}")
    rpm_found = 0
    for element in xml_root.iter():
        if element.tag.split("}")[-1] == "location":
            href = element.get("href")
            if href:
                refs.add(index_key("packages/rpm/x86_64", href))
                rpm_found += 1
    if not rpm_found:
        fail("primary.xml.gz lists no package locations")
    for key in sorted(refs):
        print(key)


ELIGIBLE = ("packages/", "tarball/")


def read_docs(data):
    """Parse one or more JSON documents (aws cli may print one per page)."""
    decoder = json.JSONDecoder()
    pos = 0
    items = []
    while pos < len(data):
        while pos < len(data) and data[pos].isspace():
            pos += 1
        if pos >= len(data):
            break
        try:
            obj, pos = decoder.raw_decode(data, pos)
        except ValueError as exc:
            fail(f"Cannot parse bucket listing: {exc}")
        if obj is None:
            continue
        if not isinstance(obj, list):
            fail("Unexpected bucket listing format")
        items.extend(obj)
    return items


def parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def read_tsv(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        rel, size, modified = line.split("\t")
        rows[rel] = (int(size), modified)
    return rows


def cmd_listobjs(prefix, mode):
    """Read an aws s3api list-objects-v2 'Contents[].[Key,Size,LastModified]'
    JSON from stdin and print 'relative-key<TAB>size<TAB>last-modified' for the
    objects under packages/ and tarball/ of <prefix>. Everything else (latest.json,
    run folders, ...) is ignored, so it is never published. strict mode fails on an
    invalid eligible key; lenient mode skips it with a warning."""
    data = sys.stdin.read().strip()
    if not data:
        fail("Empty bucket listing (the list call failed)")
    base = prefix.rstrip("/") + "/"
    rows = {}
    for item in read_docs(data):
        if not (isinstance(item, list) and len(item) == 3):
            fail("Unexpected bucket listing entry")
        key, size, modified = item
        if not isinstance(key, str) or key.endswith("/"):
            continue
        if not key.startswith(base):
            fail(f"Listed key {key!r} is outside {base}")
        rel = key[len(base):]
        if not rel.startswith(ELIGIBLE):
            continue
        if not valid_key(rel, ELIGIBLE):
            if mode == "strict":
                fail(f"Invalid key in bucket listing: {key!r}")
            print(f"::warning::Ignoring unexpected key in bucket listing: {key!r}", file=sys.stderr)
            continue
        rows[rel] = (size, modified)
    for rel in sorted(rows):
        print(f"{rel}\t{rows[rel][0]}\t{rows[rel][1]}")


def cmd_delta(signed_path, public_path):
    """Print the signed keys that must be copied: missing publicly, a different
    size, or modified in the signed prefix after the public copy was written
    (the same rule as aws s3 sync)."""
    signed = read_tsv(signed_path)
    public = read_tsv(public_path)
    for rel in sorted(signed):
        size, modified = signed[rel]
        other = public.get(rel)
        if other is None or other[0] != size:
            print(rel)
            continue
        signed_time = parse_time(modified)
        public_time = parse_time(other[1])
        try:
            newer = signed_time is None or public_time is None or signed_time > public_time
        except TypeError:
            newer = True
        if newer:
            print(rel)


def cmd_missing(refs_path, *listing_paths):
    """Print the referenced keys that are in none of the listings."""
    available = set()
    for path in listing_paths:
        available.update(read_tsv(path))
    for line in Path(refs_path).read_text(encoding="utf-8").splitlines():
        key = line.strip()
        if key and key not in available:
            print(key)


def cmd_check_copied(delta_path, signed_path, public_path):
    """Every copied key must now be public with the signed size."""
    signed = read_tsv(signed_path)
    public = read_tsv(public_path)
    bad = 0
    for line in Path(delta_path).read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        other = public.get(rel)
        if other is None or other[0] != signed[rel][0]:
            print(f"::error::Public object missing or different size after copy: {rel}", file=sys.stderr)
            bad += 1
    if bad:
        sys.exit(1)


def load_manifest(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        fail("latest.json is not a JSON object")
    return doc


def cmd_flag_get(path):
    doc = load_manifest(path)
    print("Yes" if str(doc.get("published", "")).strip().lower() == "yes" else "No")


def cmd_flag_set(src, dst, run_id, stamp):
    """Copy latest.json with the published markers set. Every other key keeps
    its value."""
    doc = load_manifest(src)
    doc["published"] = "Yes"
    doc["public_published_at"] = stamp
    doc["public_publish_run_id"] = run_id
    Path(dst).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def cmd_index_missing(latest_path, refs_path):
    """Print the deb and rpm packages named in latest.json that the signed
    Packages / primary.xml.gz (the refs file) do not list yet. Compared by file
    name, which is unique per build, so a different pool folder layout cannot
    cause a permanent mismatch. No output means every package is listed."""
    doc = load_manifest(latest_path)
    keys = [
        text(pkg.get("s3_key"))
        for pkg in ((doc.get("deb") or {}).get("packages") or [])
        if isinstance(pkg, dict)
    ]
    keys.append(text((doc.get("rpm") or {}).get("s3_key")))
    listed = {
        line.strip().rsplit("/", 1)[-1]
        for line in Path(refs_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    seen = set()
    for key in keys:
        if not key:
            continue
        name = key.rsplit("/", 1)[-1]
        if name not in listed and name not in seen:
            seen.add(name)
            print(name)


def cmd_validate(required_prefix):
    for line in sys.stdin.read().splitlines():
        key = line.strip()
        if not key:
            continue
        if not valid_key(key, (required_prefix,)):
            fail(f"Invalid key {key!r} (must be normalized and under {required_prefix})")
        print(key)


def main():
    args = sys.argv[1:]
    if not args:
        fail("helper: missing command")
    command, rest = args[0], args[1:]
    if command == "collect" and len(rest) == 2:
        cmd_collect(*rest)
    elif command == "refs" and len(rest) == 1:
        cmd_refs(*rest)
    elif command == "listobjs" and len(rest) == 2:
        cmd_listobjs(*rest)
    elif command == "delta" and len(rest) == 2:
        cmd_delta(*rest)
    elif command == "missing" and len(rest) >= 2:
        cmd_missing(*rest)
    elif command == "check-copied" and len(rest) == 3:
        cmd_check_copied(*rest)
    elif command == "index-missing" and len(rest) == 2:
        cmd_index_missing(*rest)
    elif command == "flag-get" and len(rest) == 1:
        cmd_flag_get(*rest)
    elif command == "flag-set" and len(rest) == 4:
        cmd_flag_set(*rest)
    elif command == "validate" and len(rest) == 1:
        cmd_validate(*rest)
    else:
        fail(f"helper: bad usage: {' '.join(args)}")


main()
PY

pyhelper() {
  python3 "${TMP_ROOT}/helper.py" "$@"
}

public_bucket() {
  echo "therock-repo-amd-${1}-extras-rvs"
}

public_role() {
  printf '%s\n' "$PUB_ROLE_TEMPLATE" | sed "s/<stream>/${1}/g"
}

# Assume a role through GitHub OIDC. Any previous credentials are removed first
# and every step is checked, so a failure never leaves a previous role active.
assume_role() {
  role_arn="$1"
  session="$2"
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
  if ! token_json=$(curl -fsS -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
    "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=sts.amazonaws.com"); then
    echo "::error::Could not request the GitHub OIDC token for ${session}." >&2
    return 1
  fi
  if ! token=$(printf '%s' "$token_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['value'])"); then
    echo "::error::Could not read the GitHub OIDC token for ${session}." >&2
    return 1
  fi
  if ! creds=$(aws sts assume-role-with-web-identity \
    --role-arn "$role_arn" \
    --role-session-name "$session" \
    --web-identity-token "$token" \
    --query 'Credentials' --output json); then
    echo "::error::Could not assume the role for ${session}." >&2
    return 1
  fi
  if ! parsed=$(printf '%s' "$creds" | python3 -c "import sys,json; c=json.load(sys.stdin); print(c['AccessKeyId']); print(c['SecretAccessKey']); print(c['SessionToken'])"); then
    echo "::error::Could not parse the credentials for ${session}." >&2
    return 1
  fi
  access_key=$(printf '%s\n' "$parsed" | sed -n 1p)
  secret_key=$(printf '%s\n' "$parsed" | sed -n 2p)
  session_token=$(printf '%s\n' "$parsed" | sed -n 3p)
  if [ -z "$access_key" ] || [ -z "$secret_key" ] || [ -z "$session_token" ]; then
    echo "::error::Incomplete credentials for ${session}." >&2
    return 1
  fi
  AWS_ACCESS_KEY_ID="$access_key"
  AWS_SECRET_ACCESS_KEY="$secret_key"
  AWS_SESSION_TOKEN="$session_token"
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
  export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
  unset creds token token_json parsed access_key secret_key session_token role_arn
  return 0
}

object_exists() {
  aws s3api head-object --bucket "$1" --key "$2" >/dev/null 2>&1
}

# List the eligible objects as "relative-key<TAB>size<TAB>last-modified". Two
# separate list requests are made, one for <list-root>/packages/ and one for
# <list-root>/tarball/, so latest.json, run folders and any other path under the
# root are never requested (and the role only needs ListBucket on those two
# prefixes). Each request is checked on its own; the helper still applies the
# packages/ and tarball/ allowlist and the strict key check.
# Usage: list_objects <bucket> <list-root> strict|lenient
list_objects() {
  lo_tmp="${TMP_ROOT}/list-objects.$$"
  : > "$lo_tmp"
  for lo_dir in packages tarball; do
    if ! aws s3api list-objects-v2 --bucket "$1" --prefix "${2}/${lo_dir}/" \
      --query 'Contents[].[Key,Size,LastModified]' --output json \
      | pyhelper listobjs "$2" "$3" >> "$lo_tmp"; then
      rm -f "$lo_tmp"
      return 1
    fi
  done
  cat "$lo_tmp"
  rm -f "$lo_tmp"
}

# Download the deb Packages and rpm primary.xml.gz index files from
# s3://<bucket>/<base>/ into <dir>. Returns non-zero if either is unavailable.
fetch_index() {
  fi_bucket="$1"
  fi_base="$2"
  fi_dir="$3"
  mkdir -p "${fi_dir}/$(dirname "$DEB_INDEX_REL")" "${fi_dir}/$(dirname "$RPM_INDEX_REL")"
  aws s3 cp "s3://${fi_bucket}/${fi_base}/${DEB_INDEX_REL}" "${fi_dir}/${DEB_INDEX_REL}" \
    --no-progress >/dev/null 2>&1 || return 1
  aws s3 cp "s3://${fi_bucket}/${fi_base}/${RPM_INDEX_REL}" "${fi_dir}/${RPM_INDEX_REL}" \
    --no-progress >/dev/null 2>&1 || return 1
  return 0
}

# Print the lines of <file> with a short label, for the job log.
log_list() {
  label="$1"
  list_file="$2"
  while IFS= read -r item; do
    [ -n "$item" ] || continue
    echo "${label}: ${item}"
  done < "$list_file"
}

# True if <key> is one of the top-level index / signature files.
is_index_last() {
  printf '%s\n' "$INDEX_LAST" | grep -Fxq -- "$1"
}

# Upload the files named in <list> from the stage directory (public role).
# Uses: stage, dest_bucket.
upload_list() {
  ul_list="$1"
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    if ! aws s3 cp "${stage}/${rel}" "s3://${dest_bucket}/${COMPONENT}/${rel}" --no-progress; then
      echo "::error::Public copy failed: s3://${dest_bucket}/${COMPONENT}/${rel}" >&2
      return 1
    fi
  done < "$ul_list"
}

# Set published=Yes in the signed latest.json (RVS role). The write is conditional
# on the ETag read at the start of the run, so a new build that replaced
# latest.json meanwhile is never overwritten: it stays published=No and is
# picked up by the next run. Uses: latest_json, latest_etag, work, prefix, stream.
mark_published() {
  assume_role "$INTERNAL_ROLE" "rvs-signed-mark-${stream}" || return 1
  stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  marked_json="${work}/latest-published.json"
  pyhelper flag-set "$latest_json" "$marked_json" "${GITHUB_RUN_ID:-}" "$stamp" || return 1
  if put_out=$(aws s3api put-object --bucket "$INTERNAL_BUCKET" \
    --key "${prefix}/latest.json" --body "$marked_json" \
    --content-type application/json --if-match "$latest_etag" 2>&1); then
    echo "Marked s3://${INTERNAL_BUCKET}/${prefix}/latest.json as published=Yes."
    return 0
  fi
  case "$put_out" in
    *PreconditionFailed*|*"(412)"*)
      echo "::notice::${prefix}/latest.json changed during publish; leaving published=No so the new build is picked up by the next run."
      return 0
      ;;
  esac
  echo "::error::Could not mark s3://${INTERNAL_BUCKET}/${prefix}/latest.json as published (the files were published): ${put_out}" >&2
  return 1
}

publish_stream() {
  prefix="$1"
  stream="$2"
  work="$3"
  mkdir -p "$work"
  echo "Checking ${prefix} for stream ${stream}"

  dest_bucket=$(public_bucket "$stream")
  latest_json="${work}/latest.json"
  keys_file="${work}/keys.txt"
  signed_index="${work}/signed-index"
  public_index="${work}/public-index"
  stage="${work}/stage"

  # RVS role: read the manifest and its ETag.
  assume_role "$INTERNAL_ROLE" "rvs-signed-read-${stream}" || return 1
  if ! latest_etag=$(aws s3api get-object --bucket "$INTERNAL_BUCKET" \
    --key "${prefix}/latest.json" --query ETag --output text "$latest_json"); then
    echo "::warning::Missing s3://${INTERNAL_BUCKET}/${prefix}/latest.json; skipping ${stream}."
    return 0
  fi

  # Gate 1: the signed latest.json is already marked as published.
  published=$(pyhelper flag-get "$latest_json") || return 1
  if [ "$published" = "Yes" ]; then
    if [ "${FORCE_PUBLISH:-false}" = "true" ]; then
      echo "latest.json is marked published=Yes, but force is set; running the full sync."
    else
      echo "::notice::${stream}: the latest signed build in ${prefix} is already published (published=Yes in latest.json). Skipping."
      return 0
    fi
  fi

  # Gate 2: every file the manifest names (packages, tarball, sidecar and the
  # repo metadata) must already be in the signed prefix.
  if ! pyhelper collect "$latest_json" "$prefix" > "$keys_file"; then
    return 1
  fi
  while IFS= read -r key; do
    [ -n "$key" ] || continue
    if ! object_exists "$INTERNAL_BUCKET" "$key"; then
      echo "::warning::Signed object s3://${INTERNAL_BUCKET}/${key} is not present yet (signing job has not produced it); skipping ${stream}."
      return 0
    fi
  done < "$keys_file"

  # RVS role: the signed index (also used by gate 3) and the list of signed files.
  if ! fetch_index "$INTERNAL_BUCKET" "$prefix" "$signed_index"; then
    echo "::error::Could not read the signed Packages / primary.xml.gz under ${prefix}." >&2
    return 1
  fi
  if ! pyhelper refs "$signed_index" > "${work}/signed-refs.raw"; then
    return 1
  fi
  LC_ALL=C sort -u "${work}/signed-refs.raw" > "${work}/signed-refs.txt"

  # Gate 3: the signed Packages and primary.xml.gz must list the deb and rpm
  # packages of this build. The metadata files keep the same names in every
  # build, so they can still describe the previous build while the packages are
  # already in the signed prefix (the signing job has not finished yet).
  pyhelper index-missing "$latest_json" "${work}/signed-refs.txt" > "${work}/index-missing.txt" || return 1
  if [ -s "${work}/index-missing.txt" ]; then
    while IFS= read -r name; do
      [ -n "$name" ] || continue
      echo "::warning::Signed index does not list ${name} yet (signing job has not updated the index); skipping ${stream}."
    done < "${work}/index-missing.txt"
    return 0
  fi

  if ! list_objects "$INTERNAL_BUCKET" "$prefix" strict > "${work}/signed.tsv"; then
    return 1
  fi

  # Public role: the files that are already public.
  assume_role "$(public_role "$stream")" "rvs-signed-check-${stream}" || return 1
  if ! list_objects "$dest_bucket" "$COMPONENT" lenient > "${work}/public.tsv"; then
    return 1
  fi

  # Every package the signed index references must be in the signed prefix or
  # already public, otherwise the public repository cannot be made complete.
  pyhelper missing "${work}/signed-refs.txt" "${work}/signed.tsv" "${work}/public.tsv" \
    > "${work}/missing-refs.txt" || return 1
  if [ -s "${work}/missing-refs.txt" ]; then
    log_list "::error::Referenced package is missing from both the signed set and the public bucket" \
      "${work}/missing-refs.txt" >&2
    return 1
  fi

  # Delta: signed files that are missing publicly, have a different size, or were
  # modified in the signed prefix after the public copy was written.
  pyhelper delta "${work}/signed.tsv" "${work}/public.tsv" > "${work}/delta.txt" || return 1
  if [ ! -s "${work}/delta.txt" ]; then
    echo "::notice::No new or changed files for ${stream}; everything in ${prefix} is already public in s3://${dest_bucket}/${COMPONENT}/."
  else
    delta_count=$(wc -l < "${work}/delta.txt" | tr -d ' ')
    echo "${delta_count} file(s) to publish for ${stream}."
    log_list "To publish" "${work}/delta.txt"

    # Split the delta into upload passes. The top-level index and signature
    # files go last so a client never sees an index that points at files that
    # are not public yet.
    : > "${work}/pass1.txt"
    : > "${work}/pass2.txt"
    : > "${work}/pass3.txt"
    while IFS= read -r rel; do
      [ -n "$rel" ] || continue
      if is_index_last "$rel"; then
        echo "$rel" >> "${work}/pass3.txt"
        continue
      fi
      case "$rel" in
        packages/deb/dists/*|packages/rpm/x86_64/repodata/*) echo "$rel" >> "${work}/pass2.txt" ;;
        *) echo "$rel" >> "${work}/pass1.txt" ;;
      esac
    done < "${work}/delta.txt"

    # RVS role: download the delta unchanged.
    assume_role "$INTERNAL_ROLE" "rvs-signed-read2-${stream}" || return 1
    while IFS= read -r rel; do
      [ -n "$rel" ] || continue
      mkdir -p "$(dirname "${stage}/${rel}")"
      if ! aws s3 cp "s3://${INTERNAL_BUCKET}/${prefix}/${rel}" "${stage}/${rel}" --no-progress; then
        echo "::error::Could not download s3://${INTERNAL_BUCKET}/${prefix}/${rel}" >&2
        return 1
      fi
    done < "${work}/delta.txt"

    # Public role: 1. packages, rpm, tarball and sidecar, 2. repo files, 3. the
    # top-level index and signature files.
    assume_role "$(public_role "$stream")" "rvs-signed-pub-${stream}" || return 1
    upload_list "${work}/pass1.txt" || return 1
    upload_list "${work}/pass2.txt" || return 1
    upload_list "${work}/pass3.txt" || return 1

    # Every copied file is public with the signed size.
    if ! list_objects "$dest_bucket" "$COMPONENT" lenient > "${work}/public.tsv"; then
      return 1
    fi
    pyhelper check-copied "${work}/delta.txt" "${work}/signed.tsv" "${work}/public.tsv" || return 1
  fi

  # The public index must only reference packages that are public. This reads the
  # public Packages and primary.xml.gz, so the public role needs s3:GetObject.
  if ! fetch_index "$dest_bucket" "$COMPONENT" "$public_index"; then
    echo "::error::Could not read the public Packages / primary.xml.gz in s3://${dest_bucket}/${COMPONENT}/." >&2
    return 1
  fi
  if ! pyhelper refs "$public_index" > "${work}/public-refs.txt"; then
    return 1
  fi
  pyhelper missing "${work}/public-refs.txt" "${work}/public.tsv" > "${work}/public-missing-refs.txt" || return 1
  if [ -s "${work}/public-missing-refs.txt" ]; then
    log_list "::error::Package referenced by the public index is not public" \
      "${work}/public-missing-refs.txt" >&2
    return 1
  fi

  # Everything is public: mark the signed build as published.
  mark_published || return 1
  echo "Published ${stream} from ${prefix}."
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
  work="${TMP_ROOT}/promote"
  mkdir -p "$work"
  assume_role "$(public_role rc)" "rvs-rc-read" || exit 1
  : > "${work}/selected.txt"
  copy_matched() {
    kind="$1"
    pattern="$2"
    aws s3 ls "s3://${rc_bucket}/${COMPONENT}/${pattern}" --recursive \
      | awk '{print $NF}' > "${work}/list.raw" || true
    # Every listed key must be a normalized key under the component.
    if ! pyhelper validate "${COMPONENT}/" < "${work}/list.raw" > "${work}/list.txt"; then
      exit 1
    fi
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

  assume_role "$(public_role rc)" "rvs-rc-read-files" || exit 1
  while IFS= read -r key; do
    [ -n "$key" ] || continue
    rel=${key#"${COMPONENT}/"}
    dest="${work}/files/${rel}"
    mkdir -p "$(dirname "$dest")"
    aws s3 cp "s3://${rc_bucket}/${key}" "$dest" --no-progress
  done < "${work}/selected.txt"

  assume_role "$(public_role stable-staging)" "rvs-stable-staging-write" || exit 1
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

require_internal() {
  if [ -z "$INTERNAL_BUCKET" ] || [ -z "$INTERNAL_ROLE" ]; then
    echo "::error::AWS_S3_BUCKET and AWS_ROLE_ARN are required." >&2
    exit 1
  fi
}

case "$CMD" in
  publish)
    require_internal
    failed=0
    sh "$0" publish-stream "nightly/signed" "nightly" || failed=1
    sh "$0" publish-stream "release/signed" "rc" || failed=1
    exit "$failed"
    ;;
  publish-stream)
    require_internal
    stream_prefix="${2:-}"
    stream_name="${3:-}"
    case "$stream_name" in
      nightly|rc) ;;
      *)
        echo "::error::Unknown stream '${stream_name}'." >&2
        exit 1
        ;;
    esac
    case "$stream_prefix" in
      "nightly/signed"|"release/signed") ;;
      *)
        echo "::error::Unexpected signed prefix '${stream_prefix}'." >&2
        exit 1
        ;;
    esac
    publish_stream "$stream_prefix" "$stream_name" "${TMP_ROOT}/${stream_name}"
    ;;
  promote-stable)
    promote_stable
    ;;
  *)
    echo "Usage: $0 publish|promote-stable" >&2
    exit 1
    ;;
esac
