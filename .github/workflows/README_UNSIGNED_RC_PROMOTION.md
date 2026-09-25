# Unsigned Release Candidate Promotion

This document describes [`.github/workflows/unsigned-release-candidate-promotion.yml`](./unsigned-release-candidate-promotion.yml), which copies RVS packages for a specific build number from the **signed release source** (`release/rvs/`) into the **unsigned staging area** (`release/unsigned/`) and regenerates the APT/YUM repository metadata there. The resulting layout mirrors `nightly/unsigned/` so that the same signing CI can consume either path without changes.

## Purpose

When a release branch build completes, its packages land under `release/rvs/{deb,rpm,tar}/`. Before those packages can be signed and published to consumers, they must be promoted into `release/unsigned/` — the staging prefix that signing CI monitors. This workflow performs that promotion on demand, filtered by the build number encoded in each package filename.

## Trigger

The workflow runs only on **manual dispatch** (`workflow_dispatch`). There is no scheduled or automatic trigger — promotion is always an explicit human action.

### Input

| Input | Required | Description |
|-------|----------|-------------|
| `build_number` | **Yes** | Substring to match against package filenames. Every `.deb`, `.rpm`, and `.tar.gz` whose filename contains this string is promoted. The build number is embedded in the release string (e.g. `r0711` in `amdrocm7-rvs-1.3.15-r0711.20260423.x86_64.rpm`). |

**Matching is simple substring:** if the filename contains the exact string you enter, it is included. For maximum precision use the full release segment (e.g. `r0711.20260423`). For a broader match, use just the numeric part (e.g. `0711`).

## S3 layout

```
s3://<bucket>/
├── release/rvs/           ← source (written by build-relocatable-packages.yml)
│   ├── deb/
│   │   └── amdrocm*-rvs*.deb
│   ├── rpm/
│   │   └── amdrocm*-rvs*.rpm
│   └── tar/
│       └── amdrocm*-rvs*.tar.gz
│
└── release/unsigned/      ← destination (written by this workflow)
    ├── deb/
    │   ├── conf/          # reprepro state (internal; not for apt clients)
    │   ├── pool/main/…/amdrocm*-rvs*.deb
    │   └── dists/stable/
    │       ├── Release
    │       └── main/binary-amd64/Packages(.gz)
    ├── rpm/
    │   └── x86_64/
    │       ├── amdrocm*-rvs*.rpm
    │       └── repodata/
    │           ├── repomd.xml
    │           ├── primary.xml.gz
    │           ├── filelists.xml.gz
    │           └── other.xml.gz
    ├── tar/
    │   ├── amdrocm*-rvs*.tar.gz
    │   └── amdrocm*-rvs*.tar.gz.sha256
    └── latest.json        # signing CI entry point
```

The source paths (`release/rvs/`) are populated by `build-relocatable-packages.yml` when a `release/**` branch is pushed or manually dispatched. This workflow reads from those paths and writes to `release/unsigned/`.

## What the workflow does

The single job (`promote-release-unsigned`) runs these steps in order:

### 1. Install AWS CLI

Same pattern as `build-relocatable-packages.yml`: uses `pip install awscli` with a `--break-system-packages` fallback so it works on both Ubuntu 22.04 and Ubuntu 24.04 hosted runners.

### 2. Configure AWS credentials

Uses OIDC (`assume-role-with-web-identity`) with `secrets.AWS_ROLE_ARN`. No long-term access keys are stored. Credentials are masked in logs.

### 3. Validate inputs

Fails fast if `build_number` or `AWS_S3_BUCKET` are empty. Prints the source and destination S3 prefixes to the log.

### 4. Install packaging tools

Installs `reprepro`, `dpkg-dev`, and `createrepo-c` (or `createrepo`) via `apt-get`. These are needed to maintain the APT archive structure and RPM repodata.

### 5. Copy DEB packages and rebuild APT archive

- Downloads the existing `release/unsigned/deb/{conf,pool,dists}/` from S3 into a staging directory (**accumulate mode** — no `--delete`)
- Bootstraps a `reprepro` distributions config (`Suite: stable`, `Codename: stable`) if the archive does not yet exist
- Lists all keys under `release/rvs/deb/` and downloads those matching `amdrocm*-rvs*.deb` and containing `build_number`
- For each matching `.deb`, runs `reprepro includedeb stable` — idempotently removing the same Package+Version from the local archive first if it is already present, so S3 `PutObject` can overwrite the pool object without requiring `s3:DeleteObject`
- Syncs `conf/`, `pool/`, and `dists/` back to S3 (no `--delete`)

### 6. Copy RPM packages and rebuild YUM repodata

- Downloads existing `release/unsigned/rpm/x86_64/` RPM files from S3, excluding `repodata/` (which will be fully regenerated)
- Downloads matching `.rpm` files from `release/rvs/rpm/` into a local `x86_64/` subdirectory
- Runs `createrepo_c` (fallback: `createrepo`) on that `x86_64/` directory to regenerate `repodata/` inside it
- Syncs `x86_64/` back to `s3://<bucket>/release/unsigned/rpm/x86_64/`

RPMs are placed under `x86_64/` so that the signing CI and yum/dnf clients can use `release/unsigned/rpm/x86_64/` as the `baseurl` directly.

### 7. Copy TAR packages

- Downloads matching `.tar.gz` files and any existing `.sha256` sidecars from `release/rvs/tar/`
- Generates a SHA-256 sidecar (`sha256sum`) for any `.tar.gz` that does not already have one — `release/rvs/tar/` does not always include sidecars, but `release/unsigned/tar/` requires them for signing CI
- Copies both `.tar.gz` and `.tar.gz.sha256` files to `release/unsigned/tar/`

### 8. Publish `release/unsigned/latest.json`

Reads back the current state of `release/unsigned/` to extract metadata for the promoted build number:

- Fetches `release/unsigned/deb/dists/stable/main/binary-amd64/Packages` and parses it to find DEB pool paths whose filename contains `build_number`
- Lists `release/unsigned/rpm/` and filters for RPM filenames containing `build_number`
- Lists `release/unsigned/tar/` and filters for TAR filenames containing `build_number`
- Writes and uploads `release/unsigned/latest.json` with the same schema used by `nightly/unsigned/latest.json`

The `latest.json` schema:

```json
{
  "github_run_id": "12345678",
  "github_sha": "abc123...",
  "build_number": "r0711.20260423",
  "published_at": "2026-04-23T12:34:56Z",
  "deb": {
    "github_run_id": "12345678",
    "deb_prefix": "release/unsigned/deb",
    "packages": [
      {
        "filename": "amdrocm7-rvs_1.3.15-r0711.20260423_amd64.deb",
        "pool_key": "pool/main/a/amdrocm7-rvs/amdrocm7-rvs_1.3.15-r0711.20260423_amd64.deb",
        "s3_key": "release/unsigned/deb/pool/main/a/amdrocm7-rvs/amdrocm7-rvs_1.3.15-r0711.20260423_amd64.deb"
      }
    ]
  },
  "rpm": {
    "filename": "amdrocm7-rvs-1.3.15-r0711.20260423.x86_64.rpm",
    "s3_key": "release/unsigned/rpm/x86_64/amdrocm7-rvs-1.3.15-r0711.20260423.x86_64.rpm"
  },
  "tar": {
    "filename": "amdrocm7-rvs-1.3.15-r0711.20260423-Linux.tar.gz",
    "s3_key": "release/unsigned/tar/amdrocm7-rvs-1.3.15-r0711.20260423-Linux.tar.gz",
    "sha256_sidecar_key": "release/unsigned/tar/amdrocm7-rvs-1.3.15-r0711.20260423-Linux.tar.gz.sha256"
  }
}
```

### 9. Write job summary

Writes a Markdown table to the GitHub Actions run summary listing the build number, count of promoted packages per format, and the source/destination S3 paths.

## Accumulation semantics

All S3 writes use **accumulate mode** (no `--delete`). Packages from previous promotions remain in `release/unsigned/`. The `reprepro` DEB archive accumulates historically: older `.deb` versions remain in the pool (the reprepro db tracks each Package+Version). RPM repodata is fully regenerated from all RPMs present in the staging directory after the new ones are added.

Signing CI should always read `release/unsigned/latest.json` to find the exact `s3_key` values for a specific promotion — not list the prefix directly.

## Partial promotion

If no packages match the given `build_number` for a given format (DEB, RPM, or TAR), that format step emits a warning and exits successfully without modifying S3. The other format steps still proceed. `latest.json` is published with only the formats that had matching packages.

## Required configuration

**Repository secret** (Settings → Secrets and variables → Actions → Secrets):

| Secret | Purpose |
|--------|---------|
| `AWS_ROLE_ARN` | IAM role ARN to assume for S3 access via OIDC. The role must have `s3:PutObject`, `s3:GetObject`, and `s3:ListBucket` on `release/unsigned/*` and `release/rvs/*`. `s3:DeleteObject` is **not** required. |

**Repository variable** (Settings → Secrets and variables → Actions → Variables):

| Variable | Default | Purpose |
|----------|---------|---------|
| `AWS_S3_BUCKET` | _(required)_ | S3 bucket name. |
| `RUNNER_LABEL_UTILITY` | `ubuntu-latest` | Runner label for the promotion job. |

**AWS IAM trust policy:** The role in `AWS_ROLE_ARN` must allow GitHub OIDC (`token.actions.githubusercontent.com`, audience `sts.amazonaws.com`) to assume it for this repository.

## Relationship to nightly/unsigned

| Attribute | `nightly/unsigned/` | `release/unsigned/` |
|-----------|--------------------|--------------------|
| Populated by | `build-relocatable-packages.yml` (scheduled default branch) | This workflow (manual dispatch) |
| Source packages | `nightly/rvs/` | `release/rvs/` |
| DEB APT suite | `stable main` | `stable main` |
| RPM layout | packages flat in `rpm/` | packages in `rpm/x86_64/`; repodata inside `rpm/x86_64/repodata/` |
| RPM repodata | `createrepo_c` | `createrepo_c` (run on `x86_64/`) |
| `.tar.gz.sha256` sidecars | Always generated by build job | Generated here if absent in source |
| `latest.json` | Per-run, always overwrites | Per-promotion, always overwrites |
| Accumulation | Yes (no `--delete`) | Yes (no `--delete`) |
| Per-run fragments (`runs/<id>/`) | Yes (`deb.json`, `rpm-tar.json`) | No (not needed; promotion is explicit) |

## Triggering the workflow

**From the GitHub Actions UI:**

1. Go to **Actions** → **Unsigned Release Candidate Promotion**
2. Click **Run workflow**
3. Enter the `build_number` (e.g. `r0711.20260423`)
4. Click **Run workflow**

**From the `gh` CLI:**

```bash
gh workflow run unsigned-release-candidate-promotion.yml \
  -f build_number="r0711.20260423"
```

To promote a broader range (e.g. all builds for ROCm 7.11 libpatch):

```bash
gh workflow run unsigned-release-candidate-promotion.yml \
  -f build_number="r0711"
```

## Signing CI handoff

After a successful promotion, signing CI reads `s3://<bucket>/release/unsigned/latest.json`. The file lists:

- `deb.packages[].s3_key` — pool paths for each `.deb` in the APT archive
- `rpm.s3_key` — path for the `.rpm`
- `tar.s3_key` and `tar.sha256_sidecar_key` — paths for the tarball and its SHA-256 sidecar

Signing CI downloads the packages from those keys, signs them, and promotes the signed artifacts to the consumer-facing release repository.

## Verifying the promotion

After the workflow completes, check the S3 paths via the AWS CLI or console:

```bash
BUCKET="<your-bucket>"
BUILD_NUM="r0711.20260423"

# DEB: verify APT index contains the package
aws s3 cp "s3://${BUCKET}/release/unsigned/deb/dists/stable/main/binary-amd64/Packages" - \
  | grep -A5 "${BUILD_NUM}"

# RPM: verify package and repodata (packages live under x86_64/)
aws s3 ls "s3://${BUCKET}/release/unsigned/rpm/x86_64/" | grep "${BUILD_NUM}"
aws s3 ls "s3://${BUCKET}/release/unsigned/rpm/x86_64/repodata/"

# TAR: verify tarball and sidecar
aws s3 ls "s3://${BUCKET}/release/unsigned/tar/" | grep "${BUILD_NUM}"

# latest.json
aws s3 cp "s3://${BUCKET}/release/unsigned/latest.json" -
```

**Using the promoted repo with apt (unsigned staging, internal testing):**

```bash
echo "deb [trusted=yes arch=amd64] https://<bucket>.s3.amazonaws.com/release/unsigned/deb/ stable main" \
  | sudo tee /etc/apt/sources.list.d/rvs-unsigned-release.list
sudo apt update
sudo apt install amdrocm7-rvs
```

**Using the promoted repo with yum/dnf (unsigned staging, internal testing):**

```bash
cat <<'EOF' | sudo tee /etc/yum.repos.d/rvs-unsigned-release.repo
[rvs-unsigned-release]
name=RVS Unsigned Release Candidate RPM
baseurl=https://<bucket>.s3.amazonaws.com/release/unsigned/rpm/x86_64/
enabled=1
gpgcheck=0
EOF
sudo yum install amdrocm7-rvs
```

> **Note:** `[trusted=yes]` (apt) and `gpgcheck=0` (yum) disable GPG verification. These repo definitions are for **internal staging and testing only**, before signing CI produces signed packages for public consumption.

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `AWS_S3_BUCKET repository variable is not set` | The `AWS_S3_BUCKET` Actions variable is missing. Add it in Settings → Secrets and variables → Actions → Variables. |
| `Credentials could not be loaded` | The OIDC trust policy for `AWS_ROLE_ARN` does not cover this repository, or `AWS_ROLE_ARN` is not set as a repository secret. |
| `No .deb files matching '<build_number>'` | No DEB filename in `release/rvs/deb/` contains the given build number. Verify the exact release string in the filename with `aws s3 ls s3://<bucket>/release/rvs/deb/`. |
| `No .rpm files matching '<build_number>'` | Same as above for RPMs. Check `release/rvs/rpm/`. |
| `No .tar.gz files matching '<build_number>'` | Same for tarballs. Check `release/rvs/tar/`. |
| `latest.json` has empty `deb`/`rpm`/`tar` | The Packages index or S3 listing for the destination did not find filenames containing `build_number`. This can happen if the promotion step was skipped (warning, no files) but previous versions exist in the prefix. Re-run with a build number that matches at least one file. |
| `reprepro` fails on `includedeb` | The `.deb` control fields may have unexpected characters, or the `conf/distributions` file is corrupted. Delete `s3://<bucket>/release/unsigned/deb/conf/` to force a fresh archive on next run. |
| RPM repodata not updated | `createrepo_c` may not be available on the runner. The step falls back to `createrepo`; if neither is found, the step fails. The runner label in `RUNNER_LABEL_UTILITY` must resolve to a runner where at least one of those tools can be installed via `apt-get`. |

## References

- [`build-relocatable-packages.yml`](./build-relocatable-packages.yml) — upstream pipeline that produces packages in `release/rvs/`
- [`README_BUILD_PACKAGES.md`](./README_BUILD_PACKAGES.md) — detailed documentation for the build pipeline, including the nightly/unsigned layout it mirrors
- [`rvs-deb-unsigned-repo.sh`](../scripts/rvs-deb-unsigned-repo.sh) — reprepro accumulate logic used by the nightly unsigned pipeline (same pattern as the DEB step here)
- [`rvs-unsigned-publish-latest.sh`](../scripts/rvs-unsigned-publish-latest.sh) — latest.json publisher for nightly/unsigned (same schema as `release/unsigned/latest.json`)
- [`rvs-s3-upload-route.sh`](../scripts/rvs-s3-upload-route.sh) — S3 path routing for build jobs
