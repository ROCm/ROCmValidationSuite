# Promoting release packages to `stable/rvs` (S3)

This document describes the **manual** GitHub Actions workflow that copies **exactly the same** RVS package objects already published under **`release/rvs/`** in S3 into **`stable/rvs/`**, so what you promote is the same payload that was built and typically validated against in `release`.

Workflow file: `.github/workflows/promote-stable-rvs.yml`

Related build workflow: see [README_BUILD_PACKAGES.md](./README_BUILD_PACKAGES.md).

## Why promote?

- **`release/rvs/`** holds builds produced from **release** branches (see `build-relocatable-packages.yml`).
- **`stable/rvs/`** is a separate prefix intended for **promoted**, consumption-ready packages (for example after QA sign-off).
- Promotion uses **`aws s3 cp`** from `release` keys to `stable` keys (same bucket). It does **not** re-upload packages from GitHub Actions artifacts, so the bits match what testers used from **`release/rvs`**.

## Prerequisites

- Workflow runs in **`ROCm/ROCmValidationSuite`** only (steps that touch S3 are gated on `github.repository`).
- Repository configuration expected by the workflow:
  - **`AWS_S3_BUCKET`** (Actions variable): bucket containing `release/rvs/` and `stable/rvs/`.
  - **`AWS_ROLE_ARN`** (secret): IAM role for OIDC (`id-token: write` is set on the workflow).

## Inputs

| Input | Required | Description |
|--------|----------|-------------|
| **`build_number`** | Yes | GitHub Actions **run number** for the **Build Relocatable Packages** run on a **release** branch that uploaded to `release/rvs/`. |
| **`test_report_url`** | No | Optional URL to a test report. Stored in the promotion report (S3 + workflow artifact). |

### Choosing `build_number`

On **release** branches, `build_packages_local.sh` sets the Debian/RPM package **release** field to **`GITHUB_RUN_NUMBER`** when the branch name matches the script’s release-branch logic (`release/…` matches because it begins with `rel…`). That number is embedded in published filenames under **`release/rvs/deb`**, **`release/rvs/rpm`**, and **`release/rvs/tar`**.

Use the **run number** shown in the GitHub UI for the workflow run you are promoting (e.g. “Run #**12345**” → `12345`).

## S3 layout

| Role | Prefix (default) | Contents |
|------|------------------|----------|
| Source | `release/rvs/` | `deb/`, `rpm/`, `tar/` as produced by the build workflow |
| Destination | `stable/rvs/` | Same structure after promotion; **APT** and **RPM** repository metadata are regenerated under `stable` |
| Audit | `stable/rvs/promotions/<build_number>/` | `promotion-report.md` (includes optional test URL and link to this Actions run) |

Paths are controlled by workflow `env`: `RELEASE_PREFIX`, `PROMOTE_BASE`.

## What the workflow does

1. **Authenticate** to AWS with OIDC (same pattern as the build workflow).
2. **List** objects under `release/rvs/deb/`, `release/rvs/rpm/`, and `release/rvs/tar/`.
3. **Select** `amdrocm* … rvs …` packages whose **packaging revision** matches the given **`build_number`** (including RPM-style releases like `456.el9` when the numeric release is `456`).
4. **Copy** each matching object to the parallel key under `stable/rvs/…` via `aws s3 cp` (same bucket).
5. **Regenerate** Debian (`Packages`, `Packages.gz`, `Release`) and RPM (`repodata/`) metadata under **`stable/rvs`** so consumers can use `stable` as an APT/YUM source.
6. **Write** `promotion-report.md` to `stable/rvs/promotions/<build_number>/` and attach it as a workflow artifact.

If **no** objects match, the workflow **fails** so you do not get a silent empty promotion.

## How to run it

1. Open **Actions** → **Promote RVS packages to stable** → **Run workflow**.
2. Enter **`build_number`** (the GitHub run number of the release build you want).
3. Optionally paste **`test_report_url`**.
4. Run the workflow and confirm the job log lists the matched S3 keys and completes successfully.

## Troubleshooting

- **`No packages … matched build number`**: The run number does not appear in any `amdrocm* … rvs …` object key under `release/rvs/` for that revision (wrong number, build never uploaded to `release`, or naming differs). Confirm the correct run on a **release** branch and list objects in S3 under `release/rvs/deb/` (and rpm/tar) for that revision.
- **Steps skipped / no S3 updates**: Confirm the workflow is running on **`ROCm/ROCmValidationSuite`** and that **`AWS_S3_BUCKET`** and **`AWS_ROLE_ARN`** are configured.
