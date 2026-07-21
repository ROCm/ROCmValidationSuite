# RVS PR Tests Workflow

This document describes [`.github/workflows/rvs-pr-tests.yml`](./rvs-pr-tests.yml). It mirrors [RVS Nightly Tests](./README_NIGHTLY_TESTS.md) **orchestration** (self-hosted runner resolves the artifact, `curl` + `scp` to the GPU target, SSH-driven install, `rvs -r 4`, Markdown report) but targets the **Linux relocatable tarball** (`amdrocm*-rvs-*-Linux.tar.gz`) — not the Ubuntu `.deb` by default. A **direct `.deb` URL** is still supported for manual runs.

## Triggers

| Trigger | When | Behavior |
|---|---|---|
| `workflow_run` | After **[Build Relocatable Packages](https://github.com/ROCm/ROCmValidationSuite/actions/workflows/build-relocatable-packages.yml)** completes | Runs **only** when the triggering run **`conclusion` is `success`** and the triggering event is **`pull_request`**. Resolves the tarball at `…/rvs/<pr>/merge/<run_number>/manylinux_2_28/` (same path layout as [`.github/scripts/rvs-s3-upload-route.sh`](../../.github/scripts/rvs-s3-upload-route.sh) for PR uploads). |
| `workflow_dispatch` | Manual | Always runs (subject to secrets/inputs). |

There is **no** `schedule` and **no** CDN polling or Actions cache marker.

Install logic is in [`rvs_pr_test.sh`](../../rvs_pr_test.sh): **`.tar.gz`** → `tar -xzf` under `/opt/rocm/extras-<N>/` (same as nightly); **`.deb`** → `dpkg` path.

## How the package URL is chosen (`resolve-package-url`)

**After a successful PR build (`workflow_run`)**

1. Read **`github.event.workflow_run.pull_requests[0].number`** and **`github.event.workflow_run.run_number`** (the Build workflow’s run number, same as **`GITHUB_RUN_NUMBER`** during S3 upload in **Build Relocatable Packages**).
2. Set secret **`RVS_PR_DEB_PACKAGE_URL`** (or **`RVS_PR_PACKAGE_URL`**) to the HTTPS **`…/rvs`** (or **`…/rvs/`**) base only — no `merge/` or `manylinux` segments (for example the CloudFront root you use for PR artifacts).
3. The workflow probes **`manylinux_2_28`** listings in this order (first that returns a listing wins):
   - **`{base}/{pr}/merge/{run_number}/manylinux_2_28/`** — matches [`.github/scripts/rvs-s3-upload-route.sh`](../../.github/scripts/rvs-s3-upload-route.sh) PR layout (`rvs/${GITHUB_REF_NAME}/${GITHUB_RUN_NUMBER}/manylinux_2_28` with `GITHUB_REF_NAME` = `<pr>/merge`).
   - **`{base}/{run_number}/merge/{pr}/manylinux_2_28/`** — alternate mirror layout.
4. [`rvs_pr_test.sh resolve-package-url`](../../rvs_pr_test.sh) treats that URL as a **`manylinux_*`** directory listing, selects the latest **`amdrocm*-rvs-*-Linux.tar.gz`**, then download → target install → **`rvs -r 4`** on the GPU host (see nightly README for SSH / `RVS_TARGET_*`).

If **`pull_requests`** is empty (common for some fork PR flows), the job fails with a clear error — set up a manual run with `package_url` instead, or adjust branch/secret policy so the payload includes the PR.

**Manual (`workflow_dispatch`)**

- **`package_url`** or **`RVS_PR_DEB_PACKAGE_URL`** / **`RVS_PR_PACKAGE_URL`** may be:
  - **`…/rvs/`** index root → crawler picks latest `rvs/<build>/merge/<pr>/…/*.tar.gz` (legacy layout),
  - a **`…/manylinux_*/`** directory listing → latest tarball in that folder,
  - a **direct** `*-Linux.tar.gz` or **`.deb`** URL.

## Repository configuration

Use the same **secrets** and **variables** as nightly for SSH and ROCm (`RVS_TARGET_NODE`, `RVS_TARGET_USER`, `RVS_TARGET_SSH_KEY`, `RVS_TARGET_ROCM_PATH`, runner labels, etc.). See [README_NIGHTLY_TESTS.md](./README_NIGHTLY_TESTS.md#repository-configuration).

**PR package URL secrets**

| Name | Kind | Purpose |
|---|---|---|
| `RVS_PR_DEB_PACKAGE_URL` | Secret | **Required** for `workflow_run`: HTTPS base **`…/rvs`** or **`…/rvs/`** only (no `merge/` / `manylinux` path). **Manual runs:** same base, or `…/manylinux_*/` listing, or direct tarball / `.deb`. |
| `RVS_PR_PACKAGE_URL` | Secret | Legacy alias if `RVS_PR_DEB_PACKAGE_URL` is unset. |

## Manual run

```bash
gh workflow run rvs-pr-tests.yml
```

Examples — set your real values in secrets or `-f package_url`:

**`…/rvs/` index root** (crawler uses `rvs/<build>/merge/<pr>/…`):

```bash
gh workflow run rvs-pr-tests.yml \
  -f package_url="https://<cdn-host>/rvs/" \
  -f target_node="<HOST_OR_IP>"
```

**Direct tarball URL**:

```bash
gh workflow run rvs-pr-tests.yml \
  -f package_url="https://<internal-host>/<path>/amdrocm7-rvs-…-Linux.tar.gz" \
  -f target_node="<HOST_OR_IP>"
```

Artifacts: `rvs-pr-logs-<run_id>`, `rvs-pr-report-<run_id>`.

## Target prerequisites

Same as nightly: for **tarball** installs, target layout matches **relocatable RVS** under `/opt/rocm/extras-<N>/`; **`sudo -n`** if `/opt` is not user-writable. For **`.deb`**, passwordless **`sudo`** for `dpkg` / `apt-get`. ROCm at **`RVS_TARGET_ROCM_PATH`**. Orchestrator needs HTTPS egress to your CDN and SSH/SCP to the target.
