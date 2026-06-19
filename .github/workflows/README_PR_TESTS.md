# RVS PR Tests Workflow

This document describes [`.github/workflows/rvs-pr-tests.yml`](./rvs-pr-tests.yml). It mirrors [RVS Nightly Tests](./README_NIGHTLY_TESTS.md) **orchestration** (self-hosted runner resolves the artifact, `curl` + `scp` to the GPU target, SSH-driven install, `rvs -r 4`, Markdown report) but targets the **Linux relocatable tarball** (`amdrocm*-rvs-*-Linux.tar.gz`) — the **same artifact type as nightly** — not the Ubuntu `.deb` by default. A **direct `.deb` URL** in the secret or `package_url` is still supported.

- **Resolve** ([`rvs_pr_test.sh resolve-package-url`](../../rvs_pr_test.sh)): from an HTTPS **listing root that ends at `/rvs`** (with or without a trailing slash), walks the index and picks the latest **`merge/<pr>/…/…-Linux.tar.gz`**; or uses a **direct HTTPS URL** to that `.tar.gz` (or a `.deb`). Probes the URL with `curl` before install.
- **Scheduled** runs **poll hourly (UTC)**. A **detect** job compares an **artifact key** to an **Actions cache** marker; if unchanged, install and tests are **skipped**. After **successful** level 4, a **save** job updates the marker. **Manual** runs always execute the full pipeline (marker not updated for schedule logic).
- Triggers on **`schedule`** and **`workflow_dispatch` only** — not on **Build Relocatable Packages**.

Install logic is in [`rvs_pr_test.sh`](../../rvs_pr_test.sh): **`.tar.gz`** → `tar -xzf` under `/opt/rocm/extras-<N>/` (same as nightly); **`.deb`** → `dpkg` path.

## How “latest” is chosen (`resolve-package-url`)

**Directory root** (HTTPS index for `https://<host>/rvs` or `https://<host>/rvs/` — same behavior):

1. Largest numeric **build** directory under the root.  
2. **`merge/`** → largest numeric **PR** directory.  
3. Under that PR, a subdirectory listing **`amdrocm*-rvs-*-Linux.tar.gz`** (prefers **`manylinux_*`**).  
4. Version-sort tarball names → **full HTTPS URL** to that file.

**Direct URL**: if the configured value is already a `*-Linux.tar.gz` (or other `*.tar.gz` accepted by validation) or a **`.deb`**, that URL is used as-is.

## Triggers

| Trigger | When | Behavior |
|---|---|---|
| `schedule` | `0 * * * *` UTC (hourly) | Detect new artifact key vs cache → optionally full install + level 4 + report; save marker after **successful** level 4. |
| `workflow_dispatch` | Manual | Always run full pipeline. |

## Repository configuration

Use the same **secrets** and **variables** as nightly for SSH and ROCm (`RVS_TARGET_NODE`, `RVS_TARGET_USER`, `RVS_TARGET_SSH_KEY`, `RVS_TARGET_ROCM_PATH`, runner labels, etc.). See [README_NIGHTLY_TESTS.md](./README_NIGHTLY_TESTS.md#repository-configuration).

**PR package URL secrets** (parallel idea to nightly’s tarball index secret):

| Name | Kind | Purpose |
|---|---|---|
| `RVS_PR_DEB_PACKAGE_URL` | Secret | **Preferred.** HTTPS **index root through `/rvs/`** only (e.g. `https://<cdn-host>/rvs/` — no build/merge path; trailing `/` optional), **or** a **direct** `*-Linux.tar.gz` URL, **or** a **`.deb`** URL. Used when `package_url` is empty. |
| `RVS_PR_PACKAGE_URL` | Secret | Legacy alias if `RVS_PR_DEB_PACKAGE_URL` is unset. |

## Manual run

```bash
gh workflow run rvs-pr-tests.yml
```

Examples — set your real values in secrets or `-f package_url`:

**Index root only** (resolver follows `rvs/<build>/merge/<pr>/…/*.tar.gz`):

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
