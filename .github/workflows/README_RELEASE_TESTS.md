# RVS Release Tests Workflow

This document describes [`.github/workflows/rvs-release-tests.yml`](./rvs-release-tests.yml),
which picks up the **latest RVS release tarball** from the release tarball index
(`secrets.RVS_RELEASE_TARBALL_INDEX_URL`, passed through workflow `env.TARBALL_INDEX_URL` —
**no default URL in the repo**), copies it to a
**dedicated release test GPU node** over SSH, installs it there, and runs **RVS level 5**
(full stress) on that node.

Release testing is intentionally separate from [nightly tests](./README_NIGHTLY_TESTS.md):
different tarball channel, different target-node secrets, and a heavier test level.

The GitHub Actions runner is only an **orchestrator** — it does not need a GPU or ROCm
locally. Install, binary verification, and `rvs -r 5` run on the remote target.

## What it does

```
workflow_run (release/* build) / workflow_dispatch
    │
    ▼
gate-release-trigger        [utility runner]  — skip non-release workflow_run events
    ▼
install-rvs-on-target       [self-hosted orchestrator]  ──ssh──▶  [release GPU node]
    │  resolve release index + validate paths (orchestrator only)
    │  setup-ssh → download .tar.gz on runner → scp to target
    │  verify-rocm → install-rvs → verify-rvs-binary
    ▼
run-rvs-level-5             [self-hosted orchestrator]  ──ssh──▶  [release GPU node]
    │  rvs_release_test.sh: run-level5 (release tarballs only), collect-logs, capture-versions
    │  upload intermediate logs artifact; cleanup remote work dir
    ▼
create-test-report          [utility runner]
    │  rvs_release_test.sh build-report → SUMMARY.md + final artifact
    ▼
artifact: rvs-release-report-<run_id>
```

Install, test, and report logic lives in [`rvs_release_test.sh`](../../rvs_release_test.sh)
at the repo root — independent from [`rvs_nightly_test.sh`](../../rvs_nightly_test.sh) (nightly)
and [`rvs_pr_test.sh`](../../rvs_pr_test.sh) (PR).

## Triggers

| Trigger | When it runs |
|---|---|
| `workflow_run` | After **Build Relocatable Packages** completes **successfully** on a `release/*` branch. Builds on `main`, feature branches, or failed release builds do **not** start this workflow. |
| `workflow_dispatch` | Manual run from the Actions UI (or `gh workflow run`). Always eligible; does not require a prior package build. |

There is no `schedule` trigger — release validation is tied to release package publication
or explicit manual runs.

## Manual dispatch inputs

| Input | Default | Description |
|---|---|---|
| `tarball_url` | _(empty → latest from release index)_ | Download this exact URL instead of scraping the release index. |
| `target_node` | _(empty → `secrets.RVS_RELEASE_TARGET_NODE`)_ | Hostname or IP of the release test GPU node. |
| `target_user` | _(empty → `secrets.RVS_RELEASE_TARGET_USER`)_ | SSH user on the target node. |
| `remote_work_dir` | _(empty → `vars.RVS_RELEASE_REMOTE_WORK_DIR`, then `/tmp/rvs-release-<run_id>`)_ | Staging and log directory on the target; removed at job end. |
| `target_rocm_path` | _(empty → `secrets.RVS_RELEASE_TARGET_ROCM_PATH`)_ | Absolute path to the ROCm install root on the target. |

Workflow inputs override repo secrets for that run only.

Examples:

```bash
# Latest release tarball from the index secret, default release target secrets
gh workflow run rvs-release-tests.yml

# Pin a specific release tarball
gh workflow run rvs-release-tests.yml \
  -f tarball_url="<RELEASE_TARBALL_URL>/amdrocm7-rvs-1.4.21-r0711.20260623-Linux.tar.gz"

# One-off run against a different node (secrets unchanged)
gh workflow run rvs-release-tests.yml \
  -f target_node="<HOST_OR_IP>" \
  -f target_user="<USER>" \
  -f target_rocm_path="<ROCM_INSTALL_PATH>"
```

## Repository configuration

### Secrets (release-specific)

All release target settings are **secrets** (masked as `***` in logs):

| Name | Required? | Purpose |
|---|---|---|
| `RVS_RELEASE_TARGET_NODE` | **Required** *(unless every run sets `target_node` input)* | Hostname or IP of the release validation GPU node. |
| `RVS_RELEASE_TARGET_USER` | optional | SSH user on the release target. If unset, SSH falls back to the orchestrator runner's local user. |
| `RVS_RELEASE_TARGET_SSH_KEY` | **Required** | Private SSH key authorized on the release target for `RVS_RELEASE_TARGET_USER`. |
| `RVS_RELEASE_TARGET_ROCM_PATH` | **Required** *(unless every run sets `target_rocm_path` input)* | Absolute path to the ROCm install root on the release target (`bin/rocminfo`, `bin/amd-smi`, `lib/`, etc.). |
| `RVS_RELEASE_TARBALL_INDEX_URL` | **Required** *(unless every run provides `workflow_dispatch` `tarball_url`; `workflow_run` needs this secret)* | HTTPS directory listing URL scraped for the latest `amdrocm*-rvs-*-Linux.tar.gz` release tarball. Copied into workflow `env.TARBALL_INDEX_URL`. GitHub masks secret values in logs when printed. |

These are **independent** from nightly/PR secrets (`RVS_TARGET_*`, `RVS_TARBALL_INDEX_URL`). Use a dedicated
release-validation host so full-stress level 5 does not contend with nightly level 4.

### Variables (optional)

| Name | Default | Purpose |
|---|---|---|
| `RVS_RELEASE_INDEX_RUNNER_LABEL` | falls back to `RVS_NIGHTLY_INDEX_RUNNER_LABEL`, then `RVS_TEST_RUNNER_LABEL`, then `self-hosted` | Runner label for **install-rvs-on-target** (index resolve + download + scp). |
| `RVS_RELEASE_TEST_RUNNER_LABEL` | falls back to `RVS_TEST_RUNNER_LABEL`, then `self-hosted` | Runner label for **run-rvs-level-5**. |
| `RVS_RELEASE_REMOTE_WORK_DIR` | `/tmp/rvs-release-<run_id>` | Fixed remote work dir when not using the per-run input. |

## How the latest release tarball is picked

Unless `workflow_dispatch.tarball_url` is set, **install-rvs-on-target** scrapes
`secrets.RVS_RELEASE_TARBALL_INDEX_URL` (workflow `env.TARBALL_INDEX_URL`):

```bash
curl -sL "$INDEX_URL" \
  | grep -oE 'amdrocm[0-9]*-rvs-[0-9A-Za-z._\-]+-Linux\.tar\.gz' \
  | sort -uV \
  | tail -n 1
```

The version-sorted newest filename is downloaded on the orchestrator and `scp`'d to the
target. The target never fetches the index or tarball URL directly.

Release tarballs are published by
[Build Relocatable Packages](./build-relocatable-packages.yml) when a `release/*`
branch build completes (CentOS job uploads the `.tar.gz` to the release tarball prefix).

## The test — level 5 (full stress)

On the target node (with `$RVS_BIN` = `/opt/rocm/extras-<N>/bin/rvs`):

```bash
"$RVS_BIN" -r 5
```

Level 5 runs the full stress suite (extended duration vs level 4). Expect **hours** of
runtime depending on GPU SKU and configuration. The **run-rvs-level-5** job timeout is
**480 minutes** (8 hours); increase `timeout-minutes` in the workflow if your hardware
needs longer.

Stdout/stderr is captured to `$REMOTE_WORK_DIR/reports/rvs_level_5.log` on the target,
then copied back to `./reports/` on the orchestrator.

See [RVS user guide — test levels](../../docs/ug1main.md) for what level 5 includes.

## Test report

`create-test-report` writes `reports/SUMMARY.md` and uploads artifact
`rvs-release-report-<run_id>`:

```markdown
# RVS Release Test Report

| Field | Value |
|---|---|
| Tarball | `amdrocm7-rvs-…-Linux.tar.gz` |
| Overall result | **PASS** |

## Results

| Test    | Command                              | Result | Exit |
|---------|--------------------------------------|:------:|-----:|
| Level 5 | `/opt/rocm/extras-7/bin/rvs -r 5`    | PASS   |    0 |
```

Artifact contents:

```
rvs-release-report-<run_id>/
├── SUMMARY.md
└── rvs_level_5.log
```

## Target node prerequisites

Same requirements as nightly tests — see
[README_NIGHTLY_TESTS.md — prerequisites](./README_NIGHTLY_TESTS.md#target-node-prerequisites)
for SSH, `NOPASSWD` sudo, ROCm layout, and the manual validation one-liner.

Quick checklist for the **release** node:

- SSH reachable from the orchestrator; `RVS_RELEASE_TARGET_SSH_KEY` authorized.
- ROCm at `RVS_RELEASE_TARGET_ROCM_PATH` with `rocminfo` and `amd-smi` working.
- GPU(s) suitable for sustained level 5 stress (power/thermal headroom, no conflicting jobs).
- `sudo -n` if `/opt/rocm/extras-<N>` is not user-writable.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Workflow skipped after a main-branch package build | Expected — only `release/*` builds pass `gate-release-trigger`. |
| `No tarball index URL` | Set `RVS_RELEASE_TARBALL_INDEX_URL` secret or pass `tarball_url` on manual dispatch. |
| `Required environment variable TARGET_NODE is not set` | Add `RVS_RELEASE_TARGET_NODE` secret or `target_node` input. |
| Level 5 job cancelled at 8h | Increase `timeout-minutes` on **run-rvs-level-5** for slower SKUs. |
| `ldd` / library errors before level 5 | Fix ROCm or RVS install paths on the target; see nightly README verify steps. |

For orchestrator SSH, index resolution, and log collection details, see
[README_NIGHTLY_TESTS.md](./README_NIGHTLY_TESTS.md).

## Related workflows

| Workflow | Script | Tarball source | Test level | Target secrets |
|---|---|---|---|---|
| [rvs-nightly-tests.yml](./rvs-nightly-tests.yml) | `rvs_nightly_test.sh` | `RVS_TARBALL_INDEX_URL` | 4 | `RVS_TARGET_*` |
| [rvs-pr-tests.yml](./rvs-pr-tests.yml) | `rvs_pr_test.sh` | PR package URL | 4 | `RVS_TARGET_*` |
| **rvs-release-tests.yml** | **`rvs_release_test.sh`** | **`RVS_RELEASE_TARBALL_INDEX_URL`** | **5** | **`RVS_RELEASE_TARGET_*`** |
| [build-relocatable-packages.yml](./build-relocatable-packages.yml) | — | _(builds packages)_ | — | — |
