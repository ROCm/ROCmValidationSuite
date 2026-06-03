# RVS Nightly Tests Workflow

This document describes [`.github/workflows/rvs-nightly-tests.yml`](./rvs-nightly-tests.yml),
which picks up the **latest RVS tarball** from the index URL configured in
`vars.RVS_TARBALL_INDEX_URL` (e.g. `https://repo.amd.com/rocm/rvs/tarball/`)
once per day, copies it to a **configurable remote target node** over SSH,
installs it there, and runs RVS level 4 on that node.

The GitHub Actions runner ("RVS Runner") is only an **orchestrator** — it
doesn't need a GPU or ROCm installed locally. All RVS install, binary
verification, and `rvs -r 4` execution happens on the target
node. The target node is configurable so the same workflow can be pointed
at any GPU host without code changes.

## What it does

```
schedule / workflow_run / manual
    │
    ▼
detect-tarball              [utility runner]
    │  curl $RVS_TARBALL_INDEX_URL → pick newest amdrocm*-rvs-*.tar.gz
    ▼
prepare-test-context        [utility runner]
    │  rvs_nightly_test.sh validate-config → paths + remote work dir
    ▼
install-rvs-on-target       [test runner]  ──ssh──▶  [target GPU node]
    │  rvs_nightly_test.sh: setup-ssh, verify-rocm, download, copy,
    │                         install-rvs, verify-rvs-binary
    ▼
run-rvs-level-4             [test runner]  ──ssh──▶  [target GPU node]
    │  rvs_nightly_test.sh: run-level4, collect-logs, capture-versions
    │  upload intermediate logs artifact; cleanup remote work dir
    ▼
create-test-report          [utility runner]
    │  rvs_nightly_test.sh build-report → SUMMARY.md + final artifact
    ▼
artifact: rvs-nightly-report-<run_id>
```

Install, test, and report logic lives in [`rvs_nightly_test.sh`](../../rvs_nightly_test.sh)
at the repo root — the same split as `build-relocatable-packages.yml` +
`build_packages_local.sh`.

## Triggers

| Trigger | Cadence | What fires |
|---|---|---|
| `schedule` | `0 15 * * *` UTC daily (08:00 PST / 07:00 PDT) | Polls the tarball index and always runs install + level 4, even when the latest tarball filename matches the previous run (a notice is logged when unchanged). |
| `workflow_run` | After **Build Relocatable Packages** completes | Runs only when that workflow's overall conclusion is **success**. No changes to the build workflow are required. |
| `workflow_dispatch` | Manual | Always runs. Supports overriding the tarball URL and **retargeting at any node** without editing the workflow. |

The cron deliberately runs after AMD's typical nightly publish window;
adjust the cron string in the workflow if your publish cadence is different.

## Manual dispatch inputs

| Input | Default | Description |
|---|---|---|
| `tarball_url` | _(empty)_ | If set, the workflow downloads this exact URL instead of scraping the index. Useful for re-running an older build. |
| `target_node` | _(empty → `secrets.RVS_TARGET_NODE`)_ | Hostname or IP of the node to install RVS on and run tests against. **This is the value that retargets the test execution.** Stored as a secret so the lab node identity isn't visible in repo settings or run logs (GitHub Actions automatically masks secret values as `***` in step output). |
| `target_user` | _(empty → `secrets.RVS_TARGET_USER`; if both are unset, SSH defaults to the orchestrator runner's local user)_ | SSH user on the target node. Must have `NOPASSWD` sudo on the target (see prerequisites). Stored as a secret so the lab account name isn't visible in repo settings or run logs (GitHub Actions automatically masks secret values as `***` in step output). |
| `remote_work_dir` | _(empty → `vars.RVS_REMOTE_WORK_DIR`, then `/tmp/rvs-nightly-<run_id>`)_ | Working dir on the target node where the tarball is staged, logs are written, and which gets `rm -rf`'d at the end. |
| `target_rocm_path` | _(empty → `vars.RVS_TARGET_ROCM_PATH`)_ — **required**, no hard-coded default | Absolute path to the ROCm tarball install root on the target node. This is the directory containing `bin/rocminfo`, `bin/amd-smi`, `lib/`, `lib/llvm/lib/`, and `lib/rocm_sysdeps/lib/` (the layout produced by the ROCm tarball install method). The workflow fails fast in the validate step if neither the input nor the variable is set. |

Workflow inputs **win over repo variables**, so individual `workflow_dispatch`
runs can be retargeted from the Actions UI without changing repo settings.

Example — point a single run at a specific node:

```bash
gh workflow run rvs-nightly-tests.yml \
  -f tarball_url="<INDEX_URL>/amdrocm7-rvs-1.4.21-288-Linux.tar.gz" \
  -f target_node="<HOST_OR_IP>" \
  -f target_user="<USER>"
```

Example — keep the default tarball but run on a different host today:

```bash
gh workflow run rvs-nightly-tests.yml -f target_node="<HOST_OR_IP>"
```

Example — point at a specific ROCm install on a multi-ROCm host:

```bash
gh workflow run rvs-nightly-tests.yml \
  -f target_node="<HOST_OR_IP>" \
  -f target_rocm_path="<ROCM_INSTALL_PATH>"
```

## Repository configuration

**Variables** (Settings → Secrets and variables → Actions → Variables):

| Name | Required? | Purpose |
|---|---|---|
| `RVS_TARBALL_INDEX_URL` | **Required** | Directory listing scraped for the latest tarball, e.g. `https://repo.amd.com/rocm/rvs/tarball/`. No fallback — the workflow fails fast if unset and no `tarball_url` input is supplied. |
| `RVS_REMOTE_WORK_DIR` | optional (default `/tmp/rvs-nightly-<run_id>`) | Working dir on the target node. Cleared with `rm -rf` at the end of the job. |
| `RVS_TARGET_ROCM_PATH` | **Required** *(unless every run sets `target_rocm_path` input)* | Absolute path to the ROCm tarball install root on the target node — the directory that contains `bin/rocminfo`, `bin/amd-smi`, `lib/`, `lib/llvm/lib/`, and `lib/rocm_sysdeps/lib/`. The workflow doesn't assume any conventional path (no `/opt/rocm` default), since tarball installs land wherever you extracted them. |
| `RVS_TEST_RUNNER_LABEL` | optional (default `self-hosted`) | Label of the GitHub runner that orchestrates the workflow. The runner doesn't need a GPU or ROCm — it just needs `ssh`, `scp`, and `curl`. |

**Secrets** (Settings → Secrets and variables → Actions → Secrets):

| Name | Required? | Purpose |
|---|---|---|
| `RVS_TARGET_NODE` | **Required** *(unless every run sets `target_node` input)* | Hostname or IP of the node where RVS is installed and tests run. Stored as a secret so the lab node identity isn't visible in repo Variables or in run logs — GitHub Actions automatically masks secret values as `***` wherever they appear in step output. Workflow fails fast on `schedule` if neither this secret nor `target_node` input is set. |
| `RVS_TARGET_USER` | optional (no hard-coded default) | SSH user on the target node. If unset and `target_user` input is empty, the SSH client falls back to the orchestrator runner's local user — set this secret explicitly to avoid surprises. Stored as a secret so the lab account name isn't visible in repo settings or run logs (auto-masked as `***`). |
| `RVS_TARGET_SSH_KEY` | **Required** | Private SSH key (OpenSSH or PEM format) authorized on the target node for `RVS_TARGET_USER`. Written to `$RUNNER_TEMP/rvs_target_key` for the duration of the job and scrubbed in the cleanup step. |

## How the latest tarball is picked

The `detect` job does (with `$INDEX_URL` = `vars.RVS_TARBALL_INDEX_URL`):

```bash
curl -sL "$INDEX_URL" \
  | grep -oE 'amdrocm[0-9]*-rvs-[0-9A-Za-z._\-]+-Linux\.tar\.gz' \
  | sort -uV \
  | tail -n 1
```

The regex matches any `amdrocm<N>-rvs-…-Linux.tar.gz` filename in the
directory-listing HTML. `sort -V` is GNU "version sort" so version
suffixes like `1.4.21-9` and `1.4.21-100` compare correctly. The
**lexicographically largest by version** is selected.

## How the tarball is installed (on the target node)

The runner derives the ROCm major version from the tarball filename
(e.g. `amdrocm7-rvs-1.4.21-…-Linux.tar.gz` → `7`), combines it with
`TARGET_ROCM_PATH` (which selects *which* ROCm install to use), writes
the derived paths to `$GITHUB_ENV`, then SSHes into the target with those
values exported so the install runs against the matching `extras-<major>`
directory under the chosen ROCm. The same workflow handles ROCm 6, 7,
etc., and any version inside `7.x` without code changes:

```bash
# On the runner (Validate target node configuration step):
# Parsed from $TARBALL_NAME via [[ "$TARBALL_NAME" =~ ^amdrocm([0-9]+)- ]]
ROCM_MAJOR=7
# INSTALL_DIR is always /opt/rocm/extras-<major>/ — it's where the RVS
# tarball gets extracted. This is decoupled from TARGET_ROCM_PATH so the
# install location matches the manual command verbatim regardless of
# which ROCm install the workflow is told to run *against*.
INSTALL_DIR=/opt/rocm/extras-${ROCM_MAJOR}              # /opt/rocm/extras-7
RVS_BIN=${INSTALL_DIR}/bin/rvs

# TARGET_ROCM_PATH (from inputs.target_rocm_path / vars.RVS_TARGET_ROCM_PATH;
# required, no default) is *where the ROCm runtime libraries live*. Set
# the repo variable to the absolute install root of the ROCm tarball you
# want RVS to run against (the directory that contains bin/, lib/, etc.).

# On the target node (Install RVS on target node step), via SSH:
# sudo is used only when $INSTALL_DIR isn't user-writable. /opt/rocm/* is
# typically root-owned so this picks up sudo -n automatically.
mkdir -p "$INSTALL_DIR"     # (with `sudo -n` if needed)
tar -xzf "$REMOTE_WORK_DIR/pkg/<tarball>.tar.gz" -C "$INSTALL_DIR"

# LD_LIBRARY_PATH is wired off INSTALL_DIR (for RVS's own libs) plus the
# three TheRock-style subdirs of TARGET_ROCM_PATH (matches the official
# ROCm tarball-install docs). This is what makes a non-/opt/rocm
# TARGET_ROCM_PATH actually do something useful.
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TARGET_ROCM_PATH}/lib/rocm_sysdeps/lib:${TARGET_ROCM_PATH}/lib/llvm/lib:${TARGET_ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

"$RVS_BIN" --version
```

### Install location vs. ROCm runtime path

Two paths in the workflow look similar but mean very different things, and
keeping them straight is what made the multi-ROCm-host case finally work:

| Variable | What it is | Default | How to override |
|---|---|---|---|
| `INSTALL_DIR` | Where the RVS tarball gets extracted on the target node. Always under `/opt/rocm/`, matching the manual command. | `/opt/rocm/extras-${ROCM_MAJOR}` | Not configurable by design — the install location is canonical, decoupled from where ROCm itself lives. |
| `TARGET_ROCM_PATH` | Where the ROCm runtime libraries live (the directory containing `bin/`, `lib/`, `lib/llvm/lib/`, `lib/rocm_sysdeps/lib/`). Drives `LD_LIBRARY_PATH`, the prereq-check probe, and the version row in the report. | **(no default — required)** | `inputs.target_rocm_path` (workflow_dispatch) or `vars.RVS_TARGET_ROCM_PATH` (repo Variables tab). Workflow fails fast in the validate step if neither is set. |

This split exists because hosts that have multiple ROCm installs side-by-side
(or that installed ROCm via a TheRock tarball under `$HOME`) almost never
keep the runtime libs at `/opt/rocm/lib/`. The workflow needs to know where
`/lib/`, `/lib/llvm/lib/`, and `/lib/rocm_sysdeps/lib/` actually are; the
install location of RVS itself should not — and does not — care.

The runtime `LD_LIBRARY_PATH` for every step that executes `rvs` (install,
ldd-verify, level 4, report `--version` query) is now:

```bash
LD_LIBRARY_PATH=$INSTALL_DIR/lib:$TARGET_ROCM_PATH/lib/rocm_sysdeps/lib:$TARGET_ROCM_PATH/lib/llvm/lib:$TARGET_ROCM_PATH/lib:$LD_LIBRARY_PATH
```

Mapping back to the manual sequence the RVS team uses today:

```bash
sudo mkdir -p /opt/rocm/extras-7
sudo tar -xzf amdrocm7-rvs-1.4.21-288-Linux.tar.gz -C /opt/rocm/extras-7
export LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:/install/lib:/install/lib/rocm_sysdeps/:/install/lib/llvm/lib:$LD_LIBRARY_PATH
```

| Manual entry | Workflow equivalent | Notes |
|---|---|---|
| `/opt/rocm/extras-7/lib` | `${INSTALL_DIR}/lib` | Same path — `INSTALL_DIR` is literally `/opt/rocm/extras-<major>`. |
| `/install/lib` | `${TARGET_ROCM_PATH}/lib` | The manual command's `/install/` was a literal filesystem path that only existed on the RVS build container. The workflow replaces it with the configurable `TARGET_ROCM_PATH/lib`, which is what "the ROCm install's lib directory" actually means on a real host. |
| `/install/lib/rocm_sysdeps/` | `${TARGET_ROCM_PATH}/lib/rocm_sysdeps/lib` | Same fix, plus the trailing `/lib` that the official ROCm tarball docs include and the manual command was missing. |
| `/install/lib/llvm/lib` | `${TARGET_ROCM_PATH}/lib/llvm/lib` | Where `libomp.so` lives in TheRock builds. |

### What to set `RVS_TARGET_ROCM_PATH` to

Set it to the absolute path of the ROCm **tarball install root** on the target node — i.e. the directory you (or whoever provisioned the node) chose when extracting the ROCm distribution tarball. That directory must contain at least `bin/rocminfo`, `bin/amd-smi`, `lib/`, `lib/llvm/lib/`, and `lib/rocm_sysdeps/lib/`. There is no hard-coded default; the workflow fails fast in the validate step if `RVS_TARGET_ROCM_PATH` and the per-run `target_rocm_path` input are both empty.

The RVS tarball itself still always lands in `/opt/rocm/extras-<major>/` regardless — that part is invariant.

### TheRock-style tarball ROCm installs

The [official ROCm 7.12+ "tarball" install method](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html?fam=instinct&gpu=mi350x&os=ubuntu&os-version=24.04&i=tar)
doesn't drop ROCm under `/opt/rocm-<ver>/`. Instead you extract a
single distribution archive (e.g.
`therock-dist-linux-<arch>-dcgpu-<version>.tar.gz`) into an
arbitrary directory, set `ROCM_PATH=$(pwd)/install`, and source the
env. So the layout looks like:

```
/home/<user>/<wherever-you-ran-tar>/install/
├── bin/                     ← rocm-smi, rocminfo, hipcc, …
├── lib/
│   ├── rocm_sysdeps/lib/    ← ROCm runtime libs live HERE in TheRock builds
│   └── llvm/lib/
├── share/
└── …
```

This works with the workflow as-is, with two things to be aware of:

1. **Find the absolute path of the install on the target node** — there's no canonical location, you pick it at install time. Either ask whoever installed it, or search:
   ```bash
   ssh <USER>@<HOST_OR_IP> '
     for cand in ~/install ~/*/install ~/rocm*/install /opt/therock*/install; do
       [ -x "$cand/bin/rocminfo" ] && echo "  found ROCm install: $cand"
     done
   '
   ```
2. **Pass that absolute path as `target_rocm_path`.** Example:
   ```bash
   gh workflow run rvs-nightly-tests.yml \
     -f target_node=<HOST_OR_IP> \
     -f target_rocm_path=$HOME/<wherever-you-extracted>/install
   ```

What the workflow handles automatically for tarball installs:

- The install step **skips `sudo`** when the destination is user-writable (TheRock installs in `$HOME` don't need root); only system `/opt/rocm-*` installs trigger `sudo -n`.
- The prereq check invokes `${TARGET_ROCM_PATH}/bin/rocminfo` and `${TARGET_ROCM_PATH}/bin/amd-smi version` directly by absolute path. TheRock tarball binaries have RPATH/RUNPATH baked in relative to `${TARGET_ROCM_PATH}/lib`, so they resolve their own ROCm libs without needing `PATH` or `LD_LIBRARY_PATH` setup.
- The version-string row in the report uses what `rvs --version` prints, so it always reflects the RVS tarball — not the underlying ROCm. The major-version cross-check the workflow used to do against `.info/version` is gone (TheRock installs don't ship one).

**To use a TheRock-style install just set `vars.RVS_TARGET_ROCM_PATH` (or pass `-f target_rocm_path=...`) to the absolute install root, e.g. `$HOME/<wherever>/install`. The workflow's `LD_LIBRARY_PATH` is wired off `TARGET_ROCM_PATH` (see [Install location vs. ROCm runtime path](#install-location-vs-rocm-runtime-path)), so the loader picks up `lib/`, `lib/llvm/lib/`, and `lib/rocm_sysdeps/lib/` from the install you point at — no file edits required.

`sudo -n` is non-interactive — it fails fast instead of hanging if
`NOPASSWD` isn't configured. There's no PTY over the SSH channel anyway,
so an interactive sudo prompt would deadlock the job.

The step fails fast if the filename doesn't match `^amdrocm<digits>-`, or
if `$RVS_BIN` isn't executable after extraction.

### Prerequisites

**On the GitHub runner (orchestrator):**

- `ssh`, `scp`, `ssh-keyscan`, and `curl` on `PATH`.
- Network egress to:
  - the host serving `vars.RVS_TARBALL_INDEX_URL` (typically port 443),
  - the target node (port 22 — or wherever its sshd listens, configurable in the SSH config block of the workflow).
- The runner does **not** need a GPU, ROCm, or `sudo`.

**On the target node** (all enforced by the **Pre-flight ROCm checks** below — the workflow fails fast if any are missing):

- SSH server reachable from the runner, with the public counterpart of `secrets.RVS_TARGET_SSH_KEY` authorized for `$TARGET_USER`.
- `$TARGET_USER` has **`NOPASSWD` sudo** for `mkdir` + `tar` into `/opt/rocm/extras-<major>` — the install step uses `sudo -n` (when the path isn't user-writable) and aborts otherwise.
- A ROCm tarball install at `$TARGET_ROCM_PATH` (configured via `vars.RVS_TARGET_ROCM_PATH` or the per-run `target_rocm_path` input) that provides at least `bin/rocminfo` and `bin/amd-smi`. The RVS tarball only ships RVS, not the rest of ROCm, so the runtime libs under `$TARGET_ROCM_PATH/lib`, `$TARGET_ROCM_PATH/lib/llvm/lib`, and `$TARGET_ROCM_PATH/lib/rocm_sysdeps/lib` must be present.
- Working kernel driver (`amdgpu`) and at least one GPU enumerated by `rocminfo` / `amd-smi`.

## Pre-flight ROCm checks

Before downloading or installing the tarball, the workflow runs **`Verify ROCm prerequisites on target node`** over SSH. It's deliberately minimal — two probes against the actual install at `$TARGET_ROCM_PATH`:

| Sub-check | Action on failure | Catches |
|---|---|---|
| `$TARGET_ROCM_PATH` directory exists | hard fail (`exit 1`) | ROCm not installed on target, or the configured path is wrong (typo, missing trailing component, etc.) |
| `$TARGET_ROCM_PATH/bin/rocminfo` exits 0 | hard fail (via `set -euo pipefail`) | Driver not loaded, no GPU exposed, runtime libs under `$TARGET_ROCM_PATH/lib` missing/broken, group permissions wrong |
| `$TARGET_ROCM_PATH/bin/amd-smi version` exits 0 | hard fail | `amd-smi` missing from the install, ROCm SMI library mismatch, or driver/runtime broken |

Both binaries are invoked by absolute path (`${TARGET_ROCM_PATH}/bin/...`), so the workflow doesn't depend on `PATH` or `LD_LIBRARY_PATH` being set up — the binaries' baked-in RPATH/RUNPATH resolves the runtime libs from `${TARGET_ROCM_PATH}/lib`. If either probe fails, the step fails fast with the binary's own error message in the log, which is usually more diagnostic than anything the workflow could add on top.

A typical successful log (with `target_rocm_path=<ROCM_INSTALL_PATH>`):

```
=== System ===
Linux <hostname> 6.x.x-x-generic ...

=== Target ROCm path: <ROCM_INSTALL_PATH> ===

=== <ROCM_INSTALL_PATH>/bin/rocminfo ===
ROCk module version <X.Y> is loaded
HSA Agents
==========
Agent 1
  Name:                    AMD Instinct ...
  ...
Agent 2..N: (one per GPU)

=== <ROCM_INSTALL_PATH>/bin/amd-smi version ===
AMDSMI Tool: <X.Y.Z> | AMDSMI Library version: <X.Y.Z> | ROCm version: <ROCM_VERSION>

::notice::ROCm prerequisites OK on target node at <ROCM_INSTALL_PATH>
```

After install, **`Verify RVS binary library resolution on target node`** runs `ldd "$RVS_BIN"` over SSH (where `$RVS_BIN` = `/opt/rocm/extras-${ROCM_MAJOR}/bin/rvs`) and **hard-fails** the job if any library shows up as `not found`. This prevents the workflow from spending hours on `rvs -r 4` only to discover a `dlopen` error in the level log. The full `ldd` output is printed for diagnostic purposes — useful for spotting which `$TARGET_ROCM_PATH/lib/...` paths each library resolved from.

### Manual one-liner (validate a candidate target node)

To verify a node is viable before pointing the workflow at it, SSH into the candidate and run the same two probes the workflow runs (substituting `<ROCM_INSTALL_PATH>` with what you plan to set `target_rocm_path` to):

```bash
ROCM_PATH=<ROCM_INSTALL_PATH>
set -euo pipefail
[ -d "$ROCM_PATH" ] || { echo "$ROCM_PATH does not exist"; exit 1; }
"$ROCM_PATH/bin/rocminfo"
"$ROCM_PATH/bin/amd-smi" version
sudo -n true 2>/dev/null && echo "NOPASSWD sudo OK" || echo "warn: sudo requires password (install step will fail)"
echo "Target node OK"
```

If both `rocminfo` and `amd-smi version` exit zero, the workflow's prereq step will pass on this node.

## The test

Run verbatim on the target node (with `$RVS_BIN` = `/opt/rocm/extras-${ROCM_MAJOR}/bin/rvs`,
populated by the validate step — for an `amdrocm7-…` tarball this resolves
to `/opt/rocm/extras-7/bin/rvs`):

```bash
"$RVS_BIN" -r 4
```

The command is executed over SSH. The step captures full stdout/stderr to
`$REMOTE_WORK_DIR/reports/rvs_level_4.log` on the target, then the
**`Collect logs from target node`** step `scp`s the log file back to
`./reports/` on the runner. The command's exit code is propagated through
SSH and recorded in a step output, and the job is marked failed at the end
if level 4 exited non-zero.

## Test report

After RVS finishes and the log is collected, the `Build test report`
step generates `reports/SUMMARY.md` (also written to the GitHub job summary),
e.g.:

```markdown
# RVS Nightly Test Report

| Field | Value |
|---|---|
| Run | `1234567890` |
| Trigger | `schedule` |
| Target ROCm path | `<ROCM_INSTALL_PATH>` (version `<ROCM_VERSION>`) |
| Remote work dir | `/tmp/rvs-nightly-1234567890` |
| Tarball | `amdrocm7-rvs-1.4.21-288-Linux.tar.gz` |
| Source URL | `$RVS_TARBALL_INDEX_URL/amdrocm7-rvs-1.4.21-288-Linux.tar.gz` |
| RVS version | `RVS 1.4.21.0-...` |
| Overall result | **PASS** |

## Results

| Test    | Command                              | Result | Exit | Started (UTC)         | Ended (UTC)           |
|---------|--------------------------------------|:------:|-----:|-----------------------|-----------------------|
| Level 4 | `/opt/rocm/extras-7/bin/rvs -r 4`    | PASS   |    0 | 2026-05-19T15:25:12Z  | 2026-05-19T16:02:47Z  |
```

The full artifact contents:

```
rvs-nightly-report-<run_id>/
├── SUMMARY.md
└── rvs_level_4.log
```

Artifact retention is 30 days.

## GitHub runner vs target node

The `test` job runs on `${{ vars.RVS_TEST_RUNNER_LABEL || 'self-hosted' }}`,
but this runner only orchestrates — it doesn't need a GPU or ROCm. You can:

- Reuse an existing self-hosted runner that already has SSH access to the lab.
- Use a small purpose-built orchestration runner (any Linux box with `ssh`/`scp`/`curl`).
- In principle, use a GitHub-hosted `ubuntu-latest` runner — but that requires the target node to be reachable from GitHub's hosted runner IPs, which usually isn't the case for lab hosts behind a VPN/bastion.

The **target node** is what needs the GPU, ROCm, and `NOPASSWD` sudo. It does **not** need to be a registered GitHub runner.

If the chosen orchestrator runner is busy with another job, this workflow's
`concurrency:` group (`rvs-nightly-${{ github.workflow }}`,
`cancel-in-progress: false`) will queue the run rather than cancel the
running one.

## Verifying the pipeline end-to-end

After committing the workflow file to `master`, the fastest sanity check
is:

```bash
gh workflow run rvs-nightly-tests.yml
```

Watch the Actions tab for:

1. `detect` resolves a tarball URL (`Latest tarball : amdrocm<N>-rvs-…`).
2. `test` job picks up on your orchestrator runner.
3. **Validate target node configuration** prints the resolved `Target node`, `Target ROCm path`, `Remote work dir`, and `Expected RVS binary` path.
4. **Setup SSH key for target node** prints the target's `hostname` / `id` / `uptime` from the connectivity probe.
5. **Verify ROCm prerequisites on target node** prints `::notice::ROCm prerequisites OK on target node at <TARGET_ROCM_PATH>`.
6. **Install RVS on target node** prints the detected `ROCM_MAJOR`, the chosen `Target ROCm path`, and `Installed RVS at: <TARGET_ROCM_PATH>/extras-<N>/bin/rvs`.
7. **Verify RVS binary library resolution on target node** prints `::notice::RVS binary's library dependencies resolved OK on target, all from <TARGET_ROCM_PATH>`.
8. RVS level 4 step completes; the run summary shows the results table with the Level 4 row and overall PASS/FAIL.

## Debugging a failed run

| Symptom | Likely cause |
|---|---|
| `detect` job exits with `vars.RVS_TARBALL_INDEX_URL is not set` | The required variable is unset. Set it in the repo Variables (e.g. to `https://repo.amd.com/rocm/rvs/tarball/`), or pass `tarball_url` via `workflow_dispatch`. |
| `detect` job exits with "Could not resolve a tarball URL" | The index page returned no matches. Verify the URL in `vars.RVS_TARBALL_INDEX_URL` returns at least one `amdrocm*-rvs-*-Linux.tar.gz` link. |
| `test` job stuck "Queued" | No orchestrator runner online with the label in `vars.RVS_TEST_RUNNER_LABEL`. |
| **Validate target node configuration** fails: `No target node configured` | Neither `inputs.target_node` nor `secrets.RVS_TARGET_NODE` is set. Add the secret in Settings → Secrets and variables → Actions → Secrets, or pass `target_node` via `workflow_dispatch`. |
| **Setup SSH key for target node** fails: `secrets.RVS_TARGET_SSH_KEY is not set` | The required secret is missing. Add the private SSH key as a repo secret. |
| **Setup SSH key for target node** fails: `Permission denied (publickey)` | The key in `RVS_TARGET_SSH_KEY` isn't authorized on the target node for `$TARGET_USER`, or the key format is wrong. Verify by `ssh -i <key> $TARGET_USER@$TARGET_NODE hostname` from a workstation. |
| **Setup SSH key for target node** fails: `Connection timed out` / `Connection refused` | Network reachability problem between the orchestrator runner and the target node. Check firewall / VPN / bastion routing. |
| **Setup SSH key for target node** fails: `Host key verification failed` | `ssh-keyscan` couldn't pre-seed the key and `accept-new` rejected it (rare). Remove any stale entry for the target in the runner's `known_hosts`, or pre-populate it manually. |
| **Verify ROCm prerequisites on target node** fails: `<TARGET_ROCM_PATH> does not exist on the target node` | The configured `target_rocm_path` / `vars.RVS_TARGET_ROCM_PATH` doesn't point at a real directory on the target. Verify by `ssh <USER>@<HOST_OR_IP> 'ls -d <TARGET_ROCM_PATH>'`. |
| **Verify ROCm prerequisites on target node** fails: `<TARGET_ROCM_PATH>/bin/rocminfo: No such file or directory` | The install at `$TARGET_ROCM_PATH` is incomplete — missing `bin/rocminfo`. Either pick a different `target_rocm_path` (one that contains a complete ROCm distribution) or reinstall ROCm at the configured path. |
| **Verify ROCm prerequisites on target node** fails: `rocminfo` exits non-zero | Driver issue or runtime libs missing. Common causes: `amdgpu` kernel driver not loaded (`lsmod \| grep amdgpu`, then `sudo modprobe amdgpu` and check `dmesg`); `$TARGET_USER` not in `video`/`render` groups; `$TARGET_ROCM_PATH/lib` missing or broken. The `rocminfo` error message in the log usually pinpoints the cause. |
| **Verify ROCm prerequisites on target node** fails: `amd-smi version` exits non-zero | Either `amd-smi` is missing from `$TARGET_ROCM_PATH/bin/`, or the AMDSMI library under `$TARGET_ROCM_PATH/lib` is broken/missing. Reinstall or repoint `target_rocm_path` at a complete install. |
| **Verify RVS binary library resolution on target node** fails: `RVS binary has unresolved library dependencies on target` | One or more libraries showed up as `not found` in `ldd` output. The `LD_LIBRARY_PATH` (built from `$INSTALL_DIR/lib` + `$TARGET_ROCM_PATH/{lib,lib/llvm/lib,lib/rocm_sysdeps/lib}`) didn't have them, RUNPATH didn't have them, and `ldconfig` didn't have them either. Usually means `$TARGET_ROCM_PATH` is incomplete (missing libraries under `lib/`, `lib/llvm/lib/`, or `lib/rocm_sysdeps/lib/`). The step's `ldd` output above the error names the specific missing library. |
| **Install RVS on target node** fails: `Cannot parse ROCm major version from tarball name` | The tarball doesn't match `^amdrocm<digits>-`. Either pin a correctly-named tarball with `tarball_url`, or fix the upstream filename. |
| **Install RVS on target node** fails: `sudo: a password is required` | `$TARGET_USER` doesn't have `NOPASSWD` sudo on the target. Add a sudoers entry permitting `mkdir` and `tar` into `$TARGET_ROCM_PATH/extras-*` without a password. |
| **Install RVS on target node** fails: `rvs binary not found or not executable at <TARGET_ROCM_PATH>/extras-<N>/bin/rvs after install` | The tarball isn't rooted at `./bin/`, `./lib/`, etc. The extraction landed `bin/rvs` somewhere else inside `$TARGET_ROCM_PATH/extras-<N>/`. Inspect the step log (`ls -la` output) to see the actual layout; the workflow may need `--strip-components=<N>` added to the `tar` invocation. |
| **Verify RVS binary library resolution on target node** fails with `not found` | A ROCm runtime library is missing or not in `RPATH` on the target. The step prints the `ldd` output; install the matching ROCm component (typically `rocm-llvm`, `rocm-core`, `hip-runtime-amd`). |
| **Run RVS level 4 on target node** exits non-zero immediately (after both verify steps passed) | RVS plugin's own dependency missing on the target (e.g. `libpci3` on Debian). Check `rvs_level_4.log` in the artifact for the specific error. |
| **Collect logs from target node** warns: `No log files retrieved from target node` | The level steps exited so early they didn't produce any output, or `$REMOTE_WORK_DIR` was wiped. Inspect the level-step logs in the run UI for the original error. |
| Cron skipped a day | GitHub may delay or drop schedules under high load. Run once manually via `workflow_dispatch` to validate. |

## Retargeting at a different node

There are three ways to point the workflow at a different node, in increasing order of permanence:

1. **Single run, from the Actions UI:** Run workflow → fill in `target_node` and `target_rocm_path` (and optionally `target_user` / `remote_work_dir`). Anything left blank falls back to the matching repo configuration value — `target_node` and `target_user` read from the **secrets** `RVS_TARGET_NODE` / `RVS_TARGET_USER` (both masked as `***` in run logs); `target_rocm_path` / `remote_work_dir` read from repo Variables. `target_node` and `target_rocm_path` have no hard-coded defaults — the workflow fails fast if neither input nor stored value is set for either. `target_user` falls through to the orchestrator runner's local user; `remote_work_dir` falls through to `/tmp/rvs-nightly-<run_id>`.
2. **Single run, from `gh` CLI:**
   ```bash
   gh workflow run rvs-nightly-tests.yml \
     -f target_node="<host-or-ip>" \
     -f target_rocm_path="<ROCM_INSTALL_PATH>"
   ```
3. **Permanent change:** update repo secrets `RVS_TARGET_NODE` / `RVS_TARGET_USER` and repo variable `RVS_TARGET_ROCM_PATH` (and optionally repo variable `RVS_REMOTE_WORK_DIR`). All subsequent scheduled and manual runs will pick these up unless an input overrides them.

The same `RVS_TARGET_SSH_KEY` secret is reused across nodes — make sure the
public counterpart of that key is added to `$TARGET_USER`'s `~/.ssh/authorized_keys`
on **every** node you intend to point the workflow at.

## References

- [RVS source](../../README.md)
- [`build-relocatable-packages.yml`](./build-relocatable-packages.yml) and [`README_BUILD_PACKAGES.md`](./README_BUILD_PACKAGES.md) — the upstream packaging pipeline that produces these tarballs
- [GitHub Actions: scheduled events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)
- [GitHub Actions: encrypted secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions) — for `RVS_TARGET_SSH_KEY`
