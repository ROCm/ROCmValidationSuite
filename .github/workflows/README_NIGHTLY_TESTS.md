# RVS Nightly Tests Workflow

This document describes [`.github/workflows/rvs-nightly-tests.yml`](./rvs-nightly-tests.yml),
which picks up the **latest RVS tarball** from the index URL configured in
`vars.RVS_TARBALL_INDEX_URL` (e.g. `https://repo.amd.com/rocm/rvs/tarball/`)
once per day, copies it to a **configurable remote target node** over SSH,
installs it there, and runs RVS levels 3 and 4 on that node.

The GitHub Actions runner ("RVS Runner") is only an **orchestrator** — it
doesn't need a GPU or ROCm installed locally. All RVS install, binary
verification, and `rvs -r 3` / `rvs -r 4` execution happens on the target
node. The target node is configurable so the same workflow can be pointed
at any GPU host without code changes.

## What it does

```
schedule (or manual)
    │
    ▼
detect      [ubuntu-latest]
    │  curl $RVS_TARBALL_INDEX_URL
    │  → grep amdrocm*-rvs-*-Linux.tar.gz, sort -V, pick newest
    │  → compare with cached marker (skip if unchanged)
    ▼
test        [GitHub runner = orchestrator]    ──ssh──▶  [target node = GPU host]
    │  validate target_node / target_user                                         │
    │  write SSH key + config to $RUNNER_TEMP                                     │
    │  ssh rvs-target hostname; id                ─────────────────────────────▶  │ connectivity check
    │  ssh rvs-target <prereq script>             ─────────────────────────────▶  │ /opt/rocm, rocm-smi, rocminfo, libs
    │  curl <tarball URL>                         → ./pkg/<file>.tar.gz           │
    │  scp ./pkg/<file>.tar.gz rvs-target:…       ─────────────────────────────▶  │ $REMOTE_WORK_DIR/pkg/
    │  ssh rvs-target <install script>            ─────────────────────────────▶  │ sudo -n tar -xzf → /opt/rocm/extras-<N>/
    │  ssh rvs-target <ldd script>                ─────────────────────────────▶  │ verify RVS lib resolution
    │  ssh rvs-target rvs -r 3                    ─────────────────────────────▶  │ writes $REMOTE_WORK_DIR/reports/rvs_level_3.log
    │  ssh rvs-target rvs -r 4                    ─────────────────────────────▶  │ writes $REMOTE_WORK_DIR/reports/rvs_level_4.log
    │  scp rvs-target:…/reports/*.log ./reports/  ◀─────────────────────────────  │
    │  build Markdown SUMMARY.md                  → GitHub job summary + artifact │
    │  ssh rvs-target rm -rf $REMOTE_WORK_DIR     ─────────────────────────────▶  │ cleanup
    ▼
artifact: rvs-nightly-report-<run_id>
```

## Triggers

| Trigger | Cadence | What fires |
|---|---|---|
| `schedule` | `0 15 * * *` UTC daily (08:00 PST / 07:00 PDT) | Polls the tarball index. If the latest filename matches the previous run's, the run is **skipped** to avoid re-testing the same package. |
| `workflow_dispatch` | Manual | Always runs. Supports overriding the tarball URL, forcing a re-run, and **retargeting at any node** without editing the workflow. |

The cron deliberately runs after AMD's typical nightly publish window;
adjust the cron string in the workflow if your publish cadence is different.

## Manual dispatch inputs

| Input | Default | Description |
|---|---|---|
| `tarball_url` | _(empty)_ | If set, the workflow downloads this exact URL instead of scraping the index. Useful for re-running an older build. |
| `force` | `false` | When `true`, runs even if the latest tarball filename matches the cached marker. |
| `target_node` | _(empty → `vars.RVS_TARGET_NODE`)_ | Hostname or IP of the node to install RVS on and run tests against. **This is the variable that retargets the test execution.** |
| `target_user` | _(empty → `vars.RVS_TARGET_USER`; if both are unset, SSH defaults to the orchestrator runner's local user)_ | SSH user on the target node. Must have `NOPASSWD` sudo on the target (see prerequisites). |
| `remote_work_dir` | _(empty → `vars.RVS_REMOTE_WORK_DIR`, then `/tmp/rvs-nightly-<run_id>`)_ | Working dir on the target node where the tarball is staged, logs are written, and which gets `rm -rf`'d at the end. |

Workflow inputs **win over repo variables**, so individual `workflow_dispatch`
runs can be retargeted from the Actions UI without changing repo settings.

Example — point a single run at a specific node:

```bash
gh workflow run rvs-nightly-tests.yml \
  -f tarball_url="<INDEX_URL>/amdrocm7-rvs-1.4.21-288-Linux.tar.gz" \
  -f target_node="x.x.x.x" \
  -f target_user="userID" \
  -f force=true
```

Example — keep the default tarball but run on a different host today:

```bash
gh workflow run rvs-nightly-tests.yml -f target_node="e17u13.maas"
```

## Repository configuration

**Variables** (Settings → Secrets and variables → Actions → Variables):

| Name | Required? | Purpose |
|---|---|---|
| `RVS_TARBALL_INDEX_URL` | **Required** | Directory listing scraped for the latest tarball, e.g. `https://repo.amd.com/rocm/rvs/tarball/`. No fallback — the workflow fails fast if unset and no `tarball_url` input is supplied. |
| `RVS_TARGET_NODE` | **Required** *(unless every run sets `target_node` input)* | Default hostname/IP of the node where RVS is installed and tests run. Workflow fails fast on `schedule` if neither this var nor `target_node` input is set. |
| `RVS_TARGET_USER` | optional (no hard-coded default) | SSH user on the target node. If unset and `target_user` input is empty, the SSH client falls back to the orchestrator runner's local user — set this var explicitly to avoid surprises. |
| `RVS_REMOTE_WORK_DIR` | optional (default `/tmp/rvs-nightly-<run_id>`) | Working dir on the target node. Cleared with `rm -rf` at the end of the job. |
| `RVS_TEST_RUNNER_LABEL` | optional (default `self-hosted`) | Label of the GitHub runner that orchestrates the workflow. The runner doesn't need a GPU or ROCm — it just needs `ssh`, `scp`, and `curl`. |

**Secrets** (Settings → Secrets and variables → Actions → Secrets):

| Name | Required? | Purpose |
|---|---|---|
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
(e.g. `amdrocm7-rvs-1.4.21-…-Linux.tar.gz` → `7`), writes the derived
paths to `$GITHUB_ENV`, then SSHes into the target with those values
exported so the install runs against the matching `extras-<major>`
directory. The same workflow handles ROCm 6, 7, etc. without code changes:

```bash
# On the runner (Validate target node configuration step):
# Parsed from $TARBALL_NAME via [[ "$TARBALL_NAME" =~ ^amdrocm([0-9]+)- ]]
ROCM_MAJOR=7
INSTALL_DIR=/opt/rocm/extras-${ROCM_MAJOR}    # /opt/rocm/extras-7
RVS_BIN=${INSTALL_DIR}/bin/rvs

# On the target node (Install RVS on target node step), via SSH:
sudo -n mkdir -p "$INSTALL_DIR"
sudo -n tar -xzf "$REMOTE_WORK_DIR/pkg/<tarball>.tar.gz" -C "$INSTALL_DIR"

export LD_LIBRARY_PATH="$INSTALL_DIR/lib:/install/lib:/install/lib/rocm_sysdeps/:/install/lib/llvm/lib:${LD_LIBRARY_PATH:-}"

"$RVS_BIN" --version
```

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
- `$TARGET_USER` has **`NOPASSWD` sudo** for `mkdir` + `tar` into `/opt/rocm/extras-<major>` — the install step uses `sudo -n` and aborts otherwise.
- ROCm base stack already installed at `/opt/rocm/` with a major version that matches the tarball name (e.g. `amdrocm7-rvs-…` requires ROCm 7.x). The TGZ only ships RVS, not a full ROCm install; the RVS binary depends on ROCm libraries resolved via `RPATH` to the system ROCm.
- Working kernel driver (`amdgpu`) and at least one GPU enumerated by `rocm-smi` and `rocminfo`.

## Pre-flight ROCm checks

Before downloading or installing the tarball, the workflow runs **`Verify ROCm prerequisites on target node`** — the same checks as before, but executed remotely via SSH on the target node, not on the GitHub runner. The step accumulates failures and reports all of them in a single log so the cause is obvious:

| Sub-check | Action on failure | Catches |
|---|---|---|
| `/opt/rocm/` directory exists | hard fail (`exit 1`) | ROCm not installed on target |
| Read ROCm version from `/opt/rocm/.info/version` (fallback `/opt/rocm/share/doc/rocm-core/version`) | warn if missing | Partial install, dev pkgs only |
| Major version match (regex on tarball name vs target's ROCm version) | hard fail on mismatch | RVS-7 tarball on a ROCm-6 host |
| `rocm-smi --showid` runs and exits 0 | hard fail | `amdgpu` driver not loaded, or `rocm-smi` missing |
| `rocminfo` enumerates ≥ 1 GPU | hard fail | "ROCm installed but no GPU exposed" (group/permissions, BIOS IOMMU, etc.) |
| `libhsa-runtime64.so.1` and `libamdhip64.so` present in `ldconfig -p` | warn only | Missing/broken `ldconfig` cache (RPATH usually compensates) |

A typical successful log:

```
=== /opt/rocm/ ===
/opt/rocm -> /opt/rocm-7.11.0
=== ROCm version ===
/opt/rocm/.info/version : 7.11.0
=== Major version match (tarball vs target node) ===
tarball expects ROCm major : 7
target has ROCm major      : 7
=== rocm-smi ===
GPU[0]  : ID  : 0x74a1
=== rocminfo (GPU enumeration) ===
GPU agents enumerated: 8
=== Key ROCm runtime libraries ===
  OK    : libhsa-runtime64.so.1
  OK    : libamdhip64.so
::notice::ROCm prerequisites OK on target node (version 7.11.0)
```

After install, **`Verify RVS binary library resolution on target node`** runs `ldd "$RVS_BIN"` over SSH (where `$RVS_BIN` = `/opt/rocm/extras-${ROCM_MAJOR}/bin/rvs`) and fails the job if any library shows up as `not found` — preventing the workflow from spending hours on `rvs -r 3`/`-r 4` only to discover a `dlopen` error in the level log.

### Manual one-liner (validate a candidate target node)

To verify a node is viable before pointing the workflow at it, SSH into the candidate and run:

```bash
set -uo pipefail
[ -d /opt/rocm ] || { echo "no /opt/rocm"; exit 1; }
echo "version: $(cat /opt/rocm/.info/version 2>/dev/null || \
                 cat /opt/rocm/share/doc/rocm-core/version 2>/dev/null || \
                 echo unknown)"
rocm-smi --showid >/dev/null && echo "rocm-smi OK"
N=$(rocminfo 2>/dev/null | grep -c 'Device Type:.*GPU' || echo 0)
[ "$N" -ge 1 ] && echo "rocminfo OK ($N GPU agents)" || { echo "no GPU"; exit 1; }
ldconfig -p | grep -q libhsa-runtime64.so.1 && echo "libhsa-runtime64 OK" || echo "warn: libhsa-runtime64 missing"
sudo -n true 2>/dev/null && echo "NOPASSWD sudo OK" || echo "warn: sudo requires password (install step will fail)"
echo "Target node OK"
```

## The tests

Run verbatim on the target node (with `$RVS_BIN` = `/opt/rocm/extras-${ROCM_MAJOR}/bin/rvs`,
populated by the validate step — for an `amdrocm7-…` tarball this resolves
to `/opt/rocm/extras-7/bin/rvs`):

```bash
"$RVS_BIN" -r 3
"$RVS_BIN" -r 4
```

Both commands are executed sequentially over SSH. Each step captures full
stdout/stderr to `$REMOTE_WORK_DIR/reports/rvs_level_<N>.log` on the
target, then the **`Collect logs from target node`** step `scp`s the
`*.log` files back to `./reports/` on the runner. Each command's exit
code is propagated through SSH and recorded in a step output. Neither
failure short-circuits the other — both levels always run, and the job
is marked failed at the end if either exited non-zero.

## Test report

After both RVS commands finish and logs are collected, the `Build test report`
step generates `reports/SUMMARY.md` (also written to the GitHub job summary),
e.g.:

```markdown
# RVS Nightly Test Report

| Field | Value |
|---|---|
| Run | `1234567890` |
| Trigger | `schedule` |
| Orchestrator (GitHub runner) | `gha-orchestrator-01` |
| Target node (test execution) | `hostname` (`user@x.x.x.x`) |
| Remote work dir | `/tmp/rvs-nightly-1234567890` |
| Tarball | `amdrocm7-rvs-1.4.21-288-Linux.tar.gz` |
| Source URL | `$RVS_TARBALL_INDEX_URL/amdrocm7-rvs-1.4.21-288-Linux.tar.gz` |
| RVS version | `RVS 1.4.21.0-...` |
| Overall result | **PASS** |

## Results

| Test    | Command                                              | Result | Exit | Started (UTC)         | Ended (UTC)           |
|---------|------------------------------------------------------|:------:|-----:|-----------------------|-----------------------|
| Level 3 | `/opt/rocm/extras-7/bin/rvs -r 3` (on `hostname`)      | PASS   |    0 | 2026-05-19T15:01:00Z  | 2026-05-19T15:25:11Z  |
| Level 4 | `/opt/rocm/extras-7/bin/rvs -r 4` (on `hostname`)      | PASS   |    0 | 2026-05-19T15:25:12Z  | 2026-05-19T16:02:47Z  |
```

The full artifact contents:

```
rvs-nightly-report-<run_id>/
├── SUMMARY.md
├── rvs_level_3.log
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
gh workflow run rvs-nightly-tests.yml -f force=true
```

Watch the Actions tab for:

1. `detect` resolves a tarball URL (`Latest tarball : amdrocm<N>-rvs-…`).
2. `test` job picks up on your orchestrator runner.
3. **Validate target node configuration** prints the resolved `Target node`, `Remote work dir`, and `Expected RVS binary` path.
4. **Setup SSH key for target node** prints the target's `hostname` / `id` / `uptime` from the connectivity probe.
5. **Verify ROCm prerequisites on target node** prints `::notice::ROCm prerequisites OK on target node`.
6. **Install RVS on target node** prints the detected `ROCM_MAJOR` and `Installed RVS at: /opt/rocm/extras-<N>/bin/rvs`.
7. **Verify RVS binary library resolution on target node** prints `::notice::RVS binary's library dependencies resolved OK on target`.
8. Two RVS level steps complete; the run summary shows the results table with both **Orchestrator** and **Target node** rows.

## Debugging a failed run

| Symptom | Likely cause |
|---|---|
| `detect` job exits with `vars.RVS_TARBALL_INDEX_URL is not set` | The required variable is unset. Set it in the repo Variables (e.g. to `https://repo.amd.com/rocm/rvs/tarball/`), or pass `tarball_url` via `workflow_dispatch`. |
| `detect` job exits with "Could not resolve a tarball URL" | The index page returned no matches. Verify the URL in `vars.RVS_TARBALL_INDEX_URL` returns at least one `amdrocm*-rvs-*-Linux.tar.gz` link. |
| `test` job stuck "Queued" | No orchestrator runner online with the label in `vars.RVS_TEST_RUNNER_LABEL`. |
| **Validate target node configuration** fails: `No target node configured` | Neither `inputs.target_node` nor `vars.RVS_TARGET_NODE` is set. Set the variable, or pass `target_node` via `workflow_dispatch`. |
| **Setup SSH key for target node** fails: `secrets.RVS_TARGET_SSH_KEY is not set` | The required secret is missing. Add the private SSH key as a repo secret. |
| **Setup SSH key for target node** fails: `Permission denied (publickey)` | The key in `RVS_TARGET_SSH_KEY` isn't authorized on the target node for `$TARGET_USER`, or the key format is wrong. Verify by `ssh -i <key> $TARGET_USER@$TARGET_NODE hostname` from a workstation. |
| **Setup SSH key for target node** fails: `Connection timed out` / `Connection refused` | Network reachability problem between the orchestrator runner and the target node. Check firewall / VPN / bastion routing. |
| **Setup SSH key for target node** fails: `Host key verification failed` | `ssh-keyscan` couldn't pre-seed the key and `accept-new` rejected it (rare). Remove any stale entry for the target in the runner's `known_hosts`, or pre-populate it manually. |
| **Verify ROCm prerequisites on target node** fails: `/opt/rocm/ does not exist` | Target node has no ROCm install. Install ROCm or pick a different `target_node`. |
| **Verify ROCm prerequisites on target node** fails: `ROCm major version mismatch` | The tarball is for ROCm `<X>` but the target has ROCm `<Y>`. Either pin a matching tarball with `tarball_url`, or upgrade the target's ROCm. |
| **Verify ROCm prerequisites on target node** fails: `rocm-smi … exited non-zero` | `amdgpu` kernel driver is not loaded on the target. SSH in and `lsmod \| grep amdgpu`, then `sudo modprobe amdgpu` and check `dmesg`. |
| **Verify ROCm prerequisites on target node** fails: `rocminfo enumerated 0 GPU agents` | Driver loaded but no GPU exposed to `$TARGET_USER`. Add the user to `video` and `render` groups, log out and back in. |
| **Install RVS on target node** fails: `Cannot parse ROCm major version from tarball name` | The tarball doesn't match `^amdrocm<digits>-`. Either pin a correctly-named tarball with `tarball_url`, or fix the upstream filename. |
| **Install RVS on target node** fails: `sudo: a password is required` | `$TARGET_USER` doesn't have `NOPASSWD` sudo on the target. Add a sudoers entry permitting `mkdir` and `tar` into `/opt/rocm/extras-*` without a password. |
| **Install RVS on target node** fails: `rvs binary not found or not executable at /opt/rocm/extras-<N>/bin/rvs after install` | The tarball isn't rooted at `./bin/`, `./lib/`, etc. The extraction landed `bin/rvs` somewhere else inside `/opt/rocm/extras-<N>/`. Inspect the step log (`ls -la` output) to see the actual layout; the workflow may need `--strip-components=<N>` added to the `tar` invocation. |
| **Verify RVS binary library resolution on target node** fails with `not found` | A ROCm runtime library is missing or not in `RPATH` on the target. The step prints the `ldd` output; install the matching ROCm component (typically `rocm-llvm`, `rocm-core`, `hip-runtime-amd`). |
| **Run RVS level N on target node** exits non-zero immediately (after both verify steps passed) | RVS plugin's own dependency missing on the target (e.g. `libpci3` on Debian). Check the level log for the specific error. |
| **Collect logs from target node** warns: `No log files retrieved from target node` | The level steps exited so early they didn't produce any output, or `$REMOTE_WORK_DIR` was wiped. Inspect the level-step logs in the run UI for the original error. |
| Cron skipped a day | GitHub may delay or drop schedules under high load. Run once manually with `force=true` to validate. |

## Retargeting at a different node

There are three ways to point the workflow at a different node, in increasing order of permanence:

1. **Single run, from the Actions UI:** Run workflow → fill in `target_node` (and optionally `target_user` / `remote_work_dir`). Anything left blank falls back to the matching repo variable. `target_node` has no hard-coded default (workflow fails fast), `target_user` falls through to the orchestrator runner's local user, and `remote_work_dir` falls through to `/tmp/rvs-nightly-<run_id>`. This is the right path for "test this tarball on host X today".
2. **Single run, from `gh` CLI:**
   ```bash
   gh workflow run rvs-nightly-tests.yml -f target_node="<host-or-ip>"
   ```
3. **Permanent change:** update repo variable `RVS_TARGET_NODE` (and optionally `RVS_TARGET_USER` / `RVS_REMOTE_WORK_DIR`). All subsequent scheduled and manual runs will pick this up unless an input overrides it.

The same `RVS_TARGET_SSH_KEY` secret is reused across nodes — make sure the
public counterpart of that key is added to `$TARGET_USER`'s `~/.ssh/authorized_keys`
on **every** node you intend to point the workflow at.

## References

- [RVS source](../../README.md)
- [`build-relocatable-packages.yml`](./build-relocatable-packages.yml) and [`README_BUILD_PACKAGES.md`](./README_BUILD_PACKAGES.md) — the upstream packaging pipeline that produces these tarballs
- [GitHub Actions: scheduled events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)
- [GitHub Actions: encrypted secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions) — for `RVS_TARGET_SSH_KEY`
