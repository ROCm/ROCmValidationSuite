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
| `target_rocm_path` | _(empty → `vars.RVS_TARGET_ROCM_PATH`, then `/opt/rocm`)_ | Which ROCm install on the target node to run RVS against, e.g. `/opt/rocm-7.13`. RVS is installed into `<this>/extras-<major>/` and `LD_LIBRARY_PATH` is set to `<this>/lib:...` so it wins over the binary's RUNPATH-derived `/opt/rocm/lib`. Use this to pick a specific version on hosts with multiple ROCm installs side-by-side. |

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

Example — point at a specific ROCm install on a multi-ROCm host:

```bash
gh workflow run rvs-nightly-tests.yml \
  -f target_node="10.245.128.41" \
  -f target_rocm_path="/opt/rocm-7.13"
```

## Repository configuration

**Variables** (Settings → Secrets and variables → Actions → Variables):

| Name | Required? | Purpose |
|---|---|---|
| `RVS_TARBALL_INDEX_URL` | **Required** | Directory listing scraped for the latest tarball, e.g. `https://repo.amd.com/rocm/rvs/tarball/`. No fallback — the workflow fails fast if unset and no `tarball_url` input is supplied. |
| `RVS_TARGET_NODE` | **Required** *(unless every run sets `target_node` input)* | Default hostname/IP of the node where RVS is installed and tests run. Workflow fails fast on `schedule` if neither this var nor `target_node` input is set. |
| `RVS_TARGET_USER` | optional (no hard-coded default) | SSH user on the target node. If unset and `target_user` input is empty, the SSH client falls back to the orchestrator runner's local user — set this var explicitly to avoid surprises. |
| `RVS_REMOTE_WORK_DIR` | optional (default `/tmp/rvs-nightly-<run_id>`) | Working dir on the target node. Cleared with `rm -rf` at the end of the job. |
| `RVS_TARGET_ROCM_PATH` | optional (default `/opt/rocm`) | ROCm install on the target node that RVS should run against, e.g. `/opt/rocm-7.13`. Required on hosts where `/opt/rocm` symlinks to the "wrong" version — without it the binary's RUNPATH will silently pull libs from the system default. The `Verify RVS binary library resolution` step fails fast if any lib resolves from a different ROCm install. |
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
# TARGET_ROCM_PATH comes from inputs.target_rocm_path / vars.RVS_TARGET_ROCM_PATH
# (default: /opt/rocm). E.g. /opt/rocm-7.13 on a multi-ROCm host.
INSTALL_DIR=${TARGET_ROCM_PATH}/extras-${ROCM_MAJOR}   # /opt/rocm-7.13/extras-7
RVS_BIN=${INSTALL_DIR}/bin/rvs

# On the target node (Install RVS on target node step), via SSH:
# sudo is used only when $INSTALL_DIR isn't user-writable (e.g. under
# /opt/rocm-*). For TheRock tarball installs in $HOME, plain mkdir/tar is used.
mkdir -p "$INSTALL_DIR"     # (with `sudo -n` if needed)
tar -xzf "$REMOTE_WORK_DIR/pkg/<tarball>.tar.gz" -C "$INSTALL_DIR"

# LD_LIBRARY_PATH for the rvs binary is currently HARDCODED to match the
# layout the RVS team uses manually. If you point target_rocm_path at a
# non-/opt/rocm install (e.g. TheRock under $HOME), update the literal
# strings in the workflow's install / verify-ldd / level-3 / level-4 /
# report steps to match — see the "Hardcoded LD_LIBRARY_PATH" note below.
export LD_LIBRARY_PATH="/opt/rocm/extras-7/lib:/install/lib:/install/lib/rocm_sysdeps/:/install/lib/llvm/lib:${LD_LIBRARY_PATH:-}"

"$RVS_BIN" --version
```

### Hardcoded `LD_LIBRARY_PATH` (current temporary state)

To stay 1:1 with the manual sequence the RVS team uses today, the workflow
hardcodes the runtime `LD_LIBRARY_PATH` for every step that executes the
`rvs` binary (install, ldd-verify, level 3, level 4, and the report's
`--version` query) to exactly:

```bash
LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:/install/lib:/install/lib/rocm_sysdeps/:/install/lib/llvm/lib:$LD_LIBRARY_PATH
```

That mirrors the manual flow:

```bash
sudo mkdir -p /opt/rocm/extras-7
sudo tar -xzf amdrocm7-rvs-1.4.21-288-Linux.tar.gz -C /opt/rocm/extras-7
export LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:/install/lib:/install/lib/rocm_sysdeps/:/install/lib/llvm/lib:$LD_LIBRARY_PATH
```

Consequences and trade-offs:

- The default `target_rocm_path=/opt/rocm` case "just works" — `INSTALL_DIR` resolves to `/opt/rocm/extras-7`, which is exactly what the hardcoded path points at.
- If you set `target_rocm_path` to something else (e.g. `/opt/rocm-7.13` or a TheRock home-dir install), RVS will install into the new path but the hardcoded `LD_LIBRARY_PATH` will still point at `/opt/rocm/extras-7/lib`. The `Verify RVS binary library resolution` step will catch this and fail with a clear error listing the offending paths — so it's loud, not silent. To actually retarget at a different ROCm, edit the five `LD_LIBRARY_PATH=` lines in `.github/workflows/rvs-nightly-tests.yml` (grep for `/opt/rocm/extras-7/lib`) to the corresponding paths under the new install.
- This is a deliberate temporary state. When the team is ready for full multi-ROCm support, replace those five literals with `"${TARGET_ROCM_PATH}/lib/rocm_sysdeps/lib:${TARGET_ROCM_PATH}/lib/llvm/lib:${TARGET_ROCM_PATH}/lib:${INSTALL_DIR}/lib/rocm_sysdeps/lib:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"` — the prereq-check step already uses that dynamic form as a reference.

### Multi-ROCm hosts (`/opt/rocm-7.0.2`, `/opt/rocm-7.13`, …)

When the target node has several versioned ROCm installs side-by-side,
`/opt/rocm` is typically a symlink to one of them and the RVS binary's
RUNPATH resolves to `/opt/rocm/lib`. Without `target_rocm_path`/
`RVS_TARGET_ROCM_PATH`, the dynamic linker will silently pull libraries
from whichever ROCm the symlink happens to point at — which may not be
the one you want to test. Symptom: `ldd $RVS_BIN` shows paths like
`/opt/rocm-7.2.0/lib/libamd_smi.so.26` when you expected 7.13.

Find the right path on the node:

```bash
ssh urtiwari@<target> '
  ls -d /opt/rocm* 2>/dev/null
  echo "/opt/rocm -> $(readlink /opt/rocm 2>/dev/null || echo "(not a symlink)")"
  for d in /opt/rocm-*; do
    v=$(cat "$d/.info/version" 2>/dev/null || echo "?")
    printf "  %-30s version=%s\n" "$d" "$v"
  done
'
```

Then either set `vars.RVS_TARGET_ROCM_PATH=/opt/rocm-7.13` once for all
runs, or pass `-f target_rocm_path=/opt/rocm-7.13` for a single dispatch.
The workflow's `Verify RVS binary library resolution` step explicitly
fails the job if any resolved library lives in a `/opt/rocm-*` other
than `TARGET_ROCM_PATH`, so cross-contamination can't go unnoticed.

### TheRock-style tarball ROCm installs (no `/opt/rocm-<ver>`)

The [official ROCm 7.12+ "tarball" install method](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html?fam=instinct&gpu=mi350x&os=ubuntu&os-version=24.04&i=tar)
doesn't drop ROCm under `/opt/rocm-<ver>/`. Instead you extract a
single distribution archive (e.g.
`therock-dist-linux-gfx94X-dcgpu-7.13.0.dev0+<sha>.tar.gz`) into an
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
   ssh urtiwari@<target> '
     for cand in ~/install ~/*/install ~/rocm*/install /opt/therock*/install; do
       [ -x "$cand/bin/rocminfo" ] && echo "  found ROCm install: $cand"
     done
   '
   ```
2. **Pass that absolute path as `target_rocm_path`.** Example:
   ```bash
   gh workflow run rvs-nightly-tests.yml \
     -f target_node=10.245.128.41 \
     -f target_rocm_path=/home/urtiwari/rocm-7.13/install
   ```

What the workflow handles automatically for tarball installs:

- The install step **skips `sudo`** when the destination is user-writable (TheRock installs in `$HOME` don't need root); only system `/opt/rocm-*` installs trigger `sudo -n`.
- The prereq check uses whatever `rocm-smi` / `rocminfo` are first on `PATH` (system package installs land them in `/usr/bin/`; versioned installs typically register them via `/etc/profile.d/rocm.sh`). The binaries' RPATH locates the matching ROCm runtime libs, so no `LD_LIBRARY_PATH` setup is needed for the prereq tools.
- Tarball installs typically don't ship a `.info/version` file, so the version-string row in the report may show `unknown` — the major-version cross-check is automatically skipped in that case (no false failure).

**What you have to do manually right now** to actually run RVS against a TheRock-style install:

Because `LD_LIBRARY_PATH` for the rvs binary itself is currently [hardcoded](#hardcoded-ld_library_path-current-temporary-state) to `/opt/rocm/extras-7/lib:/install/lib:/install/lib/rocm_sysdeps/:/install/lib/llvm/lib:...`, simply pointing `target_rocm_path` at e.g. `/home/urtiwari/rocm-7.13/install` is **not enough** — RVS will install into the right place, but the loader won't see the libs and the `Verify RVS binary library resolution` step will hard-fail. To use a non-`/opt/rocm` ROCm install today, edit the five `LD_LIBRARY_PATH=` literals in `.github/workflows/rvs-nightly-tests.yml` (grep for `/opt/rocm/extras-7/lib`) to point at the matching paths under your install. Or revert those literals to the dynamic form documented above so the workflow follows `TARGET_ROCM_PATH` automatically.

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
| `$TARGET_ROCM_PATH` directory exists | hard fail (`exit 1`, also prints the available `/opt/rocm*` installs to ease debugging) | ROCm not installed on target, or the configured path is wrong (e.g. typo, `/opt/rocm-7.13.0` vs `/opt/rocm-7.13`) |
| Read ROCm version from `$TARGET_ROCM_PATH/.info/version` (fallback `share/doc/rocm-core/version`) | warn if missing | Partial install, dev pkgs only |
| Major version match (regex on tarball name vs target's ROCm version) | hard fail on mismatch | RVS-7 tarball pointed at a ROCm-6 install |
| `rocm-smi --showid` runs and exits 0 | hard fail | `amdgpu` driver not loaded, or `rocm-smi` missing |
| `rocminfo` enumerates ≥ 1 GPU | hard fail | "ROCm installed but no GPU exposed" (group/permissions, BIOS IOMMU, etc.) |
| `libhsa-runtime64.so.1` and `libamdhip64.so` present in `ldconfig -p` | warn only | Covers both system-package installs (libs in `/usr/lib/x86_64-linux-gnu`) and properly-registered versioned `/opt/rocm-*` installs. Warn-only because TheRock-style tarball installs that aren't registered with `ldconfig` still resolve at runtime via the binaries' RPATH. |

A typical successful log (with `target_rocm_path=/opt/rocm-7.13`):

```
=== Target ROCm path: /opt/rocm-7.13 ===
/opt/rocm-7.13 -> /opt/rocm-7.13.0
=== ROCm version (under /opt/rocm-7.13) ===
/opt/rocm-7.13/.info/version : 7.13.0
=== Major version match (tarball vs target node) ===
tarball expects ROCm major : 7
target has ROCm major      : 7
=== rocm-smi ===
GPU[0]  : ID  : 0x74a1
=== rocminfo (GPU enumeration) ===
GPU agents enumerated: 8
=== Key ROCm runtime libraries (via ldconfig) ===
  OK    : libhsa-runtime64.so.1  (/opt/rocm-7.13/lib/libhsa-runtime64.so.1)
  OK    : libamdhip64.so  (/opt/rocm-7.13/lib/libamdhip64.so)
::notice::ROCm prerequisites OK on target node at /opt/rocm-7.13 (version 7.13.0)
```

After install, **`Verify RVS binary library resolution on target node`** runs `ldd "$RVS_BIN"` over SSH (where `$RVS_BIN` = `${TARGET_ROCM_PATH}/extras-${ROCM_MAJOR}/bin/rvs`) and fails the job in two cases:

- **Unresolved deps:** any library shows up as `not found` — prevents the workflow from spending hours on `rvs -r 3`/`-r 4` only to discover a `dlopen` error in the level log.
- **Wrong-ROCm contamination:** any resolved library lives in a `/opt/rocm-<X>` whose `realpath` differs from `$TARGET_ROCM_PATH` (e.g. `ldd` shows `/opt/rocm-7.2.0/lib/libamd_smi.so.26` while you asked for `/opt/rocm-7.13`). The step prints which paths leaked and suggests setting `target_rocm_path` to fix it.

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
| Target ROCm path | `/opt/rocm-7.13` (version `7.13.0`) |
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
3. **Validate target node configuration** prints the resolved `Target node`, `Target ROCm path`, `Remote work dir`, and `Expected RVS binary` path.
4. **Setup SSH key for target node** prints the target's `hostname` / `id` / `uptime` from the connectivity probe.
5. **Verify ROCm prerequisites on target node** prints `::notice::ROCm prerequisites OK on target node at <TARGET_ROCM_PATH>`.
6. **Install RVS on target node** prints the detected `ROCM_MAJOR`, the chosen `Target ROCm path`, and `Installed RVS at: <TARGET_ROCM_PATH>/extras-<N>/bin/rvs`.
7. **Verify RVS binary library resolution on target node** prints `::notice::RVS binary's library dependencies resolved OK on target, all from <TARGET_ROCM_PATH>`.
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
| **Verify ROCm prerequisites on target node** fails: `<TARGET_ROCM_PATH> does not exist on the target node` | The configured `target_rocm_path` / `vars.RVS_TARGET_ROCM_PATH` doesn't point at a real directory on the target. The step prints all `/opt/rocm*` installs that *do* exist — pick one of those (or fix the symlink). |
| **Verify ROCm prerequisites on target node** fails: `ROCm major version mismatch` | The tarball is for ROCm `<X>` but `$TARGET_ROCM_PATH` is a ROCm `<Y>` install. Either pin a matching tarball with `tarball_url`, or change `target_rocm_path`. |
| **Verify RVS binary library resolution on target node** fails: `RVS is resolving libraries from a ROCm install OTHER than ...` | RVS picked up libraries from a different ROCm than the one you targeted (typical on hosts where `/opt/rocm` symlinks to the "wrong" version). The step lists the offending paths. Set `target_rocm_path` to the specific versioned path you want (e.g. `/opt/rocm-7.13`). |
| **Verify ROCm prerequisites on target node** fails: `rocm-smi … exited non-zero` | `amdgpu` kernel driver is not loaded on the target. SSH in and `lsmod \| grep amdgpu`, then `sudo modprobe amdgpu` and check `dmesg`. |
| **Verify ROCm prerequisites on target node** fails: `rocminfo enumerated 0 GPU agents` | Driver loaded but no GPU exposed to `$TARGET_USER`. Add the user to `video` and `render` groups, log out and back in. |
| **Install RVS on target node** fails: `Cannot parse ROCm major version from tarball name` | The tarball doesn't match `^amdrocm<digits>-`. Either pin a correctly-named tarball with `tarball_url`, or fix the upstream filename. |
| **Install RVS on target node** fails: `sudo: a password is required` | `$TARGET_USER` doesn't have `NOPASSWD` sudo on the target. Add a sudoers entry permitting `mkdir` and `tar` into `$TARGET_ROCM_PATH/extras-*` without a password. |
| **Install RVS on target node** fails: `rvs binary not found or not executable at <TARGET_ROCM_PATH>/extras-<N>/bin/rvs after install` | The tarball isn't rooted at `./bin/`, `./lib/`, etc. The extraction landed `bin/rvs` somewhere else inside `$TARGET_ROCM_PATH/extras-<N>/`. Inspect the step log (`ls -la` output) to see the actual layout; the workflow may need `--strip-components=<N>` added to the `tar` invocation. |
| **Verify RVS binary library resolution on target node** fails with `not found` | A ROCm runtime library is missing or not in `RPATH` on the target. The step prints the `ldd` output; install the matching ROCm component (typically `rocm-llvm`, `rocm-core`, `hip-runtime-amd`). |
| **Run RVS level N on target node** exits non-zero immediately (after both verify steps passed) | RVS plugin's own dependency missing on the target (e.g. `libpci3` on Debian). Check the level log for the specific error. |
| **Collect logs from target node** warns: `No log files retrieved from target node` | The level steps exited so early they didn't produce any output, or `$REMOTE_WORK_DIR` was wiped. Inspect the level-step logs in the run UI for the original error. |
| Cron skipped a day | GitHub may delay or drop schedules under high load. Run once manually with `force=true` to validate. |

## Retargeting at a different node

There are three ways to point the workflow at a different node, in increasing order of permanence:

1. **Single run, from the Actions UI:** Run workflow → fill in `target_node` (and optionally `target_user` / `remote_work_dir` / `target_rocm_path`). Anything left blank falls back to the matching repo variable. `target_node` has no hard-coded default (workflow fails fast), `target_user` falls through to the orchestrator runner's local user, `remote_work_dir` falls through to `/tmp/rvs-nightly-<run_id>`, and `target_rocm_path` falls through to `/opt/rocm`. This is the right path for "test this tarball on host X today".
2. **Single run, from `gh` CLI:**
   ```bash
   gh workflow run rvs-nightly-tests.yml \
     -f target_node="<host-or-ip>" \
     -f target_rocm_path="/opt/rocm-7.13"   # only needed on multi-ROCm hosts
   ```
3. **Permanent change:** update repo variable `RVS_TARGET_NODE` (and optionally `RVS_TARGET_USER` / `RVS_REMOTE_WORK_DIR` / `RVS_TARGET_ROCM_PATH`). All subsequent scheduled and manual runs will pick this up unless an input overrides it.

The same `RVS_TARGET_SSH_KEY` secret is reused across nodes — make sure the
public counterpart of that key is added to `$TARGET_USER`'s `~/.ssh/authorized_keys`
on **every** node you intend to point the workflow at.

## References

- [RVS source](../../README.md)
- [`build-relocatable-packages.yml`](./build-relocatable-packages.yml) and [`README_BUILD_PACKAGES.md`](./README_BUILD_PACKAGES.md) — the upstream packaging pipeline that produces these tarballs
- [GitHub Actions: scheduled events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)
- [GitHub Actions: encrypted secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions) — for `RVS_TARGET_SSH_KEY`
