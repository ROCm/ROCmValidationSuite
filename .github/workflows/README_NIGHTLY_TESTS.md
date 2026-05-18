# RVS Nightly Tests Workflow

This document describes [`.github/workflows/rvs-nightly-tests.yml`](./rvs-nightly-tests.yml),
which picks up the **latest RVS tarball** from the index URL configured in
`vars.RVS_TARBALL_INDEX_URL` (e.g. `https://repo.amd.com/rocm/rvs/tarball/`)
once per day, installs it on a self-hosted GPU runner, and runs RVS levels
3 and 4.

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
test        [self-hosted GPU runner]
    │  verify ROCm prereqs                          → fail-fast if ROCm missing or wrong major
    │  curl <tarball URL>                           → ./pkg/<file>.tar.gz
    │  parse ROCM_MAJOR from amdrocm<N>-…           → INSTALL_DIR=/opt/rocm/extras-<N>
    │  sudo tar -xzf … -C $INSTALL_DIR              → installs to /opt/rocm/extras-<N>/
    │  ldd $INSTALL_DIR/bin/rvs                     → fail-fast if any "not found" libs
    │  $INSTALL_DIR/bin/rvs -r 3                    → reports/rvs_level_3.log
    │  $INSTALL_DIR/bin/rvs -r 4                    → reports/rvs_level_4.log
    │  build Markdown SUMMARY.md                    → GitHub job summary + artifact
    ▼
artifact: rvs-nightly-report-<run_id>
```

## Triggers

| Trigger | Cadence | What fires |
|---|---|---|
| `schedule` | `0 15 * * *` UTC daily (08:00 PST / 07:00 PDT) | Polls the tarball index. If the latest filename matches the previous run's, the run is **skipped** to avoid re-testing the same package. |
| `workflow_dispatch` | Manual | Always runs. Supports overriding the tarball URL and forcing a re-run. |

The cron deliberately runs after AMD's typical nightly publish window;
adjust the cron string in the workflow if your publish cadence is different.

## Manual dispatch inputs

| Input | Default | Description |
|---|---|---|
| `tarball_url` | _(empty)_ | If set, the workflow downloads this exact URL instead of scraping the index. Useful for re-running an older build. |
| `force` | `false` | When `true`, runs even if the latest tarball filename matches the cached marker. |

Example (replace `<INDEX_URL>` with the value of `vars.RVS_TARBALL_INDEX_URL`):

```bash
gh workflow run rvs-nightly-tests.yml \
  -f tarball_url="<INDEX_URL>/amdrocm7-rvs-1.4.21-288-Linux.tar.gz" \
  -f force=true
```

## Repository configuration

No secrets are required.

**Variables** (Settings → Secrets and variables → Actions → Variables):

| Name | Required? | Purpose |
|---|---|---|
| `RVS_TARBALL_INDEX_URL` | **Required** | Directory listing scraped for the latest tarball, e.g. `https://repo.amd.com/rocm/rvs/tarball/`. No fallback — the workflow fails fast if unset and no `tarball_url` input is supplied. |
| `RVS_TEST_RUNNER_LABEL` | optional (default `self-hosted`) | Runner label (single string) used by the `test` job. Set to a more specific label like `gfx942` or `mi300x` if you have multiple GPU runners. |

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

## How the tarball is installed

The install step parses the ROCm major version from the tarball filename
(e.g. `amdrocm7-rvs-1.4.21-…-Linux.tar.gz` → `7`) and extracts directly
into the matching `extras-<major>` directory, so the same workflow handles
ROCm 6, 7, etc. without code changes:

```bash
# Parsed from $TARBALL_NAME via [[ "$TARBALL_NAME" =~ ^amdrocm([0-9]+)- ]]
ROCM_MAJOR=7
INSTALL_DIR=/opt/rocm/extras-${ROCM_MAJOR}    # /opt/rocm/extras-7
RVS_BIN_PATH=${INSTALL_DIR}/bin/rvs

sudo mkdir -p "$INSTALL_DIR"
sudo tar -xzf ./pkg/<tarball>.tar.gz -C "$INSTALL_DIR"

export LD_LIBRARY_PATH="$INSTALL_DIR/lib:/install/lib:/install/lib/rocm_sysdeps/:/install/lib/llvm/lib:${LD_LIBRARY_PATH:-}"

"$RVS_BIN_PATH" --version
```

The step fails fast if the filename doesn't match `^amdrocm<digits>-`, or
if `$RVS_BIN_PATH` isn't executable after extraction. The resolved
`RVS_BIN_PATH` is written to `$GITHUB_ENV` so the level-3, level-4, and
report steps pick it up automatically.

**Prerequisites on the runner** (all enforced by the **Pre-flight ROCm checks** below — the workflow fails fast if any are missing):

- ROCm base stack already installed at `/opt/rocm/` with a major version that matches the tarball name (e.g. `amdrocm7-rvs-…` requires ROCm 7.x). The TGZ only ships RVS, not a full ROCm install; the RVS binary depends on ROCm libraries resolved via `RPATH` to the system ROCm.
- Working kernel driver (`amdgpu`) and at least one GPU enumerated by `rocm-smi` and `rocminfo`.
- Passwordless `sudo` (or `sudo` configured to not prompt) so the `tar -xzf -C "$INSTALL_DIR"` step works without interactive input.
- Network egress to the host serving `vars.RVS_TARBALL_INDEX_URL` (typically port 443).

## Pre-flight ROCm checks

Before downloading or installing the tarball, the workflow runs **`Verify ROCm prerequisites`** on the GPU runner. The step accumulates failures and reports all of them in a single log so the cause is obvious:

| Sub-check | Action on failure | Catches |
|---|---|---|
| `/opt/rocm/` directory exists | hard fail (`exit 1`) | ROCm not installed at all |
| Read ROCm version from `/opt/rocm/.info/version` (fallback `/opt/rocm/share/doc/rocm-core/version`) | warn if missing | Partial install, dev pkgs only |
| Major version match (regex on tarball name vs runner version) | hard fail on mismatch | RVS-7 tarball on a ROCm-6 host |
| `rocm-smi --showid` runs and exits 0 | hard fail | `amdgpu` driver not loaded, or `rocm-smi` missing |
| `rocminfo` enumerates ≥ 1 GPU | hard fail | "ROCm installed but no GPU exposed" (group/permissions, BIOS IOMMU, etc.) |
| `libhsa-runtime64.so.1` and `libamdhip64.so` present in `ldconfig -p` | warn only | Missing/broken `ldconfig` cache (RPATH usually compensates) |

A typical successful log:

```
=== /opt/rocm/ ===
/opt/rocm -> /opt/rocm-7.11.0
=== ROCm version ===
/opt/rocm/.info/version : 7.11.0
=== Major version match (tarball vs runner) ===
tarball expects ROCm major : 7
runner has ROCm major      : 7
=== rocm-smi ===
GPU[0]  : ID  : 0x74a1
=== rocminfo (GPU enumeration) ===
GPU agents enumerated: 8
=== Key ROCm runtime libraries ===
  OK    : libhsa-runtime64.so.1
  OK    : libamdhip64.so
::notice::ROCm prerequisites OK (version 7.11.0)
```

After install, **`Verify RVS binary library resolution`** runs `ldd "$RVS_BIN"` (where `$RVS_BIN` = `/opt/rocm/extras-${ROCM_MAJOR}/bin/rvs` from the install step) and fails the job if any library shows up as `not found` — preventing the workflow from spending hours on `rvs -r 3`/`-r 4` only to discover a `dlopen` error in the level log.

### Manual one-liner (no workflow needed)

To verify a runner is viable before enabling the schedule, drop this into an SSH session against any candidate runner:

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
echo "ROCm OK"
```

## The tests

Run verbatim (with `$RVS_BIN` = `/opt/rocm/extras-${ROCM_MAJOR}/bin/rvs`,
populated by the install step — for an `amdrocm7-…` tarball this resolves
to `/opt/rocm/extras-7/bin/rvs`):

```bash
"$RVS_BIN" -r 3
"$RVS_BIN" -r 4
```

Both commands are executed sequentially. Each step captures full
stdout/stderr to `reports/rvs_level_<N>.log` and its exit code is recorded
in a step output. Neither failure short-circuits the other — both levels
always run, and the job is marked failed at the end if either exited
non-zero.

## Test report

After both RVS commands finish, the `Build test report` step generates
`reports/SUMMARY.md` (also written to the GitHub job summary), e.g.:

```markdown
# RVS Nightly Test Report

| Field | Value |
|---|---|
| Run | `1234567890` |
| Trigger | `schedule` |
| Runner | `gpu-runner-01` |
| Tarball | `amdrocm7-rvs-1.4.21-288-Linux.tar.gz` |
| Source URL | `$RVS_TARBALL_INDEX_URL/amdrocm7-rvs-1.4.21-288-Linux.tar.gz` |
| RVS version | `RVS 1.4.21.0-...` |
| Overall result | **PASS** |

## Results

| Test    | Command                              | Result | Exit | Started (UTC)         | Ended (UTC)           |
|---------|--------------------------------------|:------:|-----:|-----------------------|-----------------------|
| Level 3 | `/opt/rocm/extras-7/bin/rvs -r 3`    | PASS   |    0 | 2026-05-19T15:01:00Z  | 2026-05-19T15:25:11Z  |
| Level 4 | `/opt/rocm/extras-7/bin/rvs -r 4`    | PASS   |    0 | 2026-05-19T15:25:12Z  | 2026-05-19T16:02:47Z  |
```

The full artifact contents:

```
rvs-nightly-report-<run_id>/
├── SUMMARY.md
├── rvs_level_3.log
└── rvs_level_4.log
```

Artifact retention is 30 days.

## Self-hosted runner

The `test` job runs on `${{ vars.RVS_TEST_RUNNER_LABEL || 'self-hosted' }}`.
You must have at least one self-hosted runner registered to the repo with
that label, online, and with a usable GPU. Register runners at
`https://github.com/ROCm/ROCmValidationSuite/settings/actions/runners`.

If the chosen runner is busy with another job, this workflow's
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
2. `test` job picks up on your self-hosted runner.
3. **Verify ROCm prerequisites** prints `::notice::ROCm prerequisites OK`.
4. **Install RVS (detect ROCm major from filename)** prints the detected `ROCM_MAJOR` and `Installed RVS at: /opt/rocm/extras-<N>/bin/rvs`.
5. **Verify RVS binary library resolution** prints `::notice::RVS binary's library dependencies resolved OK`.
6. Two RVS level steps complete; the run summary shows the results table.

## Debugging a failed run

| Symptom | Likely cause |
|---|---|
| `detect` job exits with `vars.RVS_TARBALL_INDEX_URL is not set` | The required variable is unset. Set it in the repo Variables (e.g. to `https://repo.amd.com/rocm/rvs/tarball/`), or pass `tarball_url` via `workflow_dispatch`. |
| `detect` job exits with "Could not resolve a tarball URL" | The index page returned no matches. Verify the URL in `vars.RVS_TARBALL_INDEX_URL` returns at least one `amdrocm*-rvs-*-Linux.tar.gz` link. |
| `test` job stuck "Queued" | No self-hosted runner online with the label in `vars.RVS_TEST_RUNNER_LABEL`. |
| **Verify ROCm prerequisites** fails: `/opt/rocm/ does not exist` | Runner has no ROCm install. Install ROCm or pick a runner that has it. |
| **Verify ROCm prerequisites** fails: `ROCm major version mismatch` | The tarball is for ROCm `<X>` but the runner has ROCm `<Y>`. Either pin a matching tarball with `tarball_url`, or upgrade the runner's ROCm. |
| **Verify ROCm prerequisites** fails: `rocm-smi … exited non-zero` | `amdgpu` kernel driver is not loaded. `lsmod \| grep amdgpu`, then `sudo modprobe amdgpu` and check `dmesg`. |
| **Verify ROCm prerequisites** fails: `rocminfo enumerated 0 GPU agents` | Driver loaded but no GPU exposed to the runner user. Add the user to `video` and `render` groups, log out and back in. |
| **Install RVS** fails: `Cannot parse ROCm major version from tarball name` | The tarball doesn't match `^amdrocm<digits>-`. Either pin a correctly-named tarball with `tarball_url`, or fix the upstream filename. |
| **Install RVS** fails: `rvs binary not found or not executable at /opt/rocm/extras-<N>/bin/rvs after install` | The tarball isn't rooted at `./bin/`, `./lib/`, etc. The extraction landed `bin/rvs` somewhere else inside `/opt/rocm/extras-<N>/`. Inspect the step log (`ls -la` output) to see the actual layout; the workflow may need `--strip-components=<N>` added to the `tar` invocation. |
| **Verify RVS binary library resolution** fails with `not found` | A ROCm runtime library is missing or not in `RPATH`. The step prints the `ldd` output; install the matching ROCm component (typically `rocm-llvm`, `rocm-core`, `hip-runtime-amd`). |
| `Install RVS` fails with "Permission denied" | Runner user can't `sudo` without password. Fix the runner's sudoers entry. |
| `Run RVS level N` exits non-zero immediately (after both verify steps passed) | RVS plugin's own dependency missing (e.g. `libpci3` on Debian). Check the level log for the specific error. |
| Cron skipped a day | GitHub may delay or drop schedules under high load. Run once manually with `force=true` to validate. |

## References

- [RVS source](../../README.md)
- [`build-relocatable-packages.yml`](./build-relocatable-packages.yml) and [`README_BUILD_PACKAGES.md`](./README_BUILD_PACKAGES.md) — the upstream packaging pipeline that produces these tarballs
- [GitHub Actions: scheduled events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)
