# RVS Nightly Tests Workflow

This document describes [`.github/workflows/rvs-nightly-tests.yml`](./rvs-nightly-tests.yml),
which picks up the **latest RVS tarball** published at
[`https://repo.amd.com/rocm/rvs/tarball/`](https://repo.amd.com/rocm/rvs/tarball/)
once per day, installs it on a self-hosted GPU runner, and runs RVS levels
3 and 4.

## What it does

```
schedule (or manual)
    │
    ▼
detect      [ubuntu-latest]
    │  curl https://repo.amd.com/rocm/rvs/tarball/
    │  → grep amdrocm*-rvs-*-Linux.tar.gz, sort -V, pick newest
    │  → compare with cached marker (skip if unchanged)
    ▼
test        [self-hosted GPU runner]
    │  curl <tarball URL>           → ./pkg/<file>.tar.gz
    │  sudo tar -xzf … -C /          → installs to /opt/rocm/extras-7/
    │  /opt/rocm/extras-7/bin/rvs -r 3   → reports/rvs_level_3.log
    │  /opt/rocm/extras-7/bin/rvs -r 4   → reports/rvs_level_4.log
    │  build Markdown SUMMARY.md     → GitHub job summary + artifact
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

Example:

```bash
gh workflow run rvs-nightly-tests.yml \
  -f tarball_url="https://repo.amd.com/rocm/rvs/tarball/amdrocm7-rvs-1.4.21-288-Linux.tar.gz" \
  -f force=true
```

## Repository configuration

No secrets are required — the source URL is public.

**Variables** (Settings → Secrets and variables → Actions → Variables):

| Name | Default | Purpose |
|---|---|---|
| `RVS_TARBALL_INDEX_URL` | `https://repo.amd.com/rocm/rvs/tarball/` | Directory listing scraped for the latest tarball. Override to mirror or staging URLs without editing the workflow. |
| `RVS_TEST_RUNNER_LABEL` | `self-hosted` | Runner label (single string) used by the `test` job. Set to a more specific label like `gfx942` or `mi300x` if you have multiple GPU runners. |

## How the latest tarball is picked

The `detect` job does:

```bash
curl -sL https://repo.amd.com/rocm/rvs/tarball/ \
  | grep -oE 'amdrocm[0-9]*-rvs-[0-9A-Za-z._\-]+-Linux\.tar\.gz' \
  | sort -uV \
  | tail -n 1
```

The regex matches any `amdrocm<N>-rvs-…-Linux.tar.gz` filename in the
directory-listing HTML. `sort -V` is GNU "version sort" so version
suffixes like `1.4.21-9` and `1.4.21-100` compare correctly. The
**lexicographically largest by version** is selected.

## How the tarball is installed

The TGZ ships content rooted at `/opt/rocm/extras-7/`, so the workflow
extracts it at `/`:

```bash
sudo mkdir -p /opt/rocm
sudo tar -xzf ./pkg/<tarball>.tar.gz -C /
/opt/rocm/extras-7/bin/rvs --version
```

**Prerequisites on the runner:**

- ROCm 7.x base stack already installed at `/opt/rocm/` (the TGZ only ships
  RVS, not a full ROCm install; the RVS binary depends on ROCm libraries
  resolved via `RPATH` and `LD_LIBRARY_PATH` to the system ROCm).
- Passwordless `sudo` (or `sudo` configured to not prompt) so the
  `tar -xzf -C /` step works without interactive input.
- Network egress to `repo.amd.com` (port 443).

## The tests

Run verbatim:

```bash
/opt/rocm/extras-7/bin/rvs -r 3
/opt/rocm/extras-7/bin/rvs -r 4
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
| Source URL | https://repo.amd.com/rocm/rvs/tarball/amdrocm7-rvs-1.4.21-288-Linux.tar.gz |
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

1. `detect` resolves a tarball URL (`Latest tarball : amdrocm7-rvs-…`).
2. `test` job picks up on your self-hosted runner.
3. RVS extracts to `/opt/rocm/extras-7/`.
4. Two RVS level steps complete; the run summary shows the results table.

## Debugging a failed run

| Symptom | Likely cause |
|---|---|
| `detect` job exits with "Could not resolve a tarball URL" | The index page returned no matches. Verify `https://repo.amd.com/rocm/rvs/tarball/` returns at least one `amdrocm*-rvs-*-Linux.tar.gz` link, or set `RVS_TARBALL_INDEX_URL` to the correct mirror. |
| `test` job stuck "Queued" | No self-hosted runner online with the label in `vars.RVS_TEST_RUNNER_LABEL`. |
| `Install RVS` fails with "Permission denied" | Runner user can't `sudo` without password. Fix the runner's sudoers entry. |
| `Run RVS level N` exits non-zero immediately | ROCm base stack missing or wrong major version. `rvs --version` and `ldd /opt/rocm/extras-7/bin/rvs` give clues. |
| Cron skipped a day | GitHub may delay or drop schedules under high load. Run once manually with `force=true` to validate. |

## References

- [RVS source](../../README.md)
- [`build-relocatable-packages.yml`](./build-relocatable-packages.yml) and [`README_BUILD_PACKAGES.md`](./README_BUILD_PACKAGES.md) — the upstream packaging pipeline that produces these tarballs
- [GitHub Actions: scheduled events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)
