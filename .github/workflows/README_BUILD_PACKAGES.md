# Building Relocatable Packages with GitHub Actions

This document describes the GitHub Actions workflow for building relocatable ROCm Validation Suite (RVS) packages using the ROCm SDK from [TheRock](https://github.com/ROCm/TheRock).

## Overview

The workflow (`.github/workflows/build-relocatable-packages.yml`) automatically builds:
- **DEB packages** for Ubuntu/Debian systems
- **RPM packages** for CentOS/RHEL/Rocky Linux systems
- **TGZ archives** for any Linux distribution (relocatable)

All packages are built with relocatable RPATH settings, meaning they can be installed to different locations.

## Workflow Triggers

The workflow runs automatically on:
- Push to `master`, `main`, or `release/**` branches
- Pull requests to `master` or `main` branches
- **Scheduled**: Daily at **5:00 AM PST** (13:00 UTC); **latest ROCm nightly** from [ROCm SDK nightly tarballs](https://rocm.nightlies.amd.com/tarball-multi-arch/) (or repo variable overrides). The cron fans out to the **default branch** plus every branch matched by the repository variable **`ACTIVE_BRANCHES`** (comma-separated literals/globs, e.g. `npi/**,release/**` — do not list the default branch there).
- **Manual** (`workflow_dispatch`): if **`ROCM_VERSION`** is pinned (**workflow input** and/or **`vars.ROCM_VERSION`**), the tarball is chosen by **version format** (nightly `x.y.za…` vs release `X.Y.Z`). If no version is pinned, the job uses **latest nightly** (same as schedule).

### Scheduled multi-branch nightly (`ACTIVE_BRANCHES`)

| Branch source | Built on schedule? | S3 upload on schedule? | S3 path (under bucket) |
|---------------|-------------------|------------------------|-------------------------|
| **Default branch** (`master` / `main`) | Always | Yes | Unchanged: `nightly/rvs/deb/`, `nightly/rvs/rpm/`, `nightly/rvs/tar/` (+ APT/YUM metadata) |
| **`ACTIVE_BRANCHES`** matches (non-default) | Yes | Yes (except `release*`) | `{branch_prefix}/{branch}/nightly/deb/`, `…/rpm/`, `…/tar/` |
| **`release*`** matches from `ACTIVE_BRANCHES` | Yes | **No** | Packages are built and verified only |

- **`branch_prefix`**: first path segment of the branch name (`npi` for `npi/foo`; for a branch with no `/`, the full branch name).
- **`push` / `pull_request` / `workflow_dispatch`**: single-branch matrix; S3 routing is unchanged from the table below.
- Set **`ACTIVE_BRANCHES`** under **Settings → Secrets and variables → Actions → Variables** (example: `npi/**,release/**`).

- **Pull requests** and **pushes** to `master`, `main`, or `release/**` (including merges): **latest ROCm release** (**X.Y.Z**) from [repo.amd.com ROCm tarballs](https://repo.amd.com/rocm/tarball/).

### ROCm SDK: nightly vs release in CI

The workflow runs [`.github/scripts/configure-rocm-sdk-channel.sh`](../scripts/configure-rocm-sdk-channel.sh) to set `ROCM_SDK_CHANNEL`, SDK URLs, and `ROCM_VERSION` (from `vars.ROCM_VERSION` on push/PR, or `workflow_dispatch` input) before calling `build_packages_local.sh`. Release-branch detection uses POSIX `case` (not bash `[[`), so it works in the Ubuntu `ubuntu:22.04` container where steps may run under `sh`.

**ROCm pin vs S3 layout:** A nightly SDK pin (e.g. `7.14.0a20260528`) only affects **which tarball is downloaded**. **`workflow_dispatch` or `push` on a `release/**` branch** still uploads packages to **`release/rvs/`** (see S3 table below).

| Trigger | SDK source |
|---------|------------|
| **`schedule`** | **Nightly** — latest nightly tarball from the nightly listing |
| **`pull_request`** (to `master` / `main`) | **Release** — latest **X.Y.Z** from the release listing |
| **`push`** to `master`, `main`, or `release/**` | **Release** — same (covers merge-after-PR) |
| **`push`** to other branches | **Nightly** |
| **`workflow_dispatch`** | **Format-based** when `rocm_version` input or `vars.ROCM_VERSION` is set; otherwise **latest nightly** |

Local builds (no CI): default is **latest nightly** if `ROCM_VERSION` is unset; if set, tarball location follows the same **format** rules as in `build_packages_local.sh`.

### Manual Trigger Parameters

When manually triggering the workflow, you can specify:

1. **ROCm Version**
   - Empty — **latest nightly** (unless `vars.ROCM_VERSION` is set in the repo, which pins a version and triggers format-based selection).
   - **`X.Y.Z`** — release tarball at [repo.amd.com](https://repo.amd.com/rocm/tarball/).
   - **`x.y.za…`** — nightly tarball at [nightlies](https://rocm.nightlies.amd.com/tarball-multi-arch/).
   
2. **GPU Family Target**:
   - `gfx94X-dcgpu` - MI300A/MI300X
   - `gfx950-dcgpu` - MI350X/MI355X
   - `gfx110X-all` - AMD RX 7900 XTX, 7800 XT, 7700S, Radeon 780M (default)
   - `gfx1151` - AMD Strix Halo iGPU
   - `gfx120X-all` - AMD RX 9060/XT, 9070/XT

3. **Build TransferBench CLI** (boolean, default **true** on manual runs):
   - When enabled, the `TransferBench` binary is built and bundled in DEB/RPM/TGZ (requires `libnuma1` / `numactl-libs` at runtime).
   - On **push**, **PR**, and **schedule**, CI defaults to **ON** unless repository variable `BUILD_TRANSFERBENCH_CLI` is set to `OFF`.

## How It Works

The workflow leverages the `build_packages_local.sh` script for all build operations, ensuring consistency between local development and CI/CD environments.

### Workflow Architecture

```
GitHub Actions Workflow
├── Checkout Repository (with submodules)
├── Set Environment Variables (ROCM_VERSION, GPU_FAMILY, BUILD_TYPE)
└── Execute build_packages_local.sh
    ├── 1. Detect OS and Install Dependencies
    │   ├── Ubuntu: apt-get install build-essential, cmake, python3, etc.
    │   └── AlmaLinux: yum install gcc, gcc-c++, cmake3, python3, etc.
    │       └── Enable PowerTools/CRB for doxygen and yaml-cpp
    ├── 2. Auto-fetch Latest ROCm or Use Specified Version
    │   └── URL: https://therock-nightly-tarball.s3.../${GPU_FAMILY}-${ROCM_VERSION}.tar.gz
    ├── 3. Extract and Setup ROCm Environment
    │   ├── export ROCM_PATH="$HOME/rocm-sdk/install"
    │   ├── export PATH="$ROCM_PATH/bin:$PATH"
    │   ├── export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
    │   ├── export CMAKE_PREFIX_PATH="$ROCM_PATH:$CMAKE_PREFIX_PATH"
    │   ├── export HIP_DEVICE_LIB_PATH (auto-detected amdgcn/bitcode path)
    │   ├── export ROCM_LIBPATCH_VERSION (major.minor in xxyy format, e.g., 0711)
    │   ├── export CPACK_DEBIAN_PACKAGE_RELEASE (see env table: r<libpatch>.date, PRs add .branch.commit)
    │   ├── export CPACK_RPM_PACKAGE_RELEASE (same as DEB)
    │   ├── export CMAKE_CXX_COMPILER=hipcc (AlmaLinux and Ubuntu/Debian)
    │   └── export CMAKE_COMMAND=cmake3 (AlmaLinux) or cmake (Ubuntu)
    ├── 4. Configure CMake (install RPATH defaults live in CMakeLists.txt)
    │   └── $CMAKE_COMMAND -B ./build \
    │         -DCMAKE_BUILD_TYPE=Release \
    │         -DROCM_PATH=$ROCM_PATH \
    │         -DHIP_PLATFORM=amd \
    │         -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER (if set) \
    │         -DROCM_MAJOR_VERSION=$ROCM_MAJOR \
    │         -DCMAKE_INSTALL_PREFIX=/opt/rocm/extras-$ROCM_MAJOR \
    │         -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm/extras-$ROCM_MAJOR \
    │         -DCMAKE_VERBOSE_MAKEFILE=1 \
    │         -DFETCH_ROCMPATH_FROM_ROCMCORE=ON
    ├── 5. Build RVS
    │   └── make -C ./build -j$(nproc)
    └── 6. Create Packages
        ├── Ubuntu: DEB + TGZ (via CPack)
        └── AlmaLinux: RPM + TGZ (via CPack)
```

### Key Technical Details

**Relocatable RPATH**: Defaults are set in **`CMakeLists.txt`** (`CMAKE_INSTALL_RPATH`, **`CMAKE_BUILD_RPATH`** (same list for the build tree), `CMAKE_SKIP_RPATH`, `CMAKE_INSTALL_RPATH_USE_LINK_PATH`). **`CMAKE_*_LINKER_FLAGS_INIT`** only adds **`--enable-new-dtags`** (RUNPATH behavior), not a second copy of **`$ORIGIN`**. On **GitHub Actions** (`GITHUB_ACTIONS=true`), **`CMAKE_SKIP_BUILD_RPATH`** applies when using **CMake 3.9+**, and **`CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH`** (strip implicit SDK paths on install) when using **CMake 3.16+**. The **`$ORIGIN`** relative entries resolve to the install prefix (`.../extras-<N>/bin` → `.../extras-<N>/lib`), so **`/opt/rocm/extras-<N>/lib` is not duplicated** in the list. Absolute paths add **`/opt/rocm/lib`**, **`/opt/rocm/lib/llvm/lib`** (older ROCm layouts), **`/opt/rocm/core-<ROCM_MAJOR>/lib`**, and **`/opt/rocm/core-<ROCM_MAJOR>/lib/llvm/lib`** — equivalent to:

```bash
CMAKE_INSTALL_RPATH="$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs:/opt/rocm/lib:/opt/rocm/lib/llvm/lib:/opt/rocm/core-<ROCM_MAJOR>/lib:/opt/rocm/core-<ROCM_MAJOR>/lib/llvm/lib"
```

**Automatic Version Management**: CMake reads the project version from `CMakeLists.txt` and CPack uses it for package naming automatically. The **patch version** is auto-computed at CMake configure time: `git describe --tags --match "v<major>.<minor>.*"` counts commits since the last matching `v` tag. For example, if the tag is `v1.3.0` and there have been 15 commits since, the package version becomes `1.3.15`. If no matching tag exists or git is unavailable, the patch defaults to `0` from `CMakeLists.txt`. This works for both CI builds and direct local `cmake` invocations.

**HIP Device Libraries**: Automatically located and exported as `HIP_DEVICE_LIB_PATH` for clang device library discovery.

**Compiler Selection**: `CMAKE_CXX_COMPILER` is set to ROCm's `hipcc` on both AlmaLinux (manylinux_2_28) and Ubuntu/Debian. Because hipcc is Clang-based and does not bundle libstdc++, the system GCC tree is plumbed in via `--gcc-toolchain=$GCC_TOOLCHAIN`: on AlmaLinux this points at the discovered `gcc-toolset-<N>`; on Ubuntu/Debian it is `/usr` (set when `/usr/include/c++/*/barrier` exists).

**CMake Command**: Uses `cmake3` on AlmaLinux (manylinux_2_28) and `cmake` on Ubuntu.

**Verbose Build Output**: `CMAKE_VERBOSE_MAKEFILE=1` enables detailed compilation output for debugging and transparency.

**Dynamic ROCm Path Discovery**: `FETCH_ROCMPATH_FROM_ROCMCORE=ON` allows RVS to automatically detect ROCm installation location at runtime from ROCm core libraries.

**Single Source of Truth**: `build_packages_local.sh` drives configure/build/package for CI and local use; **RPATH defaults** live in **`CMakeLists.txt`** so a plain **`cmake`** invocation gets the same install **`RPATH`** without copying flags into the shell script. When **`BUILD_TRANSFERBENCH_CLI=ON`**, [`CMakeTransferBenchCLI.cmake`](../CMakeTransferBenchCLI.cmake) applies TransferBench-specific relocatable RPATH via [`CMakeTransferBenchRPATH.cmake.in`](../CMakeTransferBenchRPATH.cmake.in).

### Workflow Steps

The GitHub Actions workflow performs minimal platform-specific operations:

1. **Checkout Repository** with recursive submodule initialization
2. **Set Environment Variables** from workflow inputs or defaults
3. **Execute Build Script** - `./build_packages_local.sh` handles everything
4. **Verify Packages** - Platform-specific verification (dpkg-deb or rpm -q)
5. **Upload to S3** (when repo is `ROCm/ROCmValidationSuite`) – Each job uploads its packages to S3 using OIDC. The bash routing logic determines the S3 path: `release/*` branch builds (push or manual) go to `release/`, scheduled/push/manual builds go to `nightly/`, and PR builds go to a ref-specific path. Requires `AWS_S3_BUCKET` (variable) and `AWS_ROLE_ARN` (secret). Skipped gracefully if `AWS_S3_BUCKET` is not set.
6. **Generate Repo Metadata** (schedule, push, and manual builds only) – Creates APT repo metadata (`Packages`, `Packages.gz`, `Release`) for DEB and YUM/DNF repodata (`repodata/`) for RPM, then uploads to S3 so the paths can be used as native package repositories. Skipped for PR builds since their packages go to one-off ref-specific paths.

### S3 Upload (OIDC – No Stored Credentials)

S3 upload runs **only when** the repository is **`ROCm/ROCmValidationSuite`** (the `if` guard prevents forks from attempting credential setup). The upload step itself is always reached, but exits gracefully if `AWS_S3_BUCKET` is not set. Uses **AWS OIDC**; no long-term access key or secret. The bash routing inside the upload step determines the S3 destination based on the event type and branch.

**Where `vars` and `secrets` are defined**

They are **not** in the workflow file. They are set in the repo:

- **Secrets** (e.g. `secrets.AWS_ROLE_ARN`): **Settings** → **Secrets and variables** → **Actions** → **Secrets** tab → New repository secret.
- **Variables** (e.g. `vars.AWS_S3_BUCKET`): **Settings** → **Secrets and variables** → **Actions** → **Variables** tab → New repository variable.

### Runner Configuration

The workflow uses GitHub repository variables to control which runners execute each job, allowing you to use self-hosted runners or GitHub-provided runners:

| Variable | Default | Used by |
|----------|---------|---------|
| `ACTIVE_BRANCHES` | _(unset)_ | **Scheduled** nightly only: comma-separated branch literals/globs (`npi/**`, `release/**`). Default branch is always built; do not list it here. `release*` matches build but do not upload on schedule. |
| `RUNNER_LABEL` | `ubuntu-22.04` | Ubuntu build job |
| `RUNNER_LABEL_CONTAINER` | `ubuntu-latest` | CentOS/manylinux build job (container) |
| `RUNNER_LABEL_UTILITY` | `ubuntu-latest` | Release summary job |

To use a self-hosted runner, set the variable to your runner's label (e.g., `self-hosted` or a custom label) in **Settings** → **Secrets and variables** → **Actions** → **Variables**.

**Required setup for S3 upload:**

1. **Repository secret** (Secrets tab, see above):
   - Name: `AWS_ROLE_ARN`
   - Value: the IAM role ARN to assume for S3 upload (e.g. `arn:aws:iam::123456789012:role/my-s3-upload-role`). Keeps the role ID hidden from the workflow.

2. **Repository variable** (Variables tab, see above):
   - Name: `AWS_S3_BUCKET`
   - Value: your S3 bucket name (e.g. `my-rocm-packages`).

3. **AWS IAM**: The role in `AWS_ROLE_ARN` must have a trust policy allowing GitHub OIDC to assume it (identity provider `token.actions.githubusercontent.com`, audience `sts.amazonaws.com`) and permissions to `s3:PutObject` (and related) on the bucket.

**S3 path layout** (resolved by [`.github/scripts/rvs-s3-upload-route.sh`](../scripts/rvs-s3-upload-route.sh), POSIX-safe for Ubuntu `sh`):

| Trigger | Path | Contents |
|--------|------|----------|
| **`release/*` branch** (`push` or `workflow_dispatch`) | `release/rvs/deb/`, `release/rvs/rpm/`, `release/rvs/tar/` | DEB → `.../deb` (Ubuntu job); RPM and TGZ → `.../rpm` and `.../tar` (manylinux job). Only PR merges into release branches or manual dispatch on release branches write here. |
| **Scheduled** (default branch only) | `nightly/rvs/deb/`, `nightly/rvs/rpm/`, `nightly/rvs/tar/` | Same as before; APT/YUM metadata generated here. |
| **Scheduled** (`ACTIVE_BRANCHES`, non-default, not `release*`) | `{branch_prefix}/{branch}/nightly/deb/`, `…/rpm/`, `…/tar/` | No shared `rvs/` segment; no repo metadata on these paths. |
| **Scheduled** (`release*` from `ACTIVE_BRANCHES`) | _(none)_ | Build only; upload skipped. |
| **Push to `master`/`main`**, or **`workflow_dispatch` on non-release branch** | `nightly/rvs/deb/`, `nightly/rvs/rpm/`, `nightly/rvs/tar/` | Same split by type. |
| **Pull request** (same-repo) | `rvs/<ref_name>/<run_number>/ubuntu-22.04/` or `.../manylinux_2_28/` | DEB only (Ubuntu job); RPM+TGZ (manylinux job). One-off path, no repo metadata generated. |

If `AWS_S3_BUCKET` is not set, the upload step is skipped with a warning (the workflow still succeeds).

When packages are uploaded to S3, the **build report** artifact includes an **S3 Upload Locations** section with clickable links to each S3 prefix (AWS Console). This makes it easy to open the bucket and browse the uploaded DEB, RPM, and TGZ files from the report.

### Repository Metadata (repodata)

For **scheduled**, **push**, and **manual** (`workflow_dispatch`) builds, the workflow generates package repository metadata so that the S3 paths can be used directly as `apt` (DEB) and `yum`/`dnf` (RPM) repositories. This runs after the package upload step in each job. PR builds are excluded since their packages go to one-off ref-specific paths.

**RPM repodata** (CentOS/RHEL job):
- Tool: `createrepo_c` (falls back to `createrepo`)
- Downloads existing RPMs from S3, merges in the newly built RPM, regenerates the `repodata/` directory, and syncs everything back
- Result: `repodata/repomd.xml`, `repodata/primary.xml.gz`, `repodata/filelists.xml.gz`, `repodata/other.xml.gz`

**DEB repo metadata** (Ubuntu job):
- Tools: `dpkg-scanpackages`, `apt-ftparchive`
- Downloads existing DEBs from S3, merges in the newly built DEB, regenerates `Packages`, `Packages.gz`, and `Release`, and syncs everything back
- Result: `Packages`, `Packages.gz`, `Release`

**S3 directory layout after metadata generation:**

```
s3://<bucket>/nightly/rvs/
├── deb/
│   ├── amdrocm7-rvs_1.3.15-r0711.20260423_amd64.deb
│   ├── Packages
│   ├── Packages.gz
│   └── Release
├── rpm/
│   ├── amdrocm7-rvs-1.3.15-r0711.20260423.x86_64.rpm
│   └── repodata/
│       ├── repomd.xml
│       ├── primary.xml.gz
│       ├── filelists.xml.gz
│       └── other.xml.gz
└── tar/
    └── amdrocm7-rvs-1.3.15-r0711.20260423-Linux.tar.gz
```

**Using the S3 repo with apt (Ubuntu/Debian):**

```bash
# Add the nightly repo (replace <bucket> with the actual S3 bucket name).
# Use "/" as the suite field for this flat repo layout.
echo "deb [trusted=yes] https://<bucket>.s3.amazonaws.com/nightly/rvs/deb /" \
  | sudo tee /etc/apt/sources.list.d/rvs-nightly.list

# Or the release repo
echo "deb [trusted=yes] https://<bucket>.s3.amazonaws.com/release/rvs/deb /" \
  | sudo tee /etc/apt/sources.list.d/rvs-release.list

sudo apt update
sudo apt install amdrocm7-rvs  # Replace 7 with your ROCm major version
```

After `apt update`, **`apt list`** should show **`rvs-nightly`** or **`rvs-release`** (matching the bucket: `nightly/rvs/deb` vs `release/rvs/deb`) in the suite/codename column, because CI sets `Suite`, `Label`, and `Codename` in the flat `Release` file via `apt-ftparchive` options. If you still see **`unknown`**, run `apt update` again after a fresh metadata upload, or check that your `Release` on the server includes those fields. **`apt install amdrocm7-rvs`** still resolves versions from `Packages` either way.

**Using the S3 repo with yum/dnf (CentOS/RHEL/Rocky):**

```bash
# Add the nightly repo (replace <bucket> with the actual S3 bucket name)
cat <<'EOF' | sudo tee /etc/yum.repos.d/rvs-nightly.repo
[rvs-nightly]
name=RVS Nightly Packages
baseurl=https://<bucket>.s3.amazonaws.com/nightly/rvs/rpm/
enabled=1
gpgcheck=0
EOF

# Or the release repo
cat <<'EOF' | sudo tee /etc/yum.repos.d/rvs-release.repo
[rvs-release]
name=RVS Release Packages
baseurl=https://<bucket>.s3.amazonaws.com/release/rvs/rpm/
enabled=1
gpgcheck=0
EOF

sudo yum install amdrocm7-rvs  # Replace 7 with your ROCm major version
# or: sudo dnf install amdrocm7-rvs
```

> **Note:** `[trusted=yes]` (apt) and `gpgcheck=0` (yum) disable GPG verification. For production use, sign the packages and metadata with a GPG key and distribute the public key to users. Repository metadata is generated for **scheduled**, **push**, and **manual** builds; PR builds upload raw packages to ref-specific paths without metadata.

## Build Script: build_packages_local.sh

The workflow uses `build_packages_local.sh` as the core build engine. This script provides a complete, self-contained build system that works identically in both local development and CI/CD environments.

### Script Features

- **OS Detection**: Automatically identifies Ubuntu vs AlmaLinux
- **Dependency Management**: Installs all required build tools and libraries
  - Enables PowerTools/CRB repository on AlmaLinux for doxygen and yaml-cpp
- **ROCm SDK Setup**: Auto-fetches latest version or downloads specified version from TheRock tarballs
- **HIP Device Libraries**: Auto-locates amdgcn/bitcode directory and exports HIP_DEVICE_LIB_PATH
- **CMake Configuration**: Sets up relocatable RPATHs and all build parameters
  - Uses cmake3 on AlmaLinux, cmake on Ubuntu
  - Sets CMAKE_CXX_COMPILER to hipcc on AlmaLinux and Ubuntu/Debian (Ubuntu also gets `--gcc-toolchain=/usr` for C++20 libstdc++ headers)
- **Building**: Compiles RVS with parallel builds
- **Packaging**: Creates DEB, RPM, and TGZ packages using CPack
- **Color Output**: Clear, colored progress indicators
- **Error Handling**: Robust error checking at each step

### Running Locally

```bash
# Basic usage (automatically installs dependencies, fetches latest ROCm)
sudo ./build_packages_local.sh

# Custom ROCm version and GPU family (use sudo -E to preserve environment variables)
sudo -E ROCM_VERSION=7.11.0a20260121 GPU_FAMILY=gfx110X-all ./build_packages_local.sh

# Or export first, then run with sudo -E
export ROCM_VERSION=7.11.0a20260121
export GPU_FAMILY=gfx110X-all
sudo -E ./build_packages_local.sh

# Debug build
sudo BUILD_TYPE=Debug ./build_packages_local.sh
```

**Important**: The script requires root privileges to install system dependencies. Use `sudo` when running locally on a bare-metal host. In GitHub Actions every build job runs inside a container as root, so the workflow invokes `./build_packages_local.sh` directly without `sudo`:
- **Ubuntu job**: runs in the `ubuntu:22.04` container
- **CentOS/manylinux job**: runs in the `manylinux_2_28_x86_64` container

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROCM_VERSION` | _(unset)_ | If set, selects tarball by **format** (see below). If unset locally: **latest nightly**. If unset in CI: channel selects latest **nightly** vs **release X.Y.Z**. |
| `ROCM_SDK_RELEASE_URL` | `https://repo.amd.com/rocm/tarball/` | HTML listing for **release** tarballs (`therock-dist-linux-<GPU_FAMILY>-X.Y.Z.tar.gz`). Used when `ROCM_SDK_CHANNEL=release` or `auto` with this URL set. |
| `ROCM_SDK_RELEASE_BASE_URL` | `https://repo.amd.com/rocm/tarball` | Directory URL for downloading **X.Y.Z** tarballs; overridden when version string is nightly-shaped. |
| `ROCM_SDK_BASE_URL` | See script | Effective tarball base after channel + version-shape resolution. |
| `ROCM_SDK_INDEX_URL` | `https://rocm.nightlies.amd.com/tarball-multi-arch/` | **Nightly** listing for latest nightly SDK discovery. |
| `ROCM_SDK_NIGHTLY_BASE_URL` | `https://rocm.nightlies.amd.com/tarball-multi-arch` | Tarball base for **nightly** builds (`x.y.za…` versions). |
| `ROCM_SDK_NIGHTLY_INDEX_URL` | _(same as index default)_ | Optional override for nightly listing URL. |
| `ROCM_SDK_CHANNEL` | `auto` locally | **`nightly`** / **`release`** / **`auto`**. CI sets channel per trigger (see table above); manual with a pin uses **`auto`** so tarball follows version format. |
| `GPU_FAMILY` | `gfx110X-all` | Target GPU architecture |
| `BUILD_TYPE` | `Release` | CMake build type (Release/Debug) |
| `BUILD_TRANSFERBENCH_CLI` | `OFF` locally; `ON` in CI | Build and install the TransferBench CLI in packages (`-DBUILD_TRANSFERBENCH_CLI`). CI uses `vars.BUILD_TRANSFERBENCH_CLI` or defaults to `ON`; `workflow_dispatch` input **`build_transferbench_cli`** overrides manual runs. |
| `ROCM_LIBPATCH_VERSION` | Auto-extracted from `ROCM_VERSION` | Major.minor in xxyy format with zero padding (e.g., `7.11` → `0711`, `8.0` → `0800`) - used for RVS version tagging |
| `CPACK_DEBIAN_PACKAGE_RELEASE` | Auto-generated | **Default** (`schedule`, `push`, `workflow_dispatch`, local): `r<ROCM_LIBPATCH_VERSION>.<yyyymmdd>` (e.g. `r0711.20260423` where `0711` = ROCm 7.11 from `ROCM_VERSION`). **Pull requests**: `r<libpatch>.<yyyymmdd>.<source-branch>.<commit>`. **Release branches** (name starts with `rel`, non-PR): `GITHUB_RUN_NUMBER` (fallback: `1`). |
| `CPACK_RPM_PACKAGE_RELEASE` | same as `CPACK_DEBIAN_PACKAGE_RELEASE` | Identical to DEB. |
| `GITHUB_RUN_NUMBER` | `1` (local) | GitHub Actions run number - automatically set in CI, defaults to `1` for local builds |

## Build Matrix

The workflow builds packages for:

| Platform | Container/Runner | Package Types | Script Mode |
|----------|------------------|---------------|-------------|
| Ubuntu 22.04 | `ubuntu:22.04` container on `ubuntu-22.04` host | DEB, TGZ | Auto-detects Ubuntu |
| Manylinux 2.28 (AlmaLinux 8) | `manylinux_2_28_x86_64` container | RPM, TGZ | Auto-detects AlmaLinux |

## Package Naming Convention

Packages are named automatically by CPack using the RVS version from `CMakeLists.txt`:

- **DEB**: `amdrocm<ROCM_MAJOR>-rvs_${RVS_VERSION}_amd64.deb`
- **RPM**: `amdrocm<ROCM_MAJOR>-rvs-${RVS_VERSION}.x86_64.rpm`
- **TGZ**: `amdrocm<ROCM_MAJOR>-rvs-<RVS_VERSION>-<PACKAGE_RELEASE>-Linux.tar.gz` (CPack: same **release** suffix as DEB/RPM, e.g. `r0711.20260423` from `ROCM_LIBPATCH_VERSION` and date, or a PR-specific suffix)

The **patch version** in `RVS_VERSION` is automatically computed from the number of commits since the last `v<major>.<minor>.*` git tag. For example, with tag `v1.3.0` and 15 commits since, and a release of `r0711.20260423`:
```
amdrocm7-rvs_1.3.15-r0711.20260423_amd64.deb
amdrocm7-rvs-1.3.15-r0711.20260423.el8.x86_64.rpm
amdrocm7-rvs-1.3.15-r0711.20260423-Linux.tar.gz
```

`CPACK_PACKAGE_FILE_NAME` in CMake is set to include the same **release** as DEB/RPM (from `CPACK_RPM_PACKAGE_RELEASE` in the build environment).

If no matching `v` tag is found, the patch defaults to `0` from `project(VERSION)` in `CMakeLists.txt`.

**Important**: Package filenames match the internal package metadata, ensuring compliance with Debian and RPM standards. The version major and minor are sourced from `CMakeLists.txt` via CMake's `project(VERSION)` command, while the patch is auto-computed from git history at configure time.

## Installing Generated Packages

### Ubuntu/Debian (DEB)

```bash
# Download the package from S3 (or use the apt repo described above)
sudo dpkg -i amdrocm7-rvs_*.deb

# Run RVS
/opt/rocm/extras-7/bin/rvs --help
```

### CentOS/RHEL/Rocky Linux (RPM)

```bash
# Download the package from S3 (or use the yum/dnf repo described above)
sudo rpm -i --replacefiles --nodeps amdrocm7-rvs-*.rpm

# Run RVS
/opt/rocm/extras-7/bin/rvs --help
```

### Any Linux Distribution (TGZ - Relocatable)

#### Pre-install: ROCm

Before you install the RVS TGZ, **ROCm must be installed, configured, and on your `PATH` / `LD_LIBRARY_PATH` as in AMD’s documentation** so HIP, HSA, and other ROCm libraries are discoverable. Follow the current Linux install guide in **rocm docs**:

- **ROCm documentation (start here)**: <https://rocm.docs.amd.com/>
- **Linux install / deployment (paths, env, post-install)**: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>

**Assumptions (TGZ use):** ROCm is set up on the machine, `ROCM_PATH` (or your install prefix) is correct, and the runtime can load ROCm libraries. Install a ROCm stack that includes **ROCm’s LLVM** (for example the **`rocm-llvm`** package from the ROCm repo). **DEB/RPM** packages from this project declare that dependency; TGZ users should mirror that on the host. The TGZ only ships RVS; it does not replace a full ROCm stack.

#### Install RVS run-time dependencies (on the target system)

The TGZ is built against ROCm; on the target host you still need **PCI** for GPU enumeration and **NUMA** when using the bundled **TransferBench** CLI:

| Family | Typical PCI package | NUMA (TransferBench CLI) |
|--------|---------------------|---------------------------|
| **Debian / Ubuntu** | `libpci3` (or `libpci-3-0-0` on some releases) | `libnuma1` |
| **RHEL / Rocky / Alma 8+** | `pciutils-libs` | `numactl-libs` |
| **SUSE / openSUSE** | `libpci3` / `pciutils` as appropriate for the release | `libnuma1` |

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y libpci3 libnuma1

# RHEL / Rocky / Alma 8+
sudo dnf install -y pciutils-libs numactl-libs

# openSUSE / SUSE (adjust package names per release)
sudo zypper install libpci3 libnuma1
```

User-facing install steps for TGZ (PATH / `LD_LIBRARY_PATH`, **`RPATH`** including **`/opt/rocm/lib`**, **`/opt/rocm/lib/llvm/lib`**, **`/opt/rocm/core-<major>/lib`**, and **`/opt/rocm/core-<major>/lib/llvm/lib`**) are in **[docs/INSTALL_TGZ.md](../../docs/INSTALL_TGZ.md)**.

#### Extract the TGZ

```bash
# Extract to the extras directory (example for ROCm major 7; match your path)
sudo mkdir -p /opt/rocm/extras-7
sudo tar -xzf amdrocm7-rvs-*.tar.gz -C /opt/rocm/extras-7
```

#### Post-install: `PATH` and `LD_LIBRARY_PATH`

Point the shell at the extracted RVS prefix and the ROCm you installed (replace paths with your real `ROCM_PATH` and extras major version). Copy and paste the block as one unit:

```bash
export ROCM_PATH=/opt/rocm              # or your real ROCm root, per rocm docs
export PATH=/opt/rocm/extras-7/bin:$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:$LD_LIBRARY_PATH
```

**`rvs`** embeds **`RPATH`** for **`/opt/rocm/lib`**, **`/opt/rocm/lib/llvm/lib`**, **`core-<major>`** paths, and the extras-relative **`$ORIGIN`** paths, so it usually does not need **`LD_LIBRARY_PATH`** for them. The export still includes **`$ROCM_PATH/lib/llvm/lib`** for a typical ROCm tree, other tools, and troubleshooting; if LLVM is only under **`$ROCM_PATH/core-<major>/lib/llvm/lib`**, add that path instead or as well.

**Run RVS**

```bash
rvs --help
```

## Verifying Packages

The workflow automatically verifies package contents:

### DEB Package Verification

```bash
dpkg-deb -I amdrocm*-rvs_*.deb  # Package info
dpkg-deb -c amdrocm*-rvs_*.deb  # Package contents
```

### RPM Package Verification

```bash
rpm -qip amdrocm*-rvs-*.rpm  # Package info
rpm -qlp amdrocm*-rvs-*.rpm  # Package contents
rpm -qRp amdrocm*-rvs-*.rpm  # Package dependencies
```

## Accessing Build Packages

Packages are uploaded directly to **S3** (not GitHub Actions artifacts). To find them:

1. Go to the **Actions** tab in your GitHub repository
2. Click on the latest workflow run
3. Download the **`build-report`** artifact — it contains S3 console links to each package location
4. Or browse S3 directly using the path layout described above

## Customization

### Changing ROCm Version or GPU Family

**Option 1: Via GitHub Actions UI (Manual Trigger)**

1. Go to **Actions** → **Build Relocatable Packages**
2. Click **Run workflow**
3. Enter custom values for:
   - ROCm Version (e.g., `7.11.0a20260121`)
   - GPU Family (e.g., `gfx110X-all`)

**Option 2: Edit Workflow Defaults**

Edit the `env` section in `.github/workflows/build-relocatable-packages.yml`:

```yaml
env:
  ROCM_VERSION: '7.11.0a20260121'  # Change this
  GPU_FAMILY: 'gfx110X-all'         # Change this
  BUILD_TYPE: Release
```

**Option 3: Edit Build Script Defaults**

Edit `build_packages_local.sh`:

```bash
ROCM_VERSION="${ROCM_VERSION:-7.11.0a20260121}"  # Change default here
GPU_FAMILY="${GPU_FAMILY:-gfx110X-all}"          # Change default here
BUILD_TYPE="${BUILD_TYPE:-Release}"
```

### Adding More Distributions

To add support for more distributions, extend the workflow matrix:

**For Ubuntu variants:**

```yaml
build-ubuntu:
  strategy:
    matrix:
      ubuntu_version: ['20.04', '22.04', '24.04']
  runs-on: ubuntu-${{ matrix.ubuntu_version }}
```

**For other RPM-based distributions:**

```yaml
build-fedora:
  runs-on: ubuntu-latest
  container:
    image: fedora:39
  steps:
    - uses: actions/checkout@v4
    - run: |
        chmod +x build_packages_local.sh
        ROCM_VERSION=${{ env.ROCM_VERSION }} \
        GPU_FAMILY=${{ env.GPU_FAMILY }} \
        ./build_packages_local.sh
```

**For SLES/OpenSUSE:**

```yaml
build-sles:
  runs-on: ubuntu-latest
  container:
    image: opensuse/leap:15.5
  steps:
    - uses: actions/checkout@v4
    - run: |
        chmod +x build_packages_local.sh
        ROCM_VERSION=${{ env.ROCM_VERSION }} \
        GPU_FAMILY=${{ env.GPU_FAMILY }} \
        ./build_packages_local.sh
```

Note: You may need to update the OS detection logic in `build_packages_local.sh` to handle additional distributions.

### Customizing Build Parameters

Edit the CMake arguments in `build_packages_local.sh`, or override **`CMAKE_INSTALL_RPATH`** / **`CMAKE_SKIP_RPATH`** from the command line when needed. Defaults match packaged builds:

```bash
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DROCM_PATH="$ROCM_PATH" \
    -DHIP_PLATFORM=amd \
    -DROCM_MAJOR_VERSION="$ROCM_MAJOR" \
    -DCMAKE_INSTALL_PREFIX="/opt/rocm/extras-${ROCM_MAJOR}" \
    -DCPACK_PACKAGING_INSTALL_PREFIX="/opt/rocm/extras-${ROCM_MAJOR}" \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DFETCH_ROCMPATH_FROM_ROCMCORE=ON \
    -DYOUR_CUSTOM_OPTION=ON  # Add custom options here
```

## Troubleshooting

### S3: "Credentials could not be loaded"

- **PR from a fork:** S3 upload is skipped for fork PRs (secrets are not passed). Use a branch in the same repo or push to `main`/`master` to upload.
- **Same repo / push:** Ensure the `AWS_ROLE_ARN` secret is set (Settings → Secrets and variables → Actions → Secrets) and the IAM role’s trust policy allows GitHub OIDC for this repo. Ensure the OIDC identity provider exists in the AWS account (`token.actions.githubusercontent.com`).

### Package Build Fails

1. Check the workflow logs in GitHub Actions
2. Verify the ROCm tarball URL is accessible
3. Run `./build_packages_local.sh` locally to reproduce the issue
4. Check that all dependencies were installed correctly
5. Review CMake configuration output

### RPATH Issues

If binaries can't find libraries:

```bash
# Check RPATH settings (replace 7 with your ROCm major version)
readelf -d /opt/rocm/extras-7/bin/rvs | grep RPATH

# Should include: $ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs:/opt/rocm/lib:/opt/rocm/lib/llvm/lib:/opt/rocm/core-7/lib:/opt/rocm/core-7/lib/llvm/lib
```

### Missing Dependencies

If the package reports missing dependencies:

```bash
# Check what libraries are needed (replace 7 with your ROCm major version)
ldd /opt/rocm/extras-7/bin/rvs

# Install missing ROCm components if needed
```

### Local Testing

To test the workflow locally before pushing:

```bash
# Option 1: Auto-fetch latest ROCm version
sudo ./build_packages_local.sh

# Option 2: Set environment variables for custom configuration
export ROCM_VERSION=7.11.0a20260121
export GPU_FAMILY=gfx110X-all
export BUILD_TYPE=Release

# Run with sudo -E to preserve environment variables
sudo -E ./build_packages_local.sh

# Check generated packages
ls -lh build/amdrocm*-rvs*
```

## References

- [TheRock Releases Documentation](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [TheRock Nightly Tarballs](https://therock-nightly-tarball.s3.amazonaws.com/index.html)
- [Local Build Script](../build_packages_local.sh) - Core build engine used by workflow
- [Quick Start Guide](../QUICKSTART_PACKAGES.md) - Step-by-step local build instructions
- [Package Build Summary](../PACKAGE_BUILD_SUMMARY.md) - Technical overview and architecture
- [RVS Build Instructions](../README.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## Support

For issues with:
- **RVS Build**: Open an issue in this repository
- **ROCm SDK**: See [TheRock Issues](https://github.com/ROCm/TheRock/issues)
- **Workflow**: Check GitHub Actions documentation

## License

This workflow is part of ROCm Validation Suite and follows the same MIT license.
