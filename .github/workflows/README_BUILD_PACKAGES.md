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
- **Scheduled**: Daily at **5:00 AM PST** (13:00 UTC); uses **latest ROCm version** (auto-fetched from TheRock)
- Manual trigger via GitHub Actions UI (workflow_dispatch)

### Manual Trigger Parameters

When manually triggering the workflow, you can specify:

1. **ROCm Version** (e.g., `7.11.0a20260121`)
   - Default: empty (auto-fetches latest from TheRock)
   - Or specify an explicit version string
   
2. **GPU Family Target**:
   - `gfx94X-dcgpu` - MI300A/MI300X
   - `gfx950-dcgpu` - MI350X/MI355X
   - `gfx110X-all` - AMD RX 7900 XTX, 7800 XT, 7700S, Radeon 780M (default)
   - `gfx1151` - AMD Strix Halo iGPU
   - `gfx120X-all` - AMD RX 9060/XT, 9070/XT

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
    │   ├── export CMAKE_CXX_COMPILER=hipcc (AlmaLinux only)
    │   └── export CMAKE_COMMAND=cmake3 (AlmaLinux) or cmake (Ubuntu)
    ├── 4. Configure CMake with Relocatable RPATH
    │   └── $CMAKE_COMMAND -B ./build \
    │         -DCMAKE_BUILD_TYPE=Release \
    │         -DROCM_PATH=$ROCM_PATH \
    │         -DHIP_PLATFORM=amd \
    │         -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER (if set) \
    │         -DROCM_MAJOR_VERSION=$ROCM_MAJOR \
    │         -DCMAKE_INSTALL_PREFIX=/opt/rocm/extras-$ROCM_MAJOR \
    │         -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm/extras-$ROCM_MAJOR \
    │         -DCMAKE_SKIP_RPATH=FALSE \
    │         -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
    │         -DCMAKE_INSTALL_RPATH="$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs:/opt/rocm/extras-$ROCM_MAJOR/lib" \
    │         -DCMAKE_VERBOSE_MAKEFILE=1 \
    │         -DFETCH_ROCMPATH_FROM_ROCMCORE=ON
    ├── 5. Build RVS
    │   └── make -C ./build -j$(nproc)
    └── 6. Create Packages
        ├── Ubuntu: DEB + TGZ (via CPack)
        └── AlmaLinux: RPM + TGZ (via CPack)
```

### Key Technical Details

**Relocatable RPATH**: The `$ORIGIN` relative RPATH ensures binaries can find their libraries regardless of installation location:

```bash
CMAKE_INSTALL_RPATH="$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs:/opt/rocm/extras-<ROCM_MAJOR>/lib"
```

**Automatic Version Management**: CMake reads the project version from `CMakeLists.txt` and CPack uses it for package naming automatically. The **patch version** is auto-computed at CMake configure time: `git describe --tags --match "v<major>.<minor>.*"` counts commits since the last matching `v` tag. For example, if the tag is `v1.3.0` and there have been 15 commits since, the package version becomes `1.3.15`. If no matching tag exists or git is unavailable, the patch defaults to `0` from `CMakeLists.txt`. This works for both CI builds and direct local `cmake` invocations.

**HIP Device Libraries**: Automatically located and exported as `HIP_DEVICE_LIB_PATH` for clang device library discovery.

**Compiler Selection**: For AlmaLinux (manylinux_2_28), `CMAKE_CXX_COMPILER` is set to ROCm's `hipcc`. Ubuntu uses system default compiler.

**CMake Command**: Uses `cmake3` on AlmaLinux (manylinux_2_28) and `cmake` on Ubuntu.

**Verbose Build Output**: `CMAKE_VERBOSE_MAKEFILE=1` enables detailed compilation output for debugging and transparency.

**Dynamic ROCm Path Discovery**: `FETCH_ROCMPATH_FROM_ROCMCORE=ON` allows RVS to automatically detect ROCm installation location at runtime from ROCm core libraries.

**Single Source of Truth**: All build logic resides in `build_packages_local.sh`, which can be used for both local builds and CI/CD.

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

**S3 path layout:**

| Trigger | Path | Contents |
|--------|------|----------|
| **`release/*` branch** (`push` or `workflow_dispatch`) | `release/rvs/deb/`, `release/rvs/rpm/`, `release/rvs/tar/` | DEB → `.../deb` (Ubuntu job); RPM and TGZ → `.../rpm` and `.../tar` (manylinux job). Only PR merges into release branches or manual dispatch on release branches write here. |
| **Scheduled**, **push to `master`/`main`**, or **`workflow_dispatch` on non-release branch** | `nightly/rvs/deb/`, `nightly/rvs/rpm/`, `nightly/rvs/tar/` | Same split by type. All non-release builds go to nightly. |
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
# Add the nightly repo (replace <bucket> with the actual S3 bucket name)
echo "deb [trusted=yes] https://<bucket>.s3.amazonaws.com/nightly/rvs/deb/ ./" \
  | sudo tee /etc/apt/sources.list.d/rvs-nightly.list

# Or the release repo
echo "deb [trusted=yes] https://<bucket>.s3.amazonaws.com/release/rvs/deb/ ./" \
  | sudo tee /etc/apt/sources.list.d/rvs-release.list

sudo apt update
sudo apt install amdrocm7-rvs  # Replace 7 with your ROCm major version
```

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
  - Sets CMAKE_CXX_COMPILER to hipcc on AlmaLinux
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

**Important**: The script requires root privileges to install system dependencies. Use `sudo` when running locally. In GitHub Actions:
- **Ubuntu runners**: Use `sudo` (runner has sudo access but not root by default)
- **Container builds** (Rocky/CentOS): No sudo needed (containers run as root)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROCM_VERSION` | Auto-fetched (fallback: `7.11.0a20260121`) | ROCm SDK version from TheRock. If not set, script fetches latest version automatically. |
| `GPU_FAMILY` | `gfx110X-all` | Target GPU architecture |
| `BUILD_TYPE` | `Release` | CMake build type (Release/Debug) |
| `ROCM_LIBPATCH_VERSION` | Auto-extracted from `ROCM_VERSION` | Major.minor in xxyy format with zero padding (e.g., `7.11` → `0711`, `8.0` → `0800`) - used for RVS version tagging |
| `CPACK_DEBIAN_PACKAGE_RELEASE` | Auto-generated | **Default** (`schedule`, `push`, `workflow_dispatch`, local): `r<ROCM_LIBPATCH_VERSION>.<yyyymmdd>` (e.g. `r0711.20260423` where `0711` = ROCm 7.11 from `ROCM_VERSION`). **Pull requests**: `r<libpatch>.<yyyymmdd>.<source-branch>.<commit>`. **Release branches** (name starts with `rel`, non-PR): `GITHUB_RUN_NUMBER` (fallback: `1`). |
| `CPACK_RPM_PACKAGE_RELEASE` | same as `CPACK_DEBIAN_PACKAGE_RELEASE` | Identical to DEB. |
| `GITHUB_RUN_NUMBER` | `1` (local) | GitHub Actions run number - automatically set in CI, defaults to `1` for local builds |

## Build Matrix

The workflow builds packages for:

| Platform | Container/Runner | Package Types | Script Mode |
|----------|------------------|---------------|-------------|
| Ubuntu 22.04 | ubuntu-22.04 | DEB, TGZ | Auto-detects Ubuntu |
| Manylinux 2.28 (AlmaLinux 8) | manylinux_2_28_x86_64 | RPM, TGZ | Auto-detects AlmaLinux |

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

```bash
# Extract to the extras directory
sudo mkdir -p /opt/rocm/extras-7
sudo tar -xzf amdrocm7-rvs-*.tar.gz -C /opt/rocm/extras-7

# Setup environment
export PATH=/opt/rocm/extras-7/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:$LD_LIBRARY_PATH

# Run RVS
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

Edit the CMake configuration section in `build_packages_local.sh`:

```bash
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DROCM_PATH="$ROCM_PATH" \
    -DHIP_PLATFORM=amd \
    -DROCM_MAJOR_VERSION="$ROCM_MAJOR" \
    -DCMAKE_INSTALL_PREFIX="/opt/rocm/extras-${ROCM_MAJOR}" \
    -DCPACK_PACKAGING_INSTALL_PREFIX="/opt/rocm/extras-${ROCM_MAJOR}" \
    -DCMAKE_SKIP_RPATH=FALSE \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rvs:/opt/rocm/extras-${ROCM_MAJOR}/lib" \
    -DRPATH_MODE=OFF \
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

# Should show: $ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs:/opt/rocm/extras-7/lib
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
