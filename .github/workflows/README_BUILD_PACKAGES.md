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
- Manual trigger via GitHub Actions UI (workflow_dispatch)

### Manual Trigger Parameters

When manually triggering the workflow, you can specify:

1. **ROCm Version** (e.g., `6.5.0rc20250610`)
   - Default: `6.5.0rc20250610`
   
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
    │   ├── Ubuntu/Debian: apt-get install build-essential, cmake, etc.
    │   └── CentOS/RHEL: yum install gcc, gcc-c++, cmake3, etc.
    ├── 2. Download ROCm SDK from TheRock
    │   └── URL: https://therock-nightly-tarball.s3.../${GPU_FAMILY}-${ROCM_VERSION}.tar.gz
    ├── 3. Extract and Setup ROCm Environment
    │   ├── export ROCM_PATH="$HOME/rocm-sdk/install"
    │   ├── export PATH="$ROCM_PATH/bin:$PATH"
    │   ├── export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
    │   └── export CMAKE_PREFIX_PATH="$ROCM_PATH:$CMAKE_PREFIX_PATH"
    ├── 4. Configure CMake with Relocatable RPATH
    │   └── cmake -B ./build \
    │         -DCMAKE_BUILD_TYPE=Release \
    │         -DROCM_PATH=$ROCM_PATH \
    │         -DHIP_PLATFORM=amd \
    │         -DCMAKE_INSTALL_PREFIX=/opt/rocm/rvs \
    │         -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm/rvs \
    │         -DCMAKE_SKIP_RPATH=FALSE \
    │         -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
    │         -DCMAKE_INSTALL_RPATH="$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs"
    ├── 5. Build RVS
    │   └── make -C ./build -j$(nproc)
    └── 6. Create Packages
        ├── Ubuntu: DEB + TGZ (via CPack)
        └── CentOS/RHEL: RPM + TGZ (via CPack)
```

### Key Technical Details

**Relocatable RPATH**: The `$ORIGIN` relative RPATH ensures binaries can find their libraries regardless of installation location:

```bash
CMAKE_INSTALL_RPATH="$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs"
```

**Automatic Version Management**: CMake reads the project version from `CMakeLists.txt` and CPack uses it for package naming automatically.

**HIP Platform**: Set to `amd` via CMake parameter for AMD GPU support.

**Single Source of Truth**: All build logic resides in `build_packages_local.sh`, which can be used for both local builds and CI/CD.

### Workflow Steps

The GitHub Actions workflow performs minimal platform-specific operations:

1. **Checkout Repository** with recursive submodule initialization
2. **Set Environment Variables** from workflow inputs or defaults
3. **Execute Build Script** - `./build_packages_local.sh` handles everything
4. **Verify Packages** - Platform-specific verification (dpkg-deb or rpm -q)
5. **Upload Artifacts** - Store packages with 30-day retention

## Build Script: build_packages_local.sh

The workflow uses `build_packages_local.sh` as the core build engine. This script provides a complete, self-contained build system that works identically in both local development and CI/CD environments.

### Script Features

- **OS Detection**: Automatically identifies Ubuntu/Debian vs CentOS/RHEL
- **Dependency Management**: Installs all required build tools and libraries
- **ROCm SDK Setup**: Downloads and configures ROCm from TheRock tarballs
- **CMake Configuration**: Sets up relocatable RPATHs and all build parameters
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

## Build Matrix

The workflow builds packages for:

| Platform | Container/Runner | Package Types | Script Mode |
|----------|------------------|---------------|-------------|
| Ubuntu 22.04 | ubuntu-22.04 | DEB, TGZ | Auto-detects Ubuntu |
| Rocky Linux 8 | rockylinux:8 | RPM, TGZ | Auto-detects CentOS |

## Package Naming Convention

Packages are named automatically by CPack using the RVS version from `CMakeLists.txt`:

- **DEB**: `rocm-validation-suite_${RVS_VERSION}_amd64.deb`
- **RPM**: `rocm-validation-suite-${RVS_VERSION}.x86_64.rpm`
- **TGZ**: `rocm-validation-suite-${RVS_VERSION}-Linux.tar.gz`

Example (if RVS version is 1.0.0):
```
rocm-validation-suite_1.0.0_amd64.deb
rocm-validation-suite-1.0.0.x86_64.rpm
rocm-validation-suite-1.0.0-Linux.tar.gz
```

**Important**: Package filenames match the internal package metadata, ensuring compliance with Debian and RPM standards. The version is sourced directly from `CMakeLists.txt` via CMake's `project(VERSION)` command.

## Installing Generated Packages

### Ubuntu/Debian (DEB)

```bash
# Download the artifact from GitHub Actions
sudo dpkg -i rocm-validation-suite_*.deb

# Run RVS
/opt/rocm/rvs/bin/rvs --help
```

### CentOS/RHEL/Rocky Linux (RPM)

```bash
# Download the artifact from GitHub Actions
sudo rpm -i --replacefiles --nodeps rocm-validation-suite-*.rpm

# Run RVS
/opt/rocm/rvs/bin/rvs --help
```

### Any Linux Distribution (TGZ - Relocatable)

```bash
# Extract to /opt (or any location)
sudo mkdir -p /opt/rocm
sudo tar -xzf rocm-validation-suite-*.tar.gz -C /opt/

# Or extract to custom location
mkdir -p ~/myrocm
tar -xzf rocm-validation-suite-*.tar.gz -C ~/myrocm/

# Setup environment
export PATH=/opt/rocm/rvs/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/rvs/lib:$LD_LIBRARY_PATH

# Run RVS
rvs --help
```

## Verifying Packages

The workflow automatically verifies package contents:

### DEB Package Verification

```bash
dpkg-deb -I rocm-validation-suite_*.deb  # Package info
dpkg-deb -c rocm-validation-suite_*.deb  # Package contents
```

### RPM Package Verification

```bash
rpm -qip rocm-validation-suite-*.rpm  # Package info
rpm -qlp rocm-validation-suite-*.rpm  # Package contents
rpm -qRp rocm-validation-suite-*.rpm  # Package dependencies
```

## Accessing Build Artifacts

1. Go to the **Actions** tab in your GitHub repository
2. Click on the latest workflow run
3. Scroll down to **Artifacts** section
4. Download the package artifacts:
   - `ubuntu-22.04-packages-${GPU_FAMILY}`
   - `rockylinux8-packages-${GPU_FAMILY}`
   - `build-report` (contains build summary)

## Customization

### Changing ROCm Version or GPU Family

**Option 1: Via GitHub Actions UI (Manual Trigger)**

1. Go to **Actions** → **Build Relocatable Packages**
2. Click **Run workflow**
3. Enter custom values for:
   - ROCm Version (e.g., `6.5.0rc20250610`)
   - GPU Family (e.g., `gfx110X-all`)

**Option 2: Edit Workflow Defaults**

Edit the `env` section in `.github/workflows/build-relocatable-packages.yml`:

```yaml
env:
  ROCM_VERSION: '6.5.0rc20250610'  # Change this
  GPU_FAMILY: 'gfx110X-all'         # Change this
  BUILD_TYPE: Release
```

**Option 3: Edit Build Script Defaults**

Edit `build_packages_local.sh`:

```bash
ROCM_VERSION="${ROCM_VERSION:-6.5.0rc20250610}"  # Change default here
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
    -DCMAKE_INSTALL_PREFIX=/opt/rocm/rvs \
    -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm/rvs \
    -DCMAKE_SKIP_RPATH=FALSE \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rvs" \
    -DRPATH_MODE=OFF \
    -DYOUR_CUSTOM_OPTION=ON  # Add custom options here
```

## Troubleshooting

### Package Build Fails

1. Check the workflow logs in GitHub Actions
2. Verify the ROCm tarball URL is accessible
3. Run `./build_packages_local.sh` locally to reproduce the issue
4. Check that all dependencies were installed correctly
5. Review CMake configuration output

### RPATH Issues

If binaries can't find libraries:

```bash
# Check RPATH settings
readelf -d /opt/rocm/rvs/bin/rvs | grep RPATH

# Should show: $ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs
```

### Missing Dependencies

If the package reports missing dependencies:

```bash
# Check what libraries are needed
ldd /opt/rocm/rvs/bin/rvs

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
ls -lh build/rocm-validation-suite*
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
