# Quick Start Guide: Building Relocatable RVS Packages

This guide will help you quickly build relocatable RPM, DEB, and TGZ packages for ROCm Validation Suite using the ROCm SDK from TheRock.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Option 1: Using GitHub Actions (Recommended)](#option-1-using-github-actions-recommended)
- [Option 2: Building Locally](#option-2-building-locally)
- [GPU Family Selection](#gpu-family-selection)
- [Testing Packages](#testing-packages)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### For GitHub Actions (Automated Build)
- GitHub repository with Actions enabled
- No additional software required (everything runs in the cloud)

### For Local Build
Install the following tools on your Linux system:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libpci3 \
    libpci-dev \
    doxygen \
    unzip \
    libyaml-cpp-dev \
    rpm \
    python3
```

**CentOS/RHEL/Rocky Linux:**
```bash
sudo yum install -y epel-release
sudo yum install -y \
    gcc gcc-c++ cmake3 make git wget \
    pciutils-devel doxygen rpm-build \
    yaml-cpp-devel yaml-cpp-static \
    python3
```

---

## Option 1: Using GitHub Actions (Recommended)

### Automatic Builds (Push/PR)

The workflow automatically runs when you:
1. Push to `master`, `main`, or `release/**` branches
2. Create a pull request to `master` or `main`

### Manual Build

1. **Go to GitHub Actions**
   - Navigate to your repository on GitHub
   - Click on the **Actions** tab
   - Select **Build Relocatable Packages** workflow

2. **Click "Run workflow"**
   - Choose the branch to run from
   - (Optional) Set custom ROCm version (e.g., `7.11.0a20260121`)
     - Default: `7.11.0a20260121`
     - If not specified, script auto-fetches latest version from TheRock
   - (Optional) Select GPU family target
     - Default: `gfx110X-all` (AMD RX 7000 series)
     - Options: `gfx94X-dcgpu`, `gfx950-dcgpu`, `gfx110X-all`, `gfx1151`, `gfx120X-all`

3. **Wait for Build to Complete**
   - Build typically takes 20-30 minutes
   - You can monitor progress in real-time

4. **Download Artifacts**
   - Scroll down to the **Artifacts** section
   - Download the package artifacts:
     - `ubuntu-22.04-packages-${GPU_FAMILY}` - Contains DEB and TGZ
     - `manylinux_2_28-packages-${GPU_FAMILY}` - Contains RPM and TGZ
     - `build-report` - Build summary and details

---

## Option 2: Building Locally

### Quick Build (Default Settings)

```bash
# Make the script executable
chmod +x build_packages_local.sh

# Run the build script with sudo (required for installing dependencies)
sudo ./build_packages_local.sh
```

**Note**: The script requires root privileges to install system dependencies. In GitHub Actions, Ubuntu runners use `sudo` while container builds (Manylinux/AlmaLinux) run as root directly.

This will:
1. Automatically detect and install missing dependencies
2. Fetch latest ROCm SDK version from TheRock (or use default 7.11.0a20260121)
3. Locate HIP device libraries (amdgcn/bitcode)
4. Configure and build RVS with relocatable RPATH
5. Generate DEB, RPM, and TGZ packages
6. Save packages to `./build/` directory

### Custom Build

Set environment variables before running:

```bash
# Set custom ROCm version
export ROCM_VERSION="7.11.0a20260121"

# Set GPU family
export GPU_FAMILY="gfx94X-dcgpu"  # For MI300A/MI300X

# Set build type
export BUILD_TYPE="Release"  # or "Debug"

# Run the build with sudo
sudo -E ./build_packages_local.sh
```

**Note**: Use `sudo -E` to preserve environment variables when running with sudo.

### Manual Build (Step-by-Step)

If you prefer manual control:

```bash
# 1. Download ROCm SDK
ROCM_VERSION="6.5.0rc20250610"
GPU_FAMILY="gfx110X-all"
mkdir -p ~/rocm-sdk
cd ~/rocm-sdk
wget "https://therock-nightly-tarball.s3.us-east-2.amazonaws.com/therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz"
mkdir -p install
tar -xzf therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz -C install --strip-components=1

# 2. Setup environment
export ROCM_PATH="$HOME/rocm-sdk/install"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH="$ROCM_PATH:$CMAKE_PREFIX_PATH"

# 3. Configure CMake
cd /path/to/ROCmValidationSuite
cmake -B ./build \
    -DCMAKE_BUILD_TYPE=Release \
    -DROCM_PATH=$ROCM_PATH \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm \
    -DCMAKE_SKIP_RPATH=FALSE \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rvs"

# 4. Build
make -C ./build -j$(nproc)

# 5. Create packages
cd ./build
cpack -G DEB   # Debian package
cpack -G RPM   # RPM package
cpack -G TGZ   # Tarball (relocatable)
```

---

## GPU Family Selection

Choose the appropriate GPU family for your target hardware:

| GPU Family | Supported Hardware | Use Case |
|------------|-------------------|----------|
| `gfx94X-dcgpu` | MI300A, MI300X | Data center GPUs |
| `gfx950-dcgpu` | MI350X, MI355X | Next-gen data center GPUs |
| `gfx110X-all` | RX 7900 XTX, RX 7800 XT, RX 7700S, Radeon 780M | Consumer/workstation GPUs (default) |
| `gfx1151` | Strix Halo iGPU | Integrated GPUs |
| `gfx120X-all` | RX 9060/XT, RX 9070/XT | Latest consumer GPUs |

### Example: Building for MI300X

**GitHub Actions:**
- Select `gfx94X-dcgpu` when running workflow manually

**Local Build:**
```bash
export GPU_FAMILY="gfx94X-dcgpu"
./build_packages_local.sh
```

---

## Testing Packages

### Test DEB Package (Ubuntu/Debian)

```bash
# Install package
sudo dpkg -i build/rocm-validation-suite_*.deb

# Verify installation
/opt/rocm/rvs/bin/rvs --version

# Run a simple test
/opt/rocm/rvs/bin/rvs -g

# Uninstall (if needed)
sudo dpkg -r rocm-validation-suite
```

### Test RPM Package (CentOS/RHEL)

```bash
# Install package
sudo rpm -i --replacefiles --nodeps build/rocm-validation-suite-*.rpm

# Verify installation
/opt/rocm/rvs/bin/rvs --version

# Run a simple test
/opt/rocm/rvs/bin/rvs -g

# Uninstall (if needed)
sudo rpm -e rocm-validation-suite
```

### Test TGZ Package (Any Linux)

```bash
# Extract to test location
mkdir -p ~/test-rvs
tar -xzf build/rocm-validation-suite-*.tar.gz -C ~/test-rvs/

# Setup environment
export PATH="$HOME/test-rvs/opt/rocm/rvs/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/test-rvs/opt/rocm/rvs/lib:$LD_LIBRARY_PATH"

# Verify installation
rvs --version

# Run a simple test
rvs -g

# Check RPATH (should show $ORIGIN references)
readelf -d ~/test-rvs/opt/rocm/rvs/bin/rvs | grep RPATH
```

### Verify Relocatable RPATH

Ensure the binaries use relative paths:

```bash
# Check RPATH settings
readelf -d /opt/rocm/rvs/bin/rvs | grep RPATH

# Expected output should include:
# $ORIGIN/../lib
# $ORIGIN/../lib/rvs
```

---

## Troubleshooting

### Build Fails: ROCm SDK Download Error

**Problem:** Cannot download ROCm tarball

**Solution:**
1. Check if the URL is accessible:
   ```bash
   wget --spider "https://therock-nightly-tarball.s3.us-east-2.amazonaws.com/therock-dist-linux-gfx110X-all-6.5.0rc20250610.tar.gz"
   ```

2. Verify the ROCm version exists:
   - Visit: https://therock-nightly-tarball.s3.amazonaws.com/index.html
   - Find the correct version string

3. Try a different ROCm version:
   ```bash
   export ROCM_VERSION="6.5.0rc20250115"  # Use different date
   ```

### Build Fails: Missing Dependencies

**Problem:** CMake cannot find ROCm libraries

**Solution:**
1. Verify ROCm SDK extraction:
   ```bash
   ls -la ~/rocm-sdk/install/lib/
   ls -la ~/rocm-sdk/install/bin/
   ```

2. Check environment variables:
   ```bash
   echo $ROCM_PATH
   echo $CMAKE_PREFIX_PATH
   ```

3. Re-extract the tarball if corrupted

### Package Install Fails: Dependency Errors

**Problem:** Package manager reports missing dependencies

**Solution:**
1. For DEB packages:
   ```bash
   sudo apt-get install -f  # Fix missing dependencies
   ```

2. For RPM packages:
   ```bash
   sudo yum install rocm-smi-lib rocblas amd-smi-lib  # Install ROCm runtime
   ```

3. Use TGZ package (no system dependencies required)

### Runtime Error: Libraries Not Found

**Problem:** `rvs: error while loading shared libraries`

**Solution:**
1. Set LD_LIBRARY_PATH:
   ```bash
   export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
   ```

2. Check library dependencies:
   ```bash
   ldd /opt/rocm/bin/rvs
   ```

3. Verify RPATH is set correctly:
   ```bash
   readelf -d /opt/rocm/bin/rvs | grep RPATH
   ```

### GitHub Actions: Workflow Not Triggering

**Problem:** Workflow doesn't run on push/PR

**Solution:**
1. Check if GitHub Actions is enabled:
   - Settings → Actions → Allow all actions

2. Verify workflow file syntax:
   ```bash
   # Validate YAML locally
   python -c "import yaml; yaml.safe_load(open('.github/workflows/build-relocatable-packages.yml'))"
   ```

3. Check branch name matches trigger conditions

---

## Additional Resources

- **Detailed Documentation:** [`.github/workflows/README_BUILD_PACKAGES.md`](.github/workflows/README_BUILD_PACKAGES.md)
- **ROCm Documentation:** https://rocm.docs.amd.com/
- **TheRock Releases:** https://github.com/ROCm/TheRock/blob/main/RELEASES.md
- **RVS User Guide:** [docs/ug1main.md](docs/ug1main.md)
- **RVS Features:** [FEATURES.md](FEATURES.md)

---

## Quick Reference

### Package Locations After Build

- **Local build:** `./build/rocm-validation-suite-*`
- **GitHub Actions:** Download from Actions → Artifacts section

### Default Configuration

- **RVS Version:** Read from CMakeLists.txt by CMake/CPack
- **ROCm Version:** Auto-fetched from TheRock (fallback: 7.11.0a20260121)
- **GPU Family:** gfx110X-all (AMD RX 7000 series)
- **Install Prefix:** /opt/rocm/rvs
- **Build Type:** Release

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCM_VERSION` | ROCm SDK version to download | Auto-fetched (fallback: 7.11.0a20260121) |
| `GPU_FAMILY` | Target GPU architecture | gfx110X-all |
| `BUILD_TYPE` | CMake build type | Release |

**Note:** 
- RVS version is read from `CMakeLists.txt` by CMake/CPack automatically.
- ROCm version is automatically fetched from TheRock. Set `ROCM_VERSION` to override.
- HIP device libraries (amdgcn/bitcode) are auto-located and exported.

---

## Getting Help

For issues or questions:

1. **Check the logs:** Review build output for error messages
2. **Review documentation:** See detailed guides in `.github/workflows/`
3. **Open an issue:** Create a GitHub issue with:
   - Build log output
   - System information (OS, kernel version)
   - ROCm version and GPU family used
   - Error messages

---

**Happy Building!** 🚀
