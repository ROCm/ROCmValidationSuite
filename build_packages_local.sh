#!/bin/bash
################################################################################
# Local Build Script for Testing Package Generation
# This script mimics the GitHub Actions workflow for local testing
################################################################################

set -e  # Exit on error

# Configuration
GPU_FAMILY="${GPU_FAMILY:-gfx110X-all}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="./build"
ROCM_INSTALL_DIR="$HOME/rocm-sdk"

# SDK Source Configuration
# Default: TheRock public S3 nightly bucket (works from any network).
# Override via environment variables or GitHub Actions repository variables (vars.*).
#
# If your SDK server has no index page for auto-detection, set ROCM_SDK_INDEX_URL=""
# and provide ROCM_VERSION explicitly.
ROCM_SDK_BASE_URL="${ROCM_SDK_BASE_URL:-https://therock-nightly-tarball.s3.us-east-2.amazonaws.com}"
ROCM_SDK_INDEX_URL="${ROCM_SDK_INDEX_URL:-https://therock-nightly-tarball.s3.amazonaws.com/index.html}"

# Post-build upload configuration
# Set UPLOAD_TARGET to enable automatic upload of built packages after the build.
# Packages are organized into: <UPLOAD_TARGET>/<repo>/<branch>/<YYYY-MM-DD>/
#
# Supported formats:
#   Local path:  UPLOAD_TARGET="/mnt/shared/packages/"
#   SCP:         UPLOAD_TARGET="scp://user@buildserver:/opt/packages/"
#   Rsync:       UPLOAD_TARGET="rsync://user@buildserver:/opt/packages/"
#   HTTP PUT:    UPLOAD_TARGET="http://localhost:8080"
#                (use with packages_server/ nginx setup for auto directory creation)
#
# Internal-only convenience: if UPLOAD_TARGET is unset, set RVS_AUTO_DETECT_LOCAL_UPLOAD=1
# (e.g. in workflow env or shell) to probe http://localhost:8080/ once and use it when
# something responds. Default is off so external/CI builds never curl localhost; GitHub
# uploads use the workflow S3 steps, not Step 6 of this script.
#
# UPLOAD_REPO overrides the repo name in the upload path (auto-detected from git remote).
if [ -z "${UPLOAD_TARGET:-}" ] && [ -n "${RVS_AUTO_DETECT_LOCAL_UPLOAD:-}" ]; then
    if curl -s -o /dev/null -w '' http://localhost:8080/ 2>/dev/null; then
        UPLOAD_TARGET="http://localhost:8080"
    fi
fi
UPLOAD_TARGET="${UPLOAD_TARGET:-}"
UPLOAD_REPO="${UPLOAD_REPO:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to fetch latest ROCm version for specified GPU family
# Sets the global ROCM_VERSION variable directly
fetch_latest_rocm_version() {
    local gpu_family="$1"

    print_info "Fetching latest ROCm version for $gpu_family..."
    print_info "Index URL: $ROCM_SDK_INDEX_URL"

    # Download index page and parse for latest tarball matching GPU family
    # Pattern: therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz
    local latest_version=$(wget -qO- "$ROCM_SDK_INDEX_URL" 2>/dev/null | \
        grep -oP "therock-dist-linux-${gpu_family}-\K[^<\"]+(?=\.tar\.gz)" | \
        sort -V | tail -1)
    
    if [ -z "$latest_version" ]; then
        print_error "Could not fetch latest ROCm version for $gpu_family"
        print_info "Falling back to default version: 7.11.0a20260121"
        ROCM_VERSION="7.11.0a20260121"
        return 1
    fi
    
    print_success "Found latest ROCm version: $latest_version"
    ROCM_VERSION="$latest_version"
    return 0
}

# Function to check and install dependencies
check_and_install_dependencies() {
    print_info "Checking for required build dependencies..."
    
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        print_error "Cannot detect OS. Please install dependencies manually."
        exit 1
    fi
    
    # Check for command-line tools
    MISSING_TOOLS=()
    command -v cmake >/dev/null 2>&1 || MISSING_TOOLS+=("cmake")
    command -v make >/dev/null 2>&1 || MISSING_TOOLS+=("make")
    command -v gcc >/dev/null 2>&1 || MISSING_TOOLS+=("gcc")
    command -v g++ >/dev/null 2>&1 || MISSING_TOOLS+=("g++")
    command -v git >/dev/null 2>&1 || MISSING_TOOLS+=("git")
    command -v wget >/dev/null 2>&1 || MISSING_TOOLS+=("wget")
    command -v tar >/dev/null 2>&1 || MISSING_TOOLS+=("tar")
    command -v doxygen >/dev/null 2>&1 || MISSING_TOOLS+=("doxygen")
    command -v python3 >/dev/null 2>&1 || MISSING_TOOLS+=("python3")
    
    # Check for library dependencies (platform-specific)
    MISSING_LIBS=()
    if [[ "$OS" =~ ^(ubuntu|debian)$ ]]; then
        # Check for Ubuntu/Debian library headers
        [ -f /usr/include/pci/pci.h ] || MISSING_LIBS+=("libpci-dev")
        [ -f /usr/include/yaml-cpp/yaml.h ] || MISSING_LIBS+=("libyaml-cpp-dev")
        [ -f /usr/include/numa.h ] || MISSING_LIBS+=("libnuma-dev")
        command -v rpmbuild >/dev/null 2>&1 || MISSING_LIBS+=("rpm")
        command -v unzip >/dev/null 2>&1 || MISSING_LIBS+=("unzip")
    elif [[ "$OS" =~ ^(centos|rhel|rocky|almalinux|amzn)$ ]]; then
        # Check for CentOS/RHEL/Rocky/AlmaLinux library headers
        [ -f /usr/include/pci/pci.h ] || MISSING_LIBS+=("pciutils-devel")
        [ -f /usr/include/yaml-cpp/yaml.h ] || MISSING_LIBS+=("yaml-cpp-devel")
        [ -f /usr/include/numa.h ] || MISSING_LIBS+=("numactl-devel")
        command -v rpmbuild >/dev/null 2>&1 || MISSING_LIBS+=("rpm-build")
    fi
    
    # Combine missing tools and libraries
    MISSING_DEPS=()
    [ ${#MISSING_TOOLS[@]} -ne 0 ] && MISSING_DEPS+=("${MISSING_TOOLS[@]}")
    [ ${#MISSING_LIBS[@]} -ne 0 ] && MISSING_DEPS+=("${MISSING_LIBS[@]}")
    
    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        print_warning "Missing dependencies: ${MISSING_DEPS[*]}"
        echo ""
        
        if [[ "$OS" =~ ^(ubuntu|debian)$ ]]; then
            print_info "Installing dependencies for Ubuntu/Debian..."
            apt-get update
            apt-get install -y \
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
                python3 \
                libnuma-dev
        elif [[ "$OS" =~ ^(centos|rhel|rocky|almalinux|amzn)$ ]]; then
            print_info "Installing dependencies for CentOS/RHEL/Rocky/AlmaLinux..."
            
            # Check if running in manylinux container (has /opt/python)
            if [ -d /opt/python ]; then
                print_info "Detected manylinux environment - some tools may be pre-installed"
            fi
            
            # Enable PowerTools/CRB repository (contains doxygen and yaml-cpp)
            print_info "Enabling PowerTools/CRB repository..."
            if command -v dnf >/dev/null 2>&1; then
                # AlmaLinux/Rocky Linux 8+ uses dnf
                dnf install -y dnf-plugins-core 2>/dev/null || true
                dnf config-manager --set-enabled powertools 2>/dev/null || \
                dnf config-manager --set-enabled crb 2>/dev/null || \
                dnf config-manager --set-enabled devel 2>/dev/null || \
                print_warning "Could not enable PowerTools/CRB/devel repo (may already be enabled)"
            else
                # CentOS 8 uses yum
                yum install -y yum-utils 2>/dev/null || true
                yum-config-manager --enable powertools 2>/dev/null || \
                yum-config-manager --enable PowerTools 2>/dev/null || \
                yum-config-manager --enable devel 2>/dev/null || \
                print_warning "Could not enable PowerTools/devel repo (may already be enabled)"
            fi
            
            # Install EPEL repository (may already be present in manylinux)
            print_info "Installing EPEL repository..."
            yum install -y epel-release 2>/dev/null || print_warning "EPEL may already be installed"
            
            print_info "Installing build dependencies..."
            # Note: manylinux typically has gcc, g++, make pre-installed
            yum install -y \
                gcc \
                gcc-c++ \
                make \
                git \
                wget \
                tar \
                pciutils-devel \
                doxygen \
                rpm-build \
                python3 \
                numactl-devel \
                || print_warning "Some packages may already be installed"
            
            # Install a gcc-toolset with C++20 <barrier> support (requires GCC >= 11)
            # Try highest available first for best C++20/C++23 support, fall back to minimum viable
            # RHEL/AlmaLinux/Rocky 8+ use gcc-toolset-{ver}, CentOS 7 uses devtoolset-{ver}
            if [[ "$OS" =~ ^(centos|rhel|almalinux|rocky)$ ]]; then
                GCC_TOOLSET_INSTALLED=""
                for ver in 14 13 12 11; do
                    if yum list available gcc-toolset-${ver} &>/dev/null; then
                        print_info "Found gcc-toolset-${ver} - installing for C++20 support..."
                        if yum install -y gcc-toolset-${ver}; then
                            GCC_TOOLSET_INSTALLED="gcc-toolset-${ver}"
                            print_success "Installed gcc-toolset-${ver} (C++20 <barrier> supported)"
                            break
                        fi
                    fi
                done
                # Fallback to devtoolset (CentOS 7) - only devtoolset-11 has <barrier>
                if [ -z "$GCC_TOOLSET_INSTALLED" ]; then
                    if yum list available devtoolset-11 &>/dev/null; then
                        print_info "Found devtoolset-11 - installing for C++20 support (CentOS 7)..."
                        if yum install -y devtoolset-11; then
                            GCC_TOOLSET_INSTALLED="devtoolset-11"
                            print_success "Installed devtoolset-11 (C++20 <barrier> supported)"
                        fi
                    fi
                fi
                if [ -z "$GCC_TOOLSET_INSTALLED" ]; then
                    print_warning "No gcc-toolset (11-14) or devtoolset-11 available - C++20 headers like <barrier> may be missing"
                fi
            fi
            
            # Install cmake and yaml-cpp separately as they may need special handling
            print_info "Installing cmake..."
            yum install -y cmake3 || yum install -y cmake || print_warning "cmake installation may have failed"
            
            print_info "Installing yaml-cpp..."
            yum install -y yaml-cpp-devel yaml-cpp-static 2>/dev/null || \
            print_warning "yaml-cpp may not be available - will try to continue"
        else
            print_error "Unsupported OS: $OS"
            echo ""
            echo "Please install the following dependencies manually:"
            echo ""
            echo "Build Tools:"
            echo "  - gcc, g++, make"
            echo "  - cmake"
            echo "  - git"
            echo "  - wget"
            echo "  - tar"
            echo "  - doxygen"
            echo "  - python3"
            echo ""
            echo "Development Libraries:"
            echo "  - libpci-dev (or pciutils-devel)"
            echo "  - libyaml-cpp-dev (or yaml-cpp-devel)"
            echo "  - libnuma-dev (or numactl-devel)"
            echo "  - rpm-build tools"
            exit 1
        fi
        
        print_success "Dependencies installed successfully"
        echo ""
        
        # Verify installation by re-checking
        print_info "Verifying installation..."
        VERIFY_FAILED=()
        for tool in "${MISSING_TOOLS[@]}"; do
            if ! command -v "$tool" >/dev/null 2>&1; then
                VERIFY_FAILED+=("$tool")
            fi
        done
        
        if [ ${#VERIFY_FAILED[@]} -ne 0 ]; then
            print_error "Failed to install: ${VERIFY_FAILED[*]}"
            exit 1
        fi
        print_success "All dependencies verified"
    else
        print_success "All required dependencies found"
    fi
    echo ""
}

# Check and install dependencies (installs wget, etc. - required before fetch_latest_rocm_version)
check_and_install_dependencies

# Determine ROCm version: use environment variable or fetch latest (wget is now available)
if [ -n "$ROCM_VERSION" ]; then
    print_info "Using specified ROCm version: $ROCM_VERSION"
elif [ -n "$ROCM_SDK_INDEX_URL" ]; then
    # Auto-detection only works when an index URL is available (e.g. S3 bucket)
    print_info "No ROCm version specified, fetching latest from index..."
    fetch_latest_rocm_version "$GPU_FAMILY"

    if [ -z "$ROCM_VERSION" ]; then
        print_error "Failed to determine ROCm version"
        exit 1
    fi
else
    print_error "ROCM_VERSION is required"
    print_info "The configured SDK server does not provide an index page for auto-detection."
    print_info "Set it via environment variable, e.g.:"
    echo ""
    echo "  export ROCM_VERSION=\"7.11.0a20260121\""
    echo "  ./build_packages_local.sh"
    echo ""
    print_info "Or in GitHub Actions workflow:"
    echo ""
    echo "  env:"
    echo "    ROCM_VERSION: \"7.11.0a20260121\""
    echo ""
    exit 1
fi

# Print configuration
echo "================================================================================"
echo "  ROCm Validation Suite - Local Package Build Script"
echo "================================================================================"
print_info "ROCm Version: $ROCM_VERSION"
print_info "GPU Family: $GPU_FAMILY"
print_info "Build Type: $BUILD_TYPE"
print_info "Build Directory: $BUILD_DIR"
print_info "ROCm Install: $ROCM_INSTALL_DIR"
print_info "SDK Source: $ROCM_SDK_BASE_URL"
if [ -n "$UPLOAD_TARGET" ]; then
    print_info "Upload Target: $UPLOAD_TARGET"
fi
print_info "RVS Version: Will be read from CMakeLists.txt by CMake/CPack"
echo "================================================================================"
echo ""

# Step 1: Download ROCm SDK
print_info "Step 1: Downloading ROCm SDK tarball..."
TARBALL_URL="${ROCM_SDK_BASE_URL}/therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz"
TARBALL_FILE="$ROCM_INSTALL_DIR/rocm-sdk.tar.gz"

mkdir -p "$ROCM_INSTALL_DIR"

if [ -f "$ROCM_INSTALL_DIR/install/bin/hipconfig" ]; then
    print_warning "ROCm SDK already exists at $ROCM_INSTALL_DIR/install"
    if [ -t 0 ]; then
        read -p "Do you want to re-download? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$ROCM_INSTALL_DIR/install"
        else
            print_info "Using existing ROCm SDK"
            export ROCM_PATH="$ROCM_INSTALL_DIR/install"
            print_success "ROCm SDK path set to: $ROCM_PATH"
            echo ""
            goto_step2=true
        fi
    else
        print_info "Non-interactive mode: reusing existing ROCm SDK"
        export ROCM_PATH="$ROCM_INSTALL_DIR/install"
        print_success "ROCm SDK path set to: $ROCM_PATH"
        echo ""
        goto_step2=true
    fi
fi

if [ -z "$goto_step2" ]; then
    print_info "Downloading from: $TARBALL_URL"
    if wget --spider "$TARBALL_URL" 2>/dev/null; then
        wget --show-progress -O "$TARBALL_FILE" "$TARBALL_URL"
        print_success "Download complete"
    else
        print_error "Failed to download ROCm SDK tarball"
        print_error "URL: $TARBALL_URL"
        print_info "Please check if the ROCm version and GPU family are correct"
        exit 1
    fi
    
    # Extract tarball
    print_info "Extracting ROCm SDK..."
    mkdir -p "$ROCM_INSTALL_DIR/install"
    tar -xzf "$TARBALL_FILE" -C "$ROCM_INSTALL_DIR/install" --strip-components=1
    print_success "Extraction complete"
    
    export ROCM_PATH="$ROCM_INSTALL_DIR/install"
    print_success "ROCm SDK installed to: $ROCM_PATH"
fi
echo ""

# Step 2: Setup environment
print_info "Step 2: Setting up ROCm environment..."

# Find HIP device library path (amdgcn/bitcode) first
print_info "Locating HIP device libraries (amdgcn/bitcode)..."
HIP_DEVICE_LIB_PATH=$(find "$ROCM_PATH" -type d -path "*/amdgcn/bitcode" 2>/dev/null | head -1)

if [ -z "$HIP_DEVICE_LIB_PATH" ]; then
    print_error "Could not find amdgcn/bitcode directory in $ROCM_PATH"
    print_error "HIP device libraries are required for GPU code compilation"
    print_error "Please verify your ROCm SDK installation"
    exit 1
fi

# Export all environment variables
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH="$ROCM_PATH:$CMAKE_PREFIX_PATH"
export HIP_DEVICE_LIB_PATH="$HIP_DEVICE_LIB_PATH"

# Extract major.minor version from ROCM_VERSION for ROCM_LIBPATCH_VERSION
# Convert to xxyy format with zero padding
# Example: "7.11.0a20260121" -> "0711", "8.0.0" -> "0800", "10.2.0" -> "1002"
ROCM_VERSION_MAJOR_MINOR=$(echo "$ROCM_VERSION" | grep -oP '^\d+\.\d+')
if [ -z "$ROCM_VERSION_MAJOR_MINOR" ]; then
    print_error "Could not extract major.minor version from ROCM_VERSION: $ROCM_VERSION"
    exit 1
fi

# Split into major and minor, zero-pad to 2 digits each
ROCM_MAJOR=$(echo "$ROCM_VERSION_MAJOR_MINOR" | cut -d'.' -f1)
ROCM_MINOR=$(echo "$ROCM_VERSION_MAJOR_MINOR" | cut -d'.' -f2)
ROCM_LIBPATCH_VERSION=$(printf "%02d%02d" "$ROCM_MAJOR" "$ROCM_MINOR")

export ROCM_LIBPATCH_VERSION
export ROCM_MAJOR
print_success "Set ROCM_LIBPATCH_VERSION=$ROCM_LIBPATCH_VERSION, ROCM_MAJOR=$ROCM_MAJOR (from $ROCM_VERSION_MAJOR_MINOR)"

# Set CPACK package release based on event/branch type
# - Scheduled builds: yyyymmdd date stamp
# - Release branches (starting with "rel"): GitHub run number
# - Dev/PR builds: branch.commit (same for DEB and RPM)
# Prefer GITHUB_REF_NAME / GITHUB_SHA (always available in Actions, unaffected
# by sudo safe.directory restrictions); fall back to git for local builds.
GIT_BRANCH="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")}"
if [ -n "$GITHUB_SHA" ]; then
    GIT_COMMIT_SHORT="${GITHUB_SHA:0:7}"
else
    GIT_COMMIT_SHORT=$(git rev-parse --short HEAD 2>/dev/null || echo "0000000")
fi

if [ "$GITHUB_EVENT_NAME" = "schedule" ]; then
    PACKAGE_RELEASE="$(date +%Y%m%d)"
    print_success "Set CPACK package release: $PACKAGE_RELEASE (scheduled build)"
elif [[ "$GIT_BRANCH" =~ ^rel ]]; then
    GITHUB_RUN_NUMBER="${GITHUB_RUN_NUMBER:-1}"
    PACKAGE_RELEASE="$GITHUB_RUN_NUMBER"
    print_success "Set CPACK package release: $PACKAGE_RELEASE (release branch: $GIT_BRANCH, run: $GITHUB_RUN_NUMBER)"
else
    SANITIZED_BRANCH=$(echo "$GIT_BRANCH" | sed 's/[^A-Za-z0-9.+~]/./g')
    PACKAGE_RELEASE="${SANITIZED_BRANCH}.${GIT_COMMIT_SHORT}"
    print_success "Set CPACK package release: $PACKAGE_RELEASE (dev branch: $GIT_BRANCH, commit: $GIT_COMMIT_SHORT)"
fi

export CPACK_DEBIAN_PACKAGE_RELEASE="$PACKAGE_RELEASE"
export CPACK_RPM_PACKAGE_RELEASE="$PACKAGE_RELEASE"

if [[ "$OS" =~ ^(centos|rhel|almalinux|rocky)$ ]]; then
    # Enable the installed gcc-toolset/devtoolset so hipcc (clang) finds C++20 headers like <barrier>
    # Discover the toolset path dynamically via rpm rather than hardcoding /opt/rh/

    GCC_TOOLSET_ENABLED=""
    for pkg in gcc-toolset-14 gcc-toolset-13 gcc-toolset-12 gcc-toolset-11 devtoolset-11; do
        ENABLE_SCRIPT=$(rpm -ql ${pkg}-runtime 2>/dev/null | grep '/enable$' | head -1)
        if [ -n "$ENABLE_SCRIPT" ] && [ -f "$ENABLE_SCRIPT" ]; then
            TOOLSET_ROOT=$(dirname "$ENABLE_SCRIPT")
            source "$ENABLE_SCRIPT"
            export GCC_TOOLCHAIN="${TOOLSET_ROOT}/root/usr"
            GCC_TOOLSET_ENABLED="$pkg"
            print_success "Enabled ${pkg} from ${TOOLSET_ROOT} (GCC $(gcc -dumpversion))"
            break
        fi
    done
    if [ -z "$GCC_TOOLSET_ENABLED" ]; then
        print_warning "No gcc-toolset (11-14) or devtoolset-11 found - C++20 headers may be missing"
    fi

    if [ -x "$ROCM_PATH/bin/hipcc" ]; then
        export CMAKE_CXX_COMPILER="$ROCM_PATH/bin/hipcc"
        print_success "Set CMAKE_CXX_COMPILER to hipcc (CentOS/RHEL/AlmaLinux/Rocky)"
    else
        print_warning "hipcc not found at $ROCM_PATH/bin/hipcc - using system default compiler"
    fi
    
    # Use cmake3 for RHEL-based distros
    export CMAKE_COMMAND="cmake3"
    print_info "Using cmake3 (CentOS/RHEL/AlmaLinux/Rocky)"
else
    # Ubuntu and others: use defaults (system compiler and cmake)
    export CMAKE_COMMAND="cmake"
    print_info "Using cmake (Ubuntu/other)"
fi

print_info "Environment variables set:"
echo "  ROCM_PATH=$ROCM_PATH"
echo "  PATH includes: $ROCM_PATH/bin"
echo "  LD_LIBRARY_PATH includes: $ROCM_PATH/lib"
echo "  HIP_DEVICE_LIB_PATH=$HIP_DEVICE_LIB_PATH"
echo "  ROCM_LIBPATCH_VERSION=$ROCM_LIBPATCH_VERSION"
if [ -n "$CPACK_DEBIAN_PACKAGE_RELEASE" ]; then
    echo "  CPACK_DEBIAN_PACKAGE_RELEASE=$CPACK_DEBIAN_PACKAGE_RELEASE"
fi
if [ -n "$CPACK_RPM_PACKAGE_RELEASE" ]; then
    echo "  CPACK_RPM_PACKAGE_RELEASE=$CPACK_RPM_PACKAGE_RELEASE"
fi
if [ -n "$CMAKE_CXX_COMPILER" ]; then
    echo "  CMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER"
fi
if [ -n "$GCC_TOOLCHAIN" ]; then
    echo "  GCC_TOOLCHAIN=$GCC_TOOLCHAIN"
fi
echo "  CMAKE_COMMAND=$CMAKE_COMMAND"

# Verify ROCm installation
print_info "Verifying ROCm installation..."
if [ -x "$ROCM_PATH/bin/hipconfig" ]; then
    print_success "hipconfig found"
else
    print_warning "hipconfig not found or not executable"
fi

if [ -d "$ROCM_PATH/lib" ]; then
    LIB_COUNT=$(ls -1 "$ROCM_PATH/lib"/*.so 2>/dev/null | wc -l)
    print_success "Found $LIB_COUNT shared libraries in $ROCM_PATH/lib"
else
    print_error "ROCm lib directory not found"
    exit 1
fi
echo ""

# Step 3: Configure CMake
print_info "Step 3: Configuring CMake with relocatable paths..."
if [ -d "$BUILD_DIR" ]; then
    print_warning "Build directory exists. Cleaning..."
    rm -rf "$BUILD_DIR"
fi

# Build cmake command with optional CXX compiler
# TODO: May be cleanup RPATH later with lesser options.
CMAKE_ARGS=(
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DROCM_PATH="$ROCM_PATH"
    -DHIP_PLATFORM=amd
    -DROCM_MAJOR_VERSION="$ROCM_MAJOR"
    -DCPACK_PACKAGE_NAME="amdrocm${ROCM_MAJOR}-rvs"
    -DCMAKE_INSTALL_PREFIX="/opt/rocm/extras-${ROCM_MAJOR}"
    -DCPACK_PACKAGING_INSTALL_PREFIX="/opt/rocm/extras-${ROCM_MAJOR}"
    -DCMAKE_SKIP_RPATH=FALSE
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE
    -DCMAKE_INSTALL_RPATH="\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rvs:/opt/rocm/extras-${ROCM_MAJOR}/lib"
    -DRPATH_MODE=OFF
    -DCMAKE_VERBOSE_MAKEFILE=1
    -DFETCH_ROCMPATH_FROM_ROCMCORE=ON
)

# Add CXX compiler if set
if [ -n "$CMAKE_CXX_COMPILER" ]; then
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="$CMAKE_CXX_COMPILER")
fi

# Point hipcc at the GCC toolchain for C++20 standard library headers
if [ -n "$GCC_TOOLCHAIN" ]; then
    CMAKE_ARGS+=(-DCMAKE_CXX_FLAGS="--gcc-toolchain=$GCC_TOOLCHAIN")
    print_success "Set --gcc-toolchain=$GCC_TOOLCHAIN for hipcc"
fi

$CMAKE_COMMAND "${CMAKE_ARGS[@]}"

if [ $? -eq 0 ]; then
    print_success "CMake configuration successful"
else
    print_error "CMake configuration failed"
    exit 1
fi
echo ""

# Step 4: Build RVS
print_info "Step 4: Building RVS..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
print_info "Using $NPROC parallel jobs"

make -C "$BUILD_DIR" -j"$NPROC"

if [ $? -eq 0 ]; then
    print_success "Build successful"
else
    print_error "Build failed"
    exit 1
fi
echo ""

# Step 5: Create packages
print_info "Step 5: Creating packages..."

# Create DEB package (Ubuntu/Debian)
if command -v dpkg >/dev/null 2>&1; then
    print_info "Creating DEB package..."
    cd "$BUILD_DIR"
    cpack -G DEB --verbose
    
    if [ $? -eq 0 ]; then
        DEB_FILE=$(ls amdrocm*-rvs*.deb 2>/dev/null | head -1)
        if [ -n "$DEB_FILE" ]; then
            print_success "Created DEB package: $DEB_FILE"
            DEB_SIZE=$(du -h "$DEB_FILE" | cut -f1)
            print_info "Package size: $DEB_SIZE"
            
            # Verify DEB package
            print_info "Verifying DEB package..."
            dpkg-deb -I "$DEB_FILE" | head -20
        fi
    else
        print_warning "DEB package creation failed (might be expected on non-Debian systems)"
    fi
    cd - > /dev/null
fi

# Create RPM package (CentOS/RHEL/SLES)
if command -v rpm >/dev/null 2>&1; then
    print_info "Creating RPM package..."
    cd "$BUILD_DIR"
    cpack -G RPM --verbose

    
    if [ $? -eq 0 ]; then
        RPM_FILE=$(ls amdrocm*-rvs*.rpm 2>/dev/null | head -1)
        if [ -n "$RPM_FILE" ]; then
            print_success "Created RPM package: $RPM_FILE"
            RPM_SIZE=$(du -h "$RPM_FILE" | cut -f1)
            print_info "Package size: $RPM_SIZE"
            
            # Verify RPM package
            print_info "Verifying RPM package..."
            rpm -qip "$RPM_FILE" | head -20
        fi
    else
        print_warning "RPM package creation failed (might be expected on non-RPM systems)"
    fi
    cd - > /dev/null
fi

# Create TGZ package
print_info "Creating TGZ package..."
CMAKE_ARGS+=(-DCPACK_SET_DESTDIR=ON -DCPACK_MONOLITHIC_INSTALL=ON -DCPACK_PACKAGING_INSTALL_PREFIX:PATH=/ -DCPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF)
$CMAKE_COMMAND "${CMAKE_ARGS[@]}"
cd "$BUILD_DIR"
cpack -G TGZ --verbose

if [ $? -eq 0 ]; then
    TGZ_FILE=$(ls amdrocm*-rvs*.tar.gz 2>/dev/null | head -1)
    if [ -n "$TGZ_FILE" ]; then
        print_success "Created TGZ package: $TGZ_FILE"
        TGZ_SIZE=$(du -h "$TGZ_FILE" | cut -f1)
        print_info "Package size: $TGZ_SIZE"
    fi
else
    print_error "TGZ package creation failed"
fi

cd - > /dev/null
echo ""

# Step 6: Upload packages (optional)
if [ -n "$UPLOAD_TARGET" ]; then
    print_info "Step 6: Uploading packages to $UPLOAD_TARGET..."

    # Build organized upload subpath: <repo>/<branch>/<date>
    if [ -z "$UPLOAD_REPO" ]; then
        UPLOAD_REPO="${GITHUB_REPOSITORY##*/}"
        [ -z "$UPLOAD_REPO" ] && UPLOAD_REPO=$(git remote get-url origin 2>/dev/null | sed 's|.*/||; s|\.git$||')
        [ -z "$UPLOAD_REPO" ] && UPLOAD_REPO=$(basename "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null)
        [ -z "$UPLOAD_REPO" ] && UPLOAD_REPO="unknown"
    fi
    UPLOAD_BRANCH="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")}"
    UPLOAD_BRANCH=$(echo "$UPLOAD_BRANCH" | sed 's|[^a-zA-Z0-9._-]|-|g')
    UPLOAD_DATE=$(date +%Y-%m-%d)
    UPLOAD_SUBPATH="${UPLOAD_REPO}/${UPLOAD_BRANCH}/${UPLOAD_DATE}"

    print_info "Upload path: .../${UPLOAD_SUBPATH}/"

    PKGS=$(find "$BUILD_DIR" -maxdepth 1 -name 'amdrocm*-rvs*' \( -name '*.deb' -o -name '*.rpm' -o -name '*.tar.gz' \) 2>/dev/null)
    if [ -z "$PKGS" ]; then
        print_error "No packages found to upload"
    else
        UPLOAD_OK=true

        case "$UPLOAD_TARGET" in
            scp://*)
                SCP_DEST="${UPLOAD_TARGET#scp://}"
                SCP_DEST="${SCP_DEST%/}/${UPLOAD_SUBPATH}/"
                print_info "Uploading via SCP to $SCP_DEST"
                # Create remote directory first
                SCP_HOST="${SCP_DEST%%:*}"
                SCP_PATH="${SCP_DEST#*:}"
                ssh "$SCP_HOST" "mkdir -p '$SCP_PATH'" 2>/dev/null || true
                for pkg in $PKGS; do
                    print_info "  $(basename "$pkg")"
                    if ! scp "$pkg" "$SCP_DEST"; then
                        print_error "SCP upload failed for $(basename "$pkg")"
                        UPLOAD_OK=false
                    fi
                done
                ;;
            rsync://*)
                RSYNC_DEST="${UPLOAD_TARGET#rsync://}"
                RSYNC_DEST="${RSYNC_DEST%/}/${UPLOAD_SUBPATH}/"
                print_info "Uploading via rsync to $RSYNC_DEST"
                if ! echo "$PKGS" | xargs -I{} rsync -avz --progress {} "$RSYNC_DEST"; then
                    print_error "Rsync upload failed"
                    UPLOAD_OK=false
                fi
                ;;
            http://*|https://*)
                UPLOAD_URL="${UPLOAD_TARGET%/}/${UPLOAD_SUBPATH}"
                print_info "Uploading via HTTP PUT to $UPLOAD_URL/"
                for pkg in $PKGS; do
                    local_name=$(basename "$pkg")
                    print_info "  $local_name"
                    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
                        -X PUT -T "$pkg" \
                        "${UPLOAD_URL}/${local_name}")
                    if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
                        print_success "  Uploaded $local_name (HTTP $HTTP_CODE)"
                    else
                        print_error "  Failed to upload $local_name (HTTP $HTTP_CODE)"
                        UPLOAD_OK=false
                    fi
                done
                # Show shareable URL (resolve localhost to real IP for team sharing)
                if [[ "$UPLOAD_TARGET" =~ localhost|127\.0\.0\.1 ]]; then
                    SHARE_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)
                    [ -z "$SHARE_IP" ] && SHARE_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
                    if [ -n "$SHARE_IP" ]; then
                        SHARE_URL=$(echo "$UPLOAD_URL" | sed "s|localhost|$SHARE_IP|;s|127\.0\.0\.1|$SHARE_IP|")
                        print_info "Share this URL with your team: ${SHARE_URL}/"
                    fi
                fi
                print_info "Browse uploads: ${UPLOAD_URL}/"
                ;;
            *)
                LOCAL_DEST="${UPLOAD_TARGET%/}/${UPLOAD_SUBPATH}"
                print_info "Copying packages to $LOCAL_DEST"
                mkdir -p "$LOCAL_DEST"
                for pkg in $PKGS; do
                    print_info "  $(basename "$pkg")"
                    if ! cp "$pkg" "$LOCAL_DEST/"; then
                        print_error "Copy failed for $(basename "$pkg")"
                        UPLOAD_OK=false
                    fi
                done
                ;;
        esac

        if [ "$UPLOAD_OK" = true ]; then
            print_success "All packages uploaded successfully"
        else
            print_warning "Some uploads failed - check errors above"
        fi
    fi
    echo ""
fi

# Summary
echo "================================================================================"
print_success "Package build completed successfully!"
echo "================================================================================"
print_info "Generated packages are in: $BUILD_DIR"
echo ""
# List only package types that were actually generated (DEB on Ubuntu, RPM on CentOS/RHEL, TGZ on all)
PKGS=$(find "$BUILD_DIR" -maxdepth 1 -name 'amdrocm*-rvs*' \( -name '*.deb' -o -name '*.rpm' -o -name '*.tar.gz' \) 2>/dev/null)
if [ -n "$PKGS" ]; then
    echo "$PKGS" | xargs ls -lh
else
    print_warning "No packages found in build directory"
fi
echo ""

# Installation instructions
echo "================================================================================"
echo "  Installation Instructions"
echo "================================================================================"
echo ""
echo "Ubuntu/Debian (DEB):"
echo "  sudo dpkg -i $BUILD_DIR/amdrocm${ROCM_MAJOR}-rvs_*.deb"
echo ""
echo "CentOS/RHEL (RPM):"
echo "  sudo rpm -i --replacefiles --nodeps $BUILD_DIR/amdrocm${ROCM_MAJOR}-rvs-*.rpm"
echo ""
echo "Any Linux (TGZ - Relocatable):"
echo "  sudo mkdir -p /opt/rocm/extras-${ROCM_MAJOR}"
echo "  sudo tar -xzf $BUILD_DIR/amdrocm${ROCM_MAJOR}-rvs-*.tar.gz -C /opt/rocm/extras-${ROCM_MAJOR}"
echo "  export PATH=/opt/rocm/extras-${ROCM_MAJOR}/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=/opt/rocm/extras-${ROCM_MAJOR}/lib:\$LD_LIBRARY_PATH"
echo ""
echo "================================================================================"

print_success "Done!"
