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
    local index_url="https://therock-nightly-tarball.s3.amazonaws.com/index.html"
    
    print_info "Fetching latest ROCm version for $gpu_family from TheRock..."
    
    # Download index page and parse for latest tarball matching GPU family
    # Pattern: therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz
    local latest_version=$(wget -qO- "$index_url" 2>/dev/null | \
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

# Determine ROCm version: use environment variable or fetch latest
if [ -n "$ROCM_VERSION" ]; then
    print_info "Using specified ROCm version: $ROCM_VERSION"
else
    print_info "No ROCm version specified, fetching latest..."
    fetch_latest_rocm_version "$GPU_FAMILY"
    
    # Validate that ROCM_VERSION was properly set
    if [ -z "$ROCM_VERSION" ]; then
        print_error "Failed to determine ROCm version"
        exit 1
    fi
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
print_info "RVS Version: Will be read from CMakeLists.txt by CMake/CPack"
echo "================================================================================"
echo ""

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
        command -v rpmbuild >/dev/null 2>&1 || MISSING_LIBS+=("rpm")
        command -v unzip >/dev/null 2>&1 || MISSING_LIBS+=("unzip")
    elif [[ "$OS" =~ ^(centos|rhel|rocky|almalinux|amzn)$ ]]; then
        # Check for CentOS/RHEL/Rocky/AlmaLinux library headers
        [ -f /usr/include/pci/pci.h ] || MISSING_LIBS+=("pciutils-devel")
        [ -f /usr/include/yaml-cpp/yaml.h ] || MISSING_LIBS+=("yaml-cpp-devel")
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
                python3
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
                || print_warning "Some packages may already be installed"
            
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

# Check and install dependencies
check_and_install_dependencies

# Step 1: Download ROCm SDK
print_info "Step 1: Downloading ROCm SDK tarball..."
TARBALL_URL="https://therock-nightly-tarball.s3.us-east-2.amazonaws.com/therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz"
TARBALL_FILE="$ROCM_INSTALL_DIR/rocm-sdk.tar.gz"

mkdir -p "$ROCM_INSTALL_DIR"

if [ -f "$ROCM_INSTALL_DIR/install/bin/hipconfig" ]; then
    print_warning "ROCm SDK already exists at $ROCM_INSTALL_DIR/install"
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
print_success "Set ROCM_LIBPATCH_VERSION=$ROCM_LIBPATCH_VERSION (from $ROCM_VERSION_MAJOR_MINOR)"

# Set CPACK package release based on branch type
# - Non-release branches (not starting with "rel"): use branch.commit format
# - Release branches (starting with "rel"): use GitHub run number
# Prefer GitHub Actions env vars (checkout is detached HEAD so git rev-parse --abbrev-ref returns "HEAD")
# GITHUB_HEAD_REF = source branch (set for pull_request; empty for push/schedule)
if [ -n "${GITHUB_HEAD_REF:-}" ]; then
    GIT_BRANCH="$GITHUB_HEAD_REF"
else
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
fi
if [ -n "${GITHUB_SHA:-}" ]; then
    GIT_COMMIT_SHORT="${GITHUB_SHA:0:7}"
else
    GIT_COMMIT_SHORT=$(git rev-parse --short HEAD 2>/dev/null || echo "0000000")
fi

if [[ "$GIT_BRANCH" =~ ^rel ]]; then
    # Release branch: use GitHub run number (fallback to 1 for local builds)
    GITHUB_RUN_NUMBER="${GITHUB_RUN_NUMBER:-1}"
    PACKAGE_RELEASE="$GITHUB_RUN_NUMBER"
    
    export CPACK_DEBIAN_PACKAGE_RELEASE="$PACKAGE_RELEASE"
    export CPACK_RPM_PACKAGE_RELEASE="$PACKAGE_RELEASE"
    
    print_success "Set CPACK package release: $PACKAGE_RELEASE (release branch: $GIT_BRANCH, run: $GITHUB_RUN_NUMBER)"
else
    # Development branch: use branch name and commit
    # Sanitize branch name: replace / and other special chars with -
    SANITIZED_BRANCH=$(echo "$GIT_BRANCH" | sed 's/[^a-zA-Z0-9._-]/-/g')
    PACKAGE_RELEASE="${SANITIZED_BRANCH}.${GIT_COMMIT_SHORT}"
    
    export CPACK_DEBIAN_PACKAGE_RELEASE="$PACKAGE_RELEASE"
    export CPACK_RPM_PACKAGE_RELEASE="$PACKAGE_RELEASE"
    
    print_success "Set CPACK package release: $PACKAGE_RELEASE (dev branch: $GIT_BRANCH, commit: $GIT_COMMIT_SHORT)"
fi

# TODO: Currently hipcc and cmake3 are only set for AlmaLinux (manylinux_2_28)
# Ubuntu works fine without these settings. Need to verify later if these
# should be applied universally for all OS or kept OS-specific.

# AlmaLinux-specific settings for manylinux_2_28
if [[ "$OS" == "almalinux" ]]; then
    # Set C++ compiler to hipcc from ROCm for AlmaLinux
    if [ -x "$ROCM_PATH/bin/hipcc" ]; then
        export CMAKE_CXX_COMPILER="$ROCM_PATH/bin/hipcc"
        print_success "Set CMAKE_CXX_COMPILER to hipcc (AlmaLinux)"
    else
        print_warning "hipcc not found at $ROCM_PATH/bin/hipcc - using system default compiler"
    fi
    
    # Use cmake3 for AlmaLinux
    export CMAKE_COMMAND="cmake3"
    print_info "Using cmake3 (AlmaLinux/Manylinux)"
else
    # Ubuntu: use defaults (system compiler and cmake)
    export CMAKE_COMMAND="cmake"
    print_info "Using cmake (Ubuntu)"
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
CMAKE_ARGS=(
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DROCM_PATH="$ROCM_PATH"
    -DHIP_PLATFORM=amd
    -DCMAKE_INSTALL_PREFIX=/opt/rocm/rvs
    -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm/rvs
    -DCMAKE_SKIP_RPATH=FALSE
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE
    -DCMAKE_INSTALL_RPATH="\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rvs"
    -DRPATH_MODE=OFF
    -DCMAKE_VERBOSE_MAKEFILE=1
    -DFETCH_ROCMPATH_FROM_ROCMCORE=ON
)

# Add CXX compiler if set
if [ -n "$CMAKE_CXX_COMPILER" ]; then
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="$CMAKE_CXX_COMPILER")
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

# Detect OS type
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_NAME=$ID
    OS_VERSION=$VERSION_ID
else
    OS_NAME="unknown"
    OS_VERSION="unknown"
fi

print_info "Detected OS: $OS_NAME $OS_VERSION"

# Create TGZ package
print_info "Creating TGZ package..."
cd "$BUILD_DIR"
cpack -G TGZ

if [ $? -eq 0 ]; then
    TGZ_FILE=$(ls rocm-validation-suite*.tar.gz 2>/dev/null | head -1)
    if [ -n "$TGZ_FILE" ]; then
        print_success "Created TGZ package: $TGZ_FILE"
        TGZ_SIZE=$(du -h "$TGZ_FILE" | cut -f1)
        print_info "Package size: $TGZ_SIZE"
    fi
else
    print_error "TGZ package creation failed"
fi

# Create DEB package (Ubuntu/Debian)
if command -v dpkg >/dev/null 2>&1; then
    print_info "Creating DEB package..."
    cpack -G DEB
    
    if [ $? -eq 0 ]; then
        DEB_FILE=$(ls rocm-validation-suite*.deb 2>/dev/null | head -1)
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
fi

# Create RPM package (CentOS/RHEL/SLES)
if command -v rpm >/dev/null 2>&1; then
    print_info "Creating RPM package..."
    cpack -G RPM
    
    if [ $? -eq 0 ]; then
        RPM_FILE=$(ls rocm-validation-suite*.rpm 2>/dev/null | head -1)
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
fi

cd - > /dev/null
echo ""

# Summary
echo "================================================================================"
print_success "Package build completed successfully!"
echo "================================================================================"
print_info "Generated packages are in: $BUILD_DIR"
echo ""
ls -lh "$BUILD_DIR"/rocm-validation-suite*{.deb,.rpm,.tar.gz} 2>/dev/null || \
    print_warning "No packages found in build directory"
echo ""

# Installation instructions
echo "================================================================================"
echo "  Installation Instructions"
echo "================================================================================"
echo ""
echo "Ubuntu/Debian (DEB):"
echo "  sudo dpkg -i $BUILD_DIR/rocm-validation-suite_*.deb"
echo ""
echo "CentOS/RHEL (RPM):"
echo "  sudo rpm -i --replacefiles --nodeps $BUILD_DIR/rocm-validation-suite-*.rpm"
echo ""
echo "Any Linux (TGZ - Relocatable):"
echo "  sudo tar -xzf $BUILD_DIR/rocm-validation-suite-*.tar.gz -C /"
echo "  export PATH=/opt/rocm/rvs/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=/opt/rocm/rvs/lib:\$LD_LIBRARY_PATH"
echo ""
echo "================================================================================"

print_success "Done!"
