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
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to fetch latest ROCm version for specified GPU family
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
        echo "7.11.0a20260121"
        return 1
    fi
    
    print_success "Found latest ROCm version: $latest_version"
    echo "$latest_version"
    return 0
}

# Determine ROCm version: use environment variable or fetch latest
if [ -n "$ROCM_VERSION" ]; then
    print_info "Using specified ROCm version: $ROCM_VERSION"
else
    print_info "No ROCm version specified, fetching latest..."
    ROCM_VERSION=$(fetch_latest_rocm_version "$GPU_FAMILY")
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
    
    # Check for library dependencies (platform-specific)
    MISSING_LIBS=()
    if [[ "$OS" =~ ^(ubuntu|debian)$ ]]; then
        # Check for Ubuntu/Debian library headers
        [ -f /usr/include/pci/pci.h ] || MISSING_LIBS+=("libpci-dev")
        [ -f /usr/include/yaml-cpp/yaml.h ] || MISSING_LIBS+=("libyaml-cpp-dev")
        command -v rpmbuild >/dev/null 2>&1 || MISSING_LIBS+=("rpm")
        command -v unzip >/dev/null 2>&1 || MISSING_LIBS+=("unzip")
    elif [[ "$OS" =~ ^(centos|rhel|rocky)$ ]]; then
        # Check for CentOS/RHEL library headers
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
                rpm
        elif [[ "$OS" =~ ^(centos|rhel|rocky)$ ]]; then
            print_info "Installing dependencies for CentOS/RHEL..."
            yum install -y epel-release
            yum install -y \
                gcc \
                gcc-c++ \
                cmake3 \
                make \
                git \
                wget \
                pciutils-devel \
                doxygen \
                rpm-build \
                yaml-cpp-devel \
                yaml-cpp-static
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
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH="$ROCM_PATH:$CMAKE_PREFIX_PATH"

print_info "Environment variables set:"
echo "  ROCM_PATH=$ROCM_PATH"
echo "  PATH includes: $ROCM_PATH/bin"
echo "  LD_LIBRARY_PATH includes: $ROCM_PATH/lib"

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

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DROCM_PATH="$ROCM_PATH" \
    -DHIP_PLATFORM=amd \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm/rvs \
    -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm/rvs \
    -DCMAKE_SKIP_RPATH=FALSE \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rvs" \
    -DRPATH_MODE=OFF

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
