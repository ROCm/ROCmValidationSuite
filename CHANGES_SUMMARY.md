# Changes Summary: Package Build System Evolution

## Overview
This document chronicles the evolution of the RVS package build system from manual workflow steps to a unified, automated build system using `build_packages_local.sh` as the single source of truth.

## Major Changes Timeline

### Phase 1: Initial Implementation
- Created GitHub Actions workflow for relocatable packages
- Used ROCm SDK from TheRock tarballs
- Set install path to `/opt/rocm/rvs`
- Configured relocatable RPATH with `$ORIGIN`

### Phase 2: Version Management Fix
- Changed package naming to use RVS version (from `CMakeLists.txt`) instead of ROCm version
- Fixed package metadata mismatch by removing manual `mv` renaming
- Let CPack handle all package naming automatically
- Ensured Debian and RPM standards compliance

### Phase 3: Removed Dynamic Version Extraction
- Removed manual `grep`-based version extraction from workflow
- Removed manual version extraction from build script
- Rely on CMake/CPack native version management from `project(VERSION)` in `CMakeLists.txt`
- Simplified code by eliminating redundant version handling

### Phase 4: Build System Unification (Current)
- Created `build_packages_local.sh` as single source of truth
- Refactored GitHub Actions workflow to delegate all build logic to script
- Added OS detection and automatic dependency installation to script
- Eliminated code duplication between Ubuntu and CentOS jobs
- Reduced workflow from ~380 lines to ~250 lines

## Current Architecture

### Workflow Structure

```
GitHub Actions Workflow (Minimal CI/CD Orchestration)
├── Checkout Repository
├── Set Environment Variables
└── Execute build_packages_local.sh (All Build Logic)
    ├── Detect OS (Ubuntu/Debian vs CentOS/RHEL)
    ├── Install Dependencies (platform-specific)
    ├── Download ROCm SDK from TheRock
    ├── Extract and Setup ROCm Environment
    ├── Configure CMake with relocatable RPATH + HIP_PLATFORM=amd
    ├── Build RVS (parallel make)
    └── Create Packages (DEB/RPM/TGZ via CPack)
```

### Key Technical Details

**Single Source of Truth**: `build_packages_local.sh`
- Works identically for local development and CI/CD
- Handles all platform-specific logic internally
- Self-contained: installs dependencies, builds, and packages

**Workflow Responsibilities** (GitHub Actions-specific only):
- Repository checkout with submodules
- Setting environment variables from workflow inputs
- Package verification (dpkg-deb, rpm -q)
- Artifact upload to GitHub

**Dependencies**: Automatically installed by script
- Ubuntu/Debian: `build-essential`, `cmake`, `git`, `wget`, `libpci3`, `libpci-dev`, `doxygen`, `unzip`, `libyaml-cpp-dev`, `rpm`
- CentOS/RHEL: `gcc`, `gcc-c++`, `cmake3`, `make`, `git`, `wget`, `pciutils-devel`, `doxygen`, `rpm-build`, `yaml-cpp-devel`, `yaml-cpp-static`

## Removed Components

### ❌ Removed: Dynamic Version Extraction
**Why**: CMake/CPack natively read version from `CMakeLists.txt` `project(VERSION)` command.

**Old Code (Removed)**:
```bash
# Extract RVS version from CMakeLists.txt
RVS_VERSION=$(grep -oP 'project\s*\(\s*"[^"]+"\s+VERSION\s+\K[\d.]+' CMakeLists.txt)
```

**Current**: CMake automatically uses version, CPack generates correct package names.

### ❌ Removed: Manual Package Renaming
**Why**: Renaming files caused metadata mismatch between filename and package header.

**The Problem**:
```bash
# CPack creates: rocm-validation-suite_1.3.0_amd64.deb
# Internal metadata: Package: rocm-validation-suite, Version: 1.3.0

# Then manual renaming breaks it:
mv rocm-validation-suite_1.3.0_amd64.deb \
   rocm-validation-suite_1.3.0_ubuntu22.04_gfx110X-all_amd64.deb
# ❌ Filename no longer matches package metadata!
```

**Issues Caused**:
1. **Metadata Mismatch**: `dpkg -I` shows different version than filename
2. **Repository Problems**: APT/YUM tools expect filename to match metadata
3. **Standards Violation**: Breaks Debian Policy and RPM conventions
4. **User Confusion**: `dpkg -l` shows different version than installed file

**The Fix**: Let CPack generate final filenames directly. CPack ensures filename and metadata always match.

**Old Code (Removed)**:
```bash
mv "rocm-validation-suite_1.3.0_amd64.deb" \
   "rocm-validation-suite_${RVS_VERSION}_ubuntu22.04_${GPU_FAMILY}_amd64.deb"
```

**Current**: CPack generates final filenames that comply with standards:
- DEB: `rocm-validation-suite_1.3.0_amd64.deb`
- RPM: `rocm-validation-suite-1.3.0.el8.x86_64.rpm`
- TGZ: `rocm-validation-suite-1.3.0-Linux.tar.gz`

### ❌ Removed: Patchelf Dependency
**Why**: Not used anywhere in RVS build or runtime.

**Removed from**: Dependency installation lists in workflow and script.

### ❌ Removed: Duplicate Build Logic in Workflow
**Why**: Code duplication between workflow and local script.

**Before**: 150+ lines of build steps duplicated for Ubuntu and CentOS jobs.
**After**: 6 lines calling `build_packages_local.sh`.

### ❌ Deleted: DYNAMIC_VERSION_EXTRACTION.md
**Why**: Documented an approach that was explicitly removed as unnecessary.

## Current Implementation

### GitHub Actions Workflow (`.github/workflows/build-relocatable-packages.yml`)

**Structure**: Minimal orchestration, delegates to build script

```yaml
- name: Build and Package RVS
  run: |
    chmod +x build_packages_local.sh
    ROCM_VERSION=${{ env.ROCM_VERSION }} \
    GPU_FAMILY=${{ env.GPU_FAMILY }} \
    BUILD_TYPE=${{ env.BUILD_TYPE }} \
    ./build_packages_local.sh
```

**Benefits**:
- ~250 lines total (down from ~380)
- No duplicate build logic
- Single source of truth
- Easy to test locally

### Build Script (`build_packages_local.sh`)

**Features**:
- OS detection (Ubuntu/Debian vs CentOS/RHEL)
- Automatic dependency installation
- ROCm SDK download and setup
- CMake configuration with:
  - `CMAKE_INSTALL_PREFIX=/opt/rocm/rvs`
  - `CPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm/rvs`
  - `HIP_PLATFORM=amd`
  - Relocatable RPATH: `$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/rvs`
- Parallel build with `make -j$(nproc)`
- Package creation via CPack

**Usage**:
```bash
# Default build
./build_packages_local.sh

# Custom configuration
ROCM_VERSION=7.11.0a20260121 GPU_FAMILY=gfx110X-all ./build_packages_local.sh
```

## Installation Path Changes

### Old Path Structure
```
/opt/rocm/
├── bin/
│   └── rvs
├── lib/
│   └── rvs/
└── share/
    └── rocm-validation-suite/
```

### New Path Structure
```
/opt/rocm/rvs/
├── bin/
│   └── rvs
├── lib/
│   └── rvs/
└── share/
    └── rocm-validation-suite/
```

### Benefits
- **Cleaner separation**: RVS is self-contained in its own directory
- **No conflicts**: Won't interfere with ROCm system installation
- **Easy removal**: Simple to uninstall (just delete `/opt/rocm/rvs`)
- **Multiple versions**: Could support multiple RVS versions side-by-side

## Updated Commands

### Running RVS After Installation

**Old:**
```bash
/opt/rocm/bin/rvs --version
```

**New:**
```bash
/opt/rocm/rvs/bin/rvs --version
```

### Environment Setup

**Old:**
```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

**New:**
```bash
export PATH=/opt/rocm/rvs/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/rvs/lib:$LD_LIBRARY_PATH
```

### TGZ Extraction

**Old:**
```bash
tar -xzf rocm-validation-suite-*.tar.gz -C /opt/
# Creates: /opt/opt/rocm/...
```

**New:**
```bash
tar -xzf rocm-validation-suite-*.tar.gz -C /
# Creates: /opt/rocm/rvs/...
```

## RPATH Configuration

Relocatable `$ORIGIN` references ensure portability:
```
$ORIGIN
$ORIGIN/../lib
$ORIGIN/../lib/rvs
```

This ensures packages work regardless of installation directory.

## Version Management

- **RVS Version**: Automatically read by CMake from `CMakeLists.txt` `project(VERSION)` command
- **ROCm SDK Version**: 7.11.0a20260121 (default, auto-fetched if not specified, configurable via environment variable)
- **GPU Family**: gfx110X-all (default, configurable via environment variable)

## Files Modified

### Core Build System
1. `.github/workflows/build-relocatable-packages.yml` - Simplified workflow
2. `build_packages_local.sh` - Complete build system script

### Documentation
3. `.github/workflows/README_BUILD_PACKAGES.md` - Workflow documentation
4. `QUICKSTART_PACKAGES.md` - Quick start guide
5. `PACKAGE_BUILD_SUMMARY.md` - Technical overview
6. `CHANGES_SUMMARY.md` - This file
7. `VERIFICATION_CHECKLIST.md` - Testing checklist

### Deleted Files
- `DYNAMIC_VERSION_EXTRACTION.md` - Obsolete version extraction documentation
- `PACKAGE_NAMING_FIX.md` - Consolidated into CHANGES_SUMMARY.md

## Benefits Summary

### Code Quality
✅ **DRY Principle**: Single source of truth eliminates duplication  
✅ **Maintainability**: Update build logic in one place  
✅ **Testability**: Same script for local and CI/CD  
✅ **Simplicity**: 250 lines vs 380 lines in workflow  

### Package Quality
✅ **Standards Compliance**: CPack-generated names match metadata  
✅ **Relocatable**: `$ORIGIN` RPATH works anywhere  
✅ **Version Accuracy**: Version from `CMakeLists.txt` automatically used  
✅ **Consistency**: Same packages from local and CI builds  

### Developer Experience
✅ **Local Testing**: Easy to test workflow changes locally  
✅ **Fast Iteration**: No need to push to test build changes  
✅ **Clear Output**: Color-coded progress indicators  
✅ **Error Handling**: Robust error checking at each step  

### Operations
✅ **Self-contained Install**: No conflicts with system ROCm  
✅ **Easy Uninstall**: Just delete `/opt/rocm/rvs`  
✅ **Parallel Versions**: Can install multiple RVS versions  
✅ **Platform Support**: Ubuntu, Debian, CentOS, RHEL, Rocky  

## Backward Compatibility

**Breaking Changes:**
- Install path changed from `/opt/rocm` to `/opt/rocm/rvs`
- Users with scripts expecting `/opt/rocm/bin/rvs` need to update to `/opt/rocm/rvs/bin/rvs`
- Package names now use RVS version instead of ROCm version

**Migration Guide for Users:**

```bash
# Uninstall old version
sudo dpkg -r rocm-validation-suite  # Ubuntu/Debian
sudo rpm -e rocm-validation-suite   # CentOS/RHEL

# Install new version
sudo dpkg -i rocm-validation-suite_*.deb  # Ubuntu/Debian
sudo rpm -i rocm-validation-suite-*.rpm   # CentOS/RHEL

# Update PATH and LD_LIBRARY_PATH
export PATH=/opt/rocm/rvs/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/rvs/lib:$LD_LIBRARY_PATH
```

## Summary

The RVS package build system has evolved from a verbose, duplicated workflow to a clean, unified system:

- **Single Script**: `build_packages_local.sh` handles everything
- **Minimal Workflow**: GitHub Actions orchestrates, script executes
- **Standards Compliant**: Proper package naming and metadata
- **Developer Friendly**: Easy to test locally before pushing
- **Production Ready**: Robust error handling and validation

This architecture ensures consistency between development and CI/CD while maintaining simplicity and maintainability.

---

**Date**: January 30, 2026  
**Status**: Complete  
**Architecture**: Unified build system with `build_packages_local.sh`
