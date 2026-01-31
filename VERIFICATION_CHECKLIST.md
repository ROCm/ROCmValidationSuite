# Implementation Verification Checklist

Use this checklist to verify that the relocatable package build system is working correctly.

## ✅ Pre-Deployment Checklist

### Files Created
- [ ] `.github/workflows/build-relocatable-packages.yml` exists
- [ ] `.github/workflows/README_BUILD_PACKAGES.md` exists
- [ ] `build_packages_local.sh` exists and is executable
- [ ] `QUICKSTART_PACKAGES.md` exists
- [ ] `PACKAGE_BUILD_SUMMARY.md` exists
- [ ] `VERIFICATION_CHECKLIST.md` exists (this file)

### File Permissions
```bash
# Verify script is executable
ls -la build_packages_local.sh
# Should show: -rwxr-xr-x or similar

# Make executable if needed
chmod +x build_packages_local.sh
```

### Syntax Validation
```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/build-relocatable-packages.yml'))" && echo "✓ YAML syntax OK"

# Validate Bash syntax
bash -n build_packages_local.sh && echo "✓ Bash syntax OK"

# Check for common issues
grep -n "TODO\|FIXME\|XXX" .github/workflows/*.yml build_packages_local.sh || echo "✓ No TODO markers"
```

## ✅ Local Build Testing

### Test 1: Script Execution (Dry Run)
```bash
# Check if script can be executed
./build_packages_local.sh --help 2>&1 | head -5
# Should show usage information or start running
```

### Test 2: Dependency Check
```bash
# Verify required tools are available
for tool in cmake make wget tar patchelf git; do
    command -v $tool >/dev/null 2>&1 && echo "✓ $tool found" || echo "✗ $tool missing"
done
```

### Test 3: ROCm SDK Download Test
```bash
# Test if ROCm tarball URL is accessible
ROCM_VERSION="6.5.0rc20250610"
GPU_FAMILY="gfx110X-all"
TARBALL_URL="https://therock-nightly-tarball.s3.us-east-2.amazonaws.com/therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz"

wget --spider "$TARBALL_URL" 2>&1 | grep "200 OK" && echo "✓ Tarball accessible" || echo "✗ Tarball not found"
```

### Test 4: Full Local Build (Optional - Takes 30+ minutes)
```bash
# Run full build locally
export ROCM_VERSION="6.5.0rc20250610"
export GPU_FAMILY="gfx110X-all"
./build_packages_local.sh

# Verify packages were created
ls -lh ./build/rocm-validation-suite*.{deb,rpm,tar.gz} 2>/dev/null || echo "✗ No packages found"
```

### Test 5: Package Verification
```bash
# If DEB was created
if [ -f build/rocm-validation-suite*.deb ]; then
    dpkg-deb -I build/rocm-validation-suite*.deb && echo "✓ DEB package valid"
    dpkg-deb -c build/rocm-validation-suite*.deb | grep "opt/rocm/bin/rvs" && echo "✓ RVS binary in package"
fi

# If RPM was created
if [ -f build/rocm-validation-suite*.rpm ]; then
    rpm -qip build/rocm-validation-suite*.rpm && echo "✓ RPM package valid"
    rpm -qlp build/rocm-validation-suite*.rpm | grep "opt/rocm/bin/rvs" && echo "✓ RVS binary in package"
fi

# If TGZ was created
if [ -f build/rocm-validation-suite*.tar.gz ]; then
    tar -tzf build/rocm-validation-suite*.tar.gz | head -20 && echo "✓ TGZ archive valid"
    tar -tzf build/rocm-validation-suite*.tar.gz | grep "opt/rocm/bin/rvs" && echo "✓ RVS binary in archive"
fi
```

### Test 6: RPATH Verification
```bash
# After building, check if RPATH is set correctly
if [ -f ./build/bin/rvs ]; then
    readelf -d ./build/bin/rvs | grep -i "rpath" | grep "\$ORIGIN" && echo "✓ RPATH uses \$ORIGIN" || echo "✗ RPATH not relocatable"
fi
```

## ✅ GitHub Actions Testing

### Test 1: Workflow File Validation
- [ ] Go to GitHub repository
- [ ] Navigate to **Actions** tab
- [ ] Check if "Build Relocatable Packages" workflow appears
- [ ] No syntax error warnings shown

### Test 2: Manual Workflow Trigger
- [ ] Go to Actions → Build Relocatable Packages
- [ ] Click "Run workflow"
- [ ] Select branch (e.g., `master`)
- [ ] Leave default values or customize:
  - [ ] ROCm Version: (leave default or enter custom)
  - [ ] GPU Family: (select from dropdown)
- [ ] Click "Run workflow" button
- [ ] Workflow starts successfully

### Test 3: Workflow Execution Monitor
- [ ] Workflow shows "In progress" status
- [ ] Click on the running workflow
- [ ] All jobs appear (build-ubuntu, build-centos)
- [ ] Jobs are running (yellow spinner icons)
- [ ] No immediate failures

### Test 4: Build Logs Review
For each job (build-ubuntu, build-centos):
- [ ] "Download ROCm SDK Tarball" step succeeds
- [ ] ROCm SDK extraction completes
- [ ] CMake configuration succeeds
- [ ] Build completes without errors
- [ ] Packages are created
- [ ] Artifacts are uploaded

### Test 5: Artifact Download
After workflow completes:
- [ ] Scroll to "Artifacts" section at bottom
- [ ] Verify artifacts exist:
  - [ ] `ubuntu-22.04-packages-${GPU_FAMILY}`
  - [ ] `rockylinux8-packages-${GPU_FAMILY}`
  - [ ] `build-report`
- [ ] Download one artifact
- [ ] Extract and verify contents:
  ```bash
  unzip ubuntu-22.04-packages-*.zip
  ls -lh rocm-validation-suite*
  ```

### Test 6: Automatic Trigger Test
```bash
# Create a test commit to trigger workflow
git add VERIFICATION_CHECKLIST.md
git commit -m "test: verify GitHub Actions workflow trigger"
git push origin master  # or your default branch

# Check if workflow triggered automatically
# Go to Actions tab and verify new workflow run started
```

## ✅ Package Installation Testing

### Test 7: DEB Package Installation (Ubuntu/Debian)
```bash
# Download DEB package from artifacts
# Install test
sudo dpkg -i rocm-validation-suite_*.deb

# Verify installation
[ -f /opt/rocm/bin/rvs ] && echo "✓ RVS binary installed"
/opt/rocm/bin/rvs --version && echo "✓ RVS runs"
/opt/rocm/bin/rvs -g && echo "✓ RVS GPU detection works"

# Check libraries
ldd /opt/rocm/bin/rvs | grep "not found" && echo "✗ Missing libraries" || echo "✓ All libraries found"

# Uninstall
sudo dpkg -r rocm-validation-suite
```

### Test 8: RPM Package Installation (CentOS/RHEL)
```bash
# Download RPM package from artifacts
# Install test
sudo rpm -i --replacefiles --nodeps rocm-validation-suite-*.rpm

# Verify installation
[ -f /opt/rocm/bin/rvs ] && echo "✓ RVS binary installed"
/opt/rocm/bin/rvs --version && echo "✓ RVS runs"
/opt/rocm/bin/rvs -g && echo "✓ RVS GPU detection works"

# Check libraries
ldd /opt/rocm/bin/rvs | grep "not found" && echo "✗ Missing libraries" || echo "✓ All libraries found"

# Uninstall
sudo rpm -e rocm-validation-suite
```

### Test 9: TGZ Package Relocatable Test
```bash
# Download TGZ package from artifacts
# Test extraction to custom location
mkdir -p ~/test-rvs-install
tar -xzf rocm-validation-suite-*.tar.gz -C ~/test-rvs-install/

# Setup environment
export PATH="$HOME/test-rvs-install/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/test-rvs-install/opt/rocm/lib:$LD_LIBRARY_PATH"

# Verify it works from custom location
[ -f ~/test-rvs-install/opt/rocm/bin/rvs ] && echo "✓ RVS binary extracted"
rvs --version && echo "✓ RVS runs from custom location"
rvs -g && echo "✓ RVS GPU detection works"

# Verify RPATH
readelf -d ~/test-rvs-install/opt/rocm/bin/rvs | grep RPATH | grep "\$ORIGIN" && echo "✓ RPATH is relocatable"

# Check libraries resolve
ldd ~/test-rvs-install/opt/rocm/bin/rvs | grep "not found" && echo "✗ Missing libraries" || echo "✓ All libraries found"

# Cleanup
rm -rf ~/test-rvs-install
```

## ✅ Documentation Review

### Test 10: Documentation Completeness
- [ ] `QUICKSTART_PACKAGES.md` has clear instructions
- [ ] `.github/workflows/README_BUILD_PACKAGES.md` explains workflow
- [ ] `PACKAGE_BUILD_SUMMARY.md` summarizes implementation
- [ ] All example commands are accurate
- [ ] All file paths are correct
- [ ] GPU family table is complete
- [ ] ROCm version references are current

### Test 11: Link Validation
```bash
# Check for broken links in documentation
for file in QUICKSTART_PACKAGES.md .github/workflows/README_BUILD_PACKAGES.md PACKAGE_BUILD_SUMMARY.md; do
    echo "Checking $file..."
    grep -oE 'https?://[^)]+' "$file" | while read url; do
        curl -I -s -o /dev/null -w "%{http_code}" "$url" | grep -q "^[23]" && echo "  ✓ $url" || echo "  ✗ $url"
    done
done
```

## ✅ Edge Cases & Error Handling

### Test 12: Invalid ROCm Version
```bash
# Test with non-existent ROCm version
export ROCM_VERSION="invalid-version"
./build_packages_local.sh 2>&1 | grep -i "error\|fail" && echo "✓ Error handled correctly"
```

### Test 13: Missing Dependencies
```bash
# Test behavior when a tool is missing
# (Rename a required tool temporarily)
sudo mv /usr/bin/cmake /usr/bin/cmake.bak 2>/dev/null
./build_packages_local.sh 2>&1 | grep -i "missing\|cmake" && echo "✓ Dependency check works"
sudo mv /usr/bin/cmake.bak /usr/bin/cmake 2>/dev/null
```

### Test 14: Disk Space Check
```bash
# Ensure adequate disk space (need ~10GB for build)
df -h . | awk 'NR==2 {print $4}' | numfmt --from=auto --to=unit=G | awk '{if($1 < 10) print "✗ Less than 10GB available"; else print "✓ Sufficient disk space"}'
```

## ✅ Cross-Platform Verification

### Test 15: Multi-Distribution Test Matrix
Test on different distributions if possible:
- [ ] Ubuntu 20.04
- [ ] Ubuntu 22.04
- [ ] Ubuntu 24.04
- [ ] Rocky Linux 8
- [ ] CentOS Stream 9
- [ ] RHEL 8
- [ ] Debian 11
- [ ] Debian 12

### Test 16: GPU Family Variants
Test with different GPU families:
- [ ] gfx94X-dcgpu (MI300)
- [ ] gfx950-dcgpu (MI350)
- [ ] gfx110X-all (RX 7900 series)
- [ ] gfx1151 (Strix Halo)
- [ ] gfx120X-all (RX 9000 series)

## ✅ Performance & Optimization

### Test 17: Build Time Measurement
```bash
# Measure build time
time ./build_packages_local.sh
# Record and compare:
# Expected: 15-30 minutes depending on hardware
```

### Test 18: Package Size Check
```bash
# Check package sizes are reasonable
ls -lh ./build/rocm-validation-suite*.{deb,rpm,tar.gz} 2>/dev/null
# Expected sizes:
# DEB: ~5-50 MB
# RPM: ~5-50 MB
# TGZ: ~5-50 MB
```

## ✅ Security & Best Practices

### Test 19: Security Scan
```bash
# Check for hardcoded secrets or sensitive data
grep -r "password\|token\|secret\|key" .github/workflows/*.yml build_packages_local.sh
# Should only find variable names, not actual secrets

# Check file permissions
find .github -type f -executable && echo "✗ Executable files in .github" || echo "✓ No unnecessary executable files"
```

### Test 20: Code Quality
```bash
# Check for shell script issues (if shellcheck is installed)
if command -v shellcheck >/dev/null; then
    shellcheck build_packages_local.sh && echo "✓ Shell script passes shellcheck"
fi

# Check for common mistakes
grep -n "rm -rf /" build_packages_local.sh && echo "✗ Dangerous rm command found" || echo "✓ No dangerous commands"
```

## ✅ Final Verification

### Test 21: End-to-End Test
Complete end-to-end test:
1. [ ] Fresh git clone of repository
2. [ ] Run local build script successfully
3. [ ] Verify all three package types created
4. [ ] Install one package and test RVS
5. [ ] Push to GitHub and verify Actions run
6. [ ] Download artifacts and verify contents
7. [ ] Install artifact package and test RVS

### Test 22: Documentation Test
Have someone unfamiliar with the system:
1. [ ] Follow QUICKSTART_PACKAGES.md instructions
2. [ ] Successfully build packages locally
3. [ ] Successfully trigger GitHub Actions workflow
4. [ ] Successfully install and test a package

## 📊 Test Results Summary

After completing all tests, fill in the results:

```
Total Tests: 22
Passed: ___ / 22
Failed: ___ / 22
Skipped: ___ / 22

Critical Failures: ___
Minor Issues: ___
Warnings: ___
```

## 🚀 Deployment Readiness

Mark as complete when all critical tests pass:
- [ ] All syntax validation passes
- [ ] Local build succeeds
- [ ] GitHub Actions workflow runs successfully
- [ ] At least one package type installs and runs
- [ ] Documentation is accurate and complete
- [ ] No critical security issues found

## 📝 Notes & Issues

Record any issues found during testing:

```
Issue 1:
Description:
Severity: [Critical/Major/Minor]
Status: [Open/Fixed]

Issue 2:
Description:
Severity: [Critical/Major/Minor]
Status: [Open/Fixed]
```

## ✅ Sign-off

When all critical tests pass and the system is ready for production:

```
Tested by: ________________
Date: ____________________
Status: [ ] Ready for Production / [ ] Needs Work
Signature: ________________
```

---

**Last Updated:** January 29, 2026  
**Version:** 1.0  
**Next Review:** After first production deployment
