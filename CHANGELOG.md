# Change Log for ROCm Validation Suite (RVS)

Full documentation for RVS is available at [ROCmValidationSuite.Readme](https://github.com/ROCm/ROCmValidationSuite).

## RVS 1.0.0 for ROCm 6.2

### Changed

- Updated GST performance benchmark test for better numbers.

### Added

- Gemm self-check and accuracy-check support for checking consistency & accuracy of gemm output.
- Trignometric float & random integer matrix data initialization support.
- IET (power) stress test for MI308X & MI300A.
- IET (power transition) test for MI300X.

## (Unreleased) RVS for ROCm 6.1

### Changed
- Updated pebb & pbqt logs to include PCI BDF.

### Added
- Support data types (BF16 and FP8) based GEMM operations in GPU Stress Test (GST) module.
- Babel test for MI300X.

## RVS for ROCm 6.0

### Added
- Support for Mariner OS
- Support for gfx941 & gfx942
- Navi31 and Navi32 specific configurations
- GST stress test support for MI300X

## RVS for ROCm 5.7

### Added
- Introduced new RVS interface APIs enabling test execution from external components.
- Added new "device_index" property for conf. file.

### Changed
- Moved all static internal libraries to a single public shared library (rvslib).
- Use HIP stream callback mechanism for gemm operations completion (instead of polling).

### Optimizations
- In GST and IET modules, use of callback mechanism instead of polling for HIP stream reduced the CPU utilization %.

### Removed
- yaml-cpp source download and build removed from RVS cmake build.
