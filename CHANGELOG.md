# Changelog for ROCm Validation Suite (RVS)

Full documentation for RVS is available at [ROCmValidationSuite.Readme](https://github.com/ROCm/ROCmValidationSuite).

## RVS 1.1.0 for ROCm 6.4.2 RC2

### Added

- NPS2/DPX & NPS4/CPX partition modes support for MI300X.

## RVS 1.1.0 for ROCm 6.4.0

### Added

- Configuration files for MI210.
- Support for OCP fp8 datatype.
- Support for new platform MI308X-HF.
- GPU index based cli. execution.

### Changed

- json logging with updated schema.
- Updated babel conf. files for MI308X and MI308X-HF.

## RVS 1.1.0 for ROCm 6.3.0

### Added

- Support for hipBLASLT blas library and option to select blas library in conf. file.
- Added Babel, thermal and performance benchmark test for MI308X.
- Support for new platform MI325X.

### Changed

- Babel parameters made runtime configurable.

## RVS 1.0.0 for ROCm 6.2.0

### Added

- Gemm self-check and accuracy-check support for checking consistency & accuracy of gemm output.
- Trignometric float & random integer matrix data initialization support.
- IET (power) stress test for MI300A.
- IET (power transition) test for MI300X.

### Changed

- Updated GST performance benchmark test for better numbers.

## RVS 1.0.0 for ROCm 6.1.0

### Added

- Support data types (BF16 and FP8) based GEMM operations in GPU Stress Test (GST) module.
- Babel test for MI300X.

### Changed

- Updated pebb & pbqt logs to include PCI BDF.

## RVS 1.0.0 for ROCm 6.0.0

### Added

- Support for Mariner OS
- Support for gfx941 & gfx942
- Navi31 and Navi32 specific configurations
- GST stress test support for MI300X

## RVS 1.0.0 for ROCm 5.7.0

### Added

- Introduced new RVS interface APIs enabling test execution from external components.
- Added new `device_index` property for conf. file.

### Changed

- Moved all static internal libraries to a single public shared library (rvslib).
- Use HIP stream callback mechanism for gemm operations completion (instead of polling).

### Removed

- yaml-cpp source download and build removed from RVS cmake build.

### Optimized

- In GST and IET modules, use of callback mechanism instead of polling for HIP stream reduced the CPU utilization %.
