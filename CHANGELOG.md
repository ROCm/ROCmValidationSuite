# Change Log for ROCm Validation Suite (RVS)

Full documentation for RVS is available at [ROCmValidationSuite.Readme](https://github.com/ROCm-Developer-Tools/ROCmValidationSuite).

## (Unreleased) RVS for ROCm 5.7

### Added
- Introduced new RVS interface APIs enabling test execution from external components.
- Added new "device_index" property for conf. file.

### Changed
- Moved all static internal libraries to a single public shared library (rvslib).
- Use HIP stream callback mechanism for gemm operations completion (instead of polling).

### Removed
- yaml-cpp source download and build removed from RVS cmake build.
