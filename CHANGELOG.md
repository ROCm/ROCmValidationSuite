# Changelog for ROCm Validation Suite

Documentation for ROCm Validation Suite (RVS) is available at
[https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/).

## (Unreleased) RVS for ROCm 6.0

### Additions

* Support for Mariner OS
* Support for gfx941 & gfx942
* Navi31 and Navi32 specific configurations
* GST stress test support for MI300X

## RVS for ROCm 5.7

### Additions

* Introduced new RVS interface APIs that you can use to run tests from external components
* Added a new `device_index` property for the `conf.` file

### Changes

* Moved all static internal libraries to a single public shared library (`rvslib`)
* Use HIP stream callback mechanism (instead of polling) for gemm operations completion

### Optimizations

* In GST and IET modules, use of the callback mechanism (instead of polling) for the HIP stream
  reduced the percentage of CPU utilization

### Removals

* `yaml-cpp` source download and build has been removed from the RVS CMake build
