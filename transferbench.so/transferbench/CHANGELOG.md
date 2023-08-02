# Changelog for TransferBench

## v1.22
### Modified
- Switching kernel timing function to wall_clock64

## v1.21
### Fixed
- Fixed bug with SAMPLING_FACTOR

## v1.20
### Fixed
- VALIDATE_DIRECT can now be used with USE_PREP_KERNEL
- Switch to local GPU for validating GPU memory

## v1.19
### Added
- VALIDATE_DIRECT now also applies to source memory array checking
- Adding null memory pointer check prior to deallocation

## v1.18
### Added
- Adding ability to validate GPU destination memory directly without going through CPU staging buffer (VALIDATE_DIRECT)
  - NOTE: This will only work on AMD devices with large-bar access enable and may slow things down considerably
### Changed
- Refactored how environment variables are displayed
- Mismatch stops after first detected error within an array instead of list all mismatched elements

## v1.17
### Added
- Allow switch to GFX kernel for source array initialization (USE_PREP_KERNEL)
  - USE_PREP_KERNEL cannot be used with FILL_PATTERN
- Adding ability to compile with nvcc only (TransferBenchCuda)
### Changed
- Default pattern set to [Element i = ((i * 517) modulo 383 + 31) * (srcBufferIdx + 1)]
### Fixed
- Re-adding example.cfg file

## v1.16
### Added
- Additional src array validation during preparation
- Adding new env var CONTINUE_ON_ERROR to resume tests after mis-match detection
- Initializing GPU memory to 0 during allocation

## v1.15
### Fixed
- Fixed a bug that prevented single Transfers > 8GB
### Changed
- Removed "check for latest ROCm" warning when allocating too much memory
- Printing off source memory value as well when mis-match is detected

## v1.14
### Added
- Added documentation
- Added pthread linking in src/Makefile and CMakeLists.txt
- Added printing off the hex value of the floats for output and reference

## v1.13
### Added
- Added support for cmake

### Changed
- Converted to the Pitchfork layout standard

## v1.12
### Added
- Added support for TransferBench on NVIDIA platforms (via HIP_PLATFORM=nvidia)
  - CPU executors on NVIDIA platform cannot access GPU memory (no large-bar access)

## v1.11
### Added
- New multi-input / multi-output support (MIMO).  Transfers now can reduce (element-wise summation) multiple input memory arrays
  and write the sums to multiple outputs
- New GPU-DMA executor 'D' (uses hipMemcpy for SDMA copies).  Previously this was done using USE_HIP_CALL, but now this allows
  GPU-GFX kernel to run in parallel with GPU-DMA instead of applying to all GPU executors globally.
  - GPU-DMA executor can only be used for single-input/single-output Transfers
  - GPU-DMA executor can only be associated with one SubExecutor
- Added new "Null" memory type 'N', which represents empty memory. This allows for read-only or write-only Transfers
- Added new GPU_KERNEL environment variable that allows for switching between various GPU-GFX reduction kernels

### Optimized
- Slightly improved GPU-GFX kernel performance based on hardware architecture when running with fewer CUs

### Changed
- Updated the example.cfg file to cover the new features
- Updated output to support MIMO
- Changed CUs/CPUs threads naming to SubExecutors for consistency
- Sweep Preset:
  - Default sweep preset executors now includes DMA
- P2P Benchmarks:
  - Now only works via "p2p".  Removed "p2p_rr", "g2g" and "g2g_rr".
    - Setting NUM_CPU_DEVICES=0 can be used to only benchmark GPU devices (like "g2g")
    - New environment variable USE_REMOTE_READ replaces "_rr" presets
  - New environment variable USE_GPU_DMA=1 replaces USE_HIP_CALL=1 for benchmarking with GPU-DMA Executor
  - Number of GPU SubExecutors for benchmark can be specified via NUM_GPU_SE
    - Defaults to all CUs for GPU-GFX, 1 for GPU-DMA
  - Number of CPU SubExecutors for benchmark can be specified via NUM_CPU_SE
- Psuedo-random input pattern has been slightly adjusted to have different patterns for each input array within same Transfer

### Removed
- USE_HIP_CALL has been removed.  Use GPU-DMA executor 'D' or set USE_GPU_DMA=1 for P2P benchmark presets
  - Currently warning will be issued if USE_HIP_CALL is set to 1 and program will terminate
- Removed NUM_CPU_PER_TRANSFER - The number of CPU SubExecutors will be whatever is specified for the Transfer
- Removed USE_MEMSET environment variable.  This can now be done via a Transfer using the null memory type

## v1.10
### Fixed
- Fix incorrect bandwidth calculation when using single stream mode and per-Transfer data sizes

## v1.09
### Added
- Printing off src/dst memory addresses during interactive mode
### Changed
- Switching to numa_set_preferred instead of set_mempolicy

## v1.08
### Changed
- Fixing handling of non-configured NUMA nodes
- Topology detection now shows actual NUMA node indices
- Fix for issue with NUM_GPU_DEVICES

## v1.07
### Changed
- Fix bug with allocations involving non-default CPU memory types

## v1.06
### Added
- Added unpinned CPU memory type ('U').  May require HSA_XNACK=1 in order to access via GPU executors
- Adding logging of sweep configuration to lastSweep.cfg
- Adding ability to specify number of CUs to use for sweep-based presets
### Changed
- Fixing random sweep repeatibility
- Fixing bug with CPU NUMA node memory allocation
- Modified advanced configuration file format to accept bytes per Transfer

## v1.05
### Added
- Topology output now includes NUMA node information
- Support for NUMA nodes with no CPU cores (e.g. CXL memory)
### Removed
- SWEEP_SRC_IS_EXE environment variable

## v1.04
### Added
- New environment variables for sweep based presets
  - SWEEP_XGMI_MIN   - Min number of XGMI hops for Transfers
  - SWEEP_XGMI_MAX   - Max number of XGMI hops for Transfers
  - SWEEP_SEED       - Random seed being used
  - SWEEP_RAND_BYTES - Use random amount of bytes (up to pre-specified N) for each Transfer
### Changed
  - CSV output for sweep includes env vars section followed by output
  - CSV output no longer lists env var parameters in columns
  - Default number of warmup iterations changed from 3 to 1
  - Splitting CSV output of link type to ExeToSrcLinkType and ExeToDstLinkType

## v1.03
### Added
- New preset modes stress-test benchmarks "sweep" and "randomsweep"
  - sweep iterates over all possible sets of Transfers to test
  - randomsweep iterates over random sets of Transfers
  -  New sweep-only environment variables can modify sweep
     - SWEEP_SRC - String containing only "B","C","F", or "G", defining possible source memory types
     - SWEEP_EXE - String containing only "C", or "G", defining possible executors
     - SWEEP_DST - String containing only "B","C","F", or "G", defining possible destination memory types
     - SWEEP_SRC_IS_EXE - Restrict executor to be the same as the source if non-zero
     - SWEEP_MIN - Minimum number of parallel transfers to test
     - SWEEP_MAX - Maximum number of parallel transfers to test
     - SWEEP_COUNT - Maximum number of tests to run
     - SWEEP_TIME_LIMIT - Maximum number of seconds to run tests for
- New environment variable to restrict number of available GPUs to test on (primarily for sweep runs)
  - NUM_CPU_DEVICES - Number of CPU devices
  - NUM_GPU_DEVICES - Number of GPU devices
### Changed
- Fixed timing display for CPU-executors when using single stream mode

## v1.02
### Added
- Setting NUM_ITERATIONS to negative number indicates to run for -NUM_ITERATIONS seconds per Test
### Changed
- Copies are now refered to as Transfers instead of Links
- Re-ordering how env vars are displayed (alphabetically now)
### Removed
- Combined timing is now always on for kernel-based GPU copies. COMBINED_TIMING env var has been removed
- Use single sync is no longer supported to facility variable iterations. USE_SINGLE_SYNC env var has been removed

## v1.01
### Added
- Adding USE_SINGLE_STREAM feature
  - All Links that execute on the same GPU device are executed with a single kernel launch on a single stream
  - Does not work with USE_HIP_CALL and forces USE_SINGLE_SYNC to collect timings
  - Adding ability to request coherent / fine-grained host memory ('B')
### Changed
- Separating TransferBench from RCCL repo
- Peer-to-peer benchmark mode now works OUTPUT_TO_CSV
- Toplogy display now works with OUTPUT_TO_CSV
- Moving documentation about config file into example.cfg
### Removed
- Removed config file generation
- Removed show pointer address environment variable (SHOW_ADDR)
