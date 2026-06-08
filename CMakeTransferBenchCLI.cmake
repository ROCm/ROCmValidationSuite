################################################################################
##
## Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
## MIT LICENSE
##
################################################################################
#
# Build the TransferBench CLI binary from the external/TransferBench submodule
# as a sub-build of RVS, and install it alongside the rvs executable.
#
# TransferBench pins amdclang++ as the compiler before its project() call, so
# add_subdirectory would conflict with the parent compiler. ExternalProject_Add
# runs the child build in an isolated CMake invocation.
#
# BUILD_RELOCATABLE_PACKAGE=ON selects TransferBench's non-rocm-cmake install
# branch (plain install(TARGETS ...)), avoiding a hard dependency on
# rocm-cmake-time install macros and matching the relocatable RPATH layout
# RVS itself uses.
#
# We forward CMAKE_CXX_FLAGS so the child build inherits things like
# --gcc-toolchain=... that RVS's build script sets to make amdclang++ pick up
# a modern libstdc++ (required for C++20 <barrier> in TransferBench).
#
# TRANSFERBENCH_GPU_TARGETS narrows the set of GPU archs TransferBench builds
# for. Defaults to the same set RVS itself ships on so the package isn't
# bloated and the CI build doesn't time out on archs RVS doesn't care about.
# Override with -DTRANSFERBENCH_GPU_TARGETS="gfx906;gfx90a;..." or set to
# empty to take TransferBench's default (13 archs).
#
################################################################################

include(ExternalProject)

if(NOT TRANSFERBENCH_SOURCE_DIR)
  message(FATAL_ERROR "TRANSFERBENCH_SOURCE_DIR not set")
endif()

set(TRANSFERBENCH_BUILD_DIR   "${CMAKE_BINARY_DIR}/TransferBench-cli-build")
set(TRANSFERBENCH_INSTALL_DIR "${CMAKE_BINARY_DIR}/TransferBench-cli-install")

# Narrow GPU target set by default. Empty string = take TB's own default.
set(TRANSFERBENCH_GPU_TARGETS
    "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
    CACHE STRING "GPU targets to build TransferBench for")

set(_tb_cmake_args
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=${TRANSFERBENCH_INSTALL_DIR}
  -DROCM_PATH=${ROCM_PATH}
  -DBUILD_RELOCATABLE_PACKAGE=ON
  -DBUILD_PACKAGES=OFF
)

# Forward parent CXX flags (carries --gcc-toolchain=... on RHEL/manylinux so
# amdclang++ picks up a modern libstdc++ with <barrier>).
if(CMAKE_CXX_FLAGS)
  list(APPEND _tb_cmake_args "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
endif()

if(TRANSFERBENCH_GPU_TARGETS)
  list(APPEND _tb_cmake_args "-DGPU_TARGETS=${TRANSFERBENCH_GPU_TARGETS}")
endif()

if(DEFINED ROCM_MAJOR_VERSION)
  list(APPEND _tb_cmake_args -DROCM_MAJOR_VERSION=${ROCM_MAJOR_VERSION})
endif()
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  list(APPEND _tb_cmake_args
       -DCPACK_PACKAGING_INSTALL_PREFIX=${CPACK_PACKAGING_INSTALL_PREFIX})
endif()

ExternalProject_Add(TransferBenchCLI
  SOURCE_DIR        "${TRANSFERBENCH_SOURCE_DIR}"
  BINARY_DIR        "${TRANSFERBENCH_BUILD_DIR}"
  INSTALL_DIR       "${TRANSFERBENCH_INSTALL_DIR}"
  CMAKE_ARGS        ${_tb_cmake_args}
  BUILD_ALWAYS      OFF
  INSTALL_COMMAND   ${CMAKE_COMMAND} --install <BINARY_DIR> --component devel
  TEST_COMMAND      ""
  LOG_CONFIGURE     ON
  LOG_BUILD         ON
  LOG_INSTALL       ON
  LOG_OUTPUT_ON_FAILURE ON
)

# Install the TransferBench CLI binary into the rvs package layout (same
# component as rvs itself, so it lands in the same DEB/RPM).
install(PROGRAMS "${TRANSFERBENCH_INSTALL_DIR}/bin/TransferBench"
        DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        COMPONENT rvsmodule)
