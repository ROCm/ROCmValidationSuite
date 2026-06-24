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
# The child build is BUILD-ONLY: we never invoke TransferBench's own install
# rules. TransferBench installs unconditionally via rocm_install() into an
# absolute prefix; letting it install into a build-tree directory caused that
# directory to leak verbatim into the packaged TGZ (e.g.
# __w/.../build/TransferBench-cli-install/bin/TransferBench). Instead we set
# INSTALL_COMMAND to a no-op and pick the binary straight out of the child
# build tree, then install it ourselves into the rvs package layout.
#
# TransferBench writes its CLI binary to the build-tree root via
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY .), so it lands at
# ${TRANSFERBENCH_BUILD_DIR}/TransferBench.
#
# We forward CMAKE_CXX_FLAGS so the child build inherits things like
# --gcc-toolchain=... that RVS's build script sets to make amdclang++ pick up
# a modern libstdc++ (required for C++20 <barrier> in TransferBench).
#
# TRANSFERBENCH_GPU_TARGETS defaults to TransferBench build_packages_local.sh
# DEFAULT_GPU_TARGETS. Override with -DTRANSFERBENCH_GPU_TARGETS="..." or set
# GPU_TARGETS / TRANSFERBENCH_GPU_TARGETS in build_packages_local.sh. Set to
# empty to take TransferBench's own CMake default. GPU_TARGETS is written into
# the -C initial cache file (not -D on CMAKE_ARGS) because CMake list(APPEND)
# splits semicolon-separated strings.
#
# Feature flags match upstream TransferBench build_packages_local.sh (NIC/MPI/
# DMA-BUF off, multi-arch not local-GPU-only). Upstream passes -DDISABLE_DMABUF
# but TransferBench CMake uses ENABLE_DMA_BUF instead (that upstream flag is a
# no-op). RPATH uses CMakeTransferBenchRPATH.cmake.in, not BUILD_RELOCATABLE_PACKAGE.
#
################################################################################

include(ExternalProject)

if(NOT TRANSFERBENCH_SOURCE_DIR)
  message(FATAL_ERROR "TRANSFERBENCH_SOURCE_DIR not set")
endif()

set(TRANSFERBENCH_BUILD_DIR "${CMAKE_BINARY_DIR}/TransferBench-cli-build")

# Source of truth: TransferBench build_packages_local.sh DEFAULT_GPU_TARGETS
set(TRANSFERBENCH_GPU_TARGETS
    "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"
    CACHE STRING "GPU targets to build TransferBench for")

set(_tb_cmake_args
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DROCM_PATH=${ROCM_PATH}
  -DHIP_PLATFORM=amd
  -DCMAKE_VERBOSE_MAKEFILE=ON
  -DBUILD_LOCAL_GPU_TARGET_ONLY=OFF
  -DENABLE_NIC_EXEC=OFF
  -DENABLE_MPI_COMM=OFF
  -DENABLE_DMA_BUF=OFF
)

# Forward parent CXX flags (carries --gcc-toolchain=... on RHEL/manylinux so
# amdclang++ picks up a modern libstdc++ with <barrier>).
if(CMAKE_CXX_FLAGS)
  list(APPEND _tb_cmake_args "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
endif()

if(DEFINED ROCM_MAJOR_VERSION)
  list(APPEND _tb_cmake_args -DROCM_MAJOR_VERSION=${ROCM_MAJOR_VERSION})
endif()

# Relocatable RUNPATH for the CLI (copied from the child build tree via install(PROGRAMS)).
# Parent ${CMAKE_INSTALL_RPATH} cannot be forwarded safely ($ORIGIN expands empty).
# TransferBench ignores CMAKE_INSTALL_RPATH; use -C cache with -Wl,-rpath only
# (see CMakeTransferBenchRPATH.cmake.in). GPU_TARGETS is also set in this file.
set(_tb_rpath_cache "${CMAKE_BINARY_DIR}/TransferBenchRPATH.cmake")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/CMakeTransferBenchRPATH.cmake.in"
  "${_tb_rpath_cache}"
  @ONLY
)
# On GitHub Actions skip build-tree RPATH additions (linker -rpath above is enough).
if("$ENV{GITHUB_ACTIONS}" STREQUAL "true")
  file(APPEND "${_tb_rpath_cache}"
    "set(CMAKE_SKIP_BUILD_RPATH TRUE CACHE BOOL \"\" FORCE)\n")
endif()
if(TRANSFERBENCH_GPU_TARGETS)
  file(APPEND "${_tb_rpath_cache}"
    "set(GPU_TARGETS [[${TRANSFERBENCH_GPU_TARGETS}]] CACHE STRING \"\" FORCE)\n")
endif()
list(APPEND _tb_cmake_args "-C${_tb_rpath_cache}")

# Verbose child build: --verbose on cmake --build; stream to workflow log in CI.
set(_tb_build_command
  ${CMAKE_COMMAND} --build ${TRANSFERBENCH_BUILD_DIR} --verbose)
if(CMAKE_BUILD_PARALLEL_LEVEL)
  list(APPEND _tb_build_command --parallel ${CMAKE_BUILD_PARALLEL_LEVEL})
endif()

if("$ENV{GITHUB_ACTIONS}" STREQUAL "true")
  set(_tb_log_configure OFF)
  set(_tb_log_build OFF)
else()
  set(_tb_log_configure ON)
  set(_tb_log_build ON)
endif()

message(STATUS "TransferBench CLI sub-build:")
message(STATUS "  SOURCE_DIR=${TRANSFERBENCH_SOURCE_DIR}")
message(STATUS "  BINARY_DIR=${TRANSFERBENCH_BUILD_DIR}")
message(STATUS "  ROCM_PATH=${ROCM_PATH}")
message(STATUS "  HIP_PLATFORM=amd")
message(STATUS "  GPU_TARGETS=${TRANSFERBENCH_GPU_TARGETS}")
if(TRANSFERBENCH_GPU_TARGETS)
  message(STATUS "  GPU_TARGETS via: -C${_tb_rpath_cache}")
endif()
message(STATUS "  CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "  BUILD_COMMAND: ${CMAKE_COMMAND} --build ${TRANSFERBENCH_BUILD_DIR} --verbose")
message(STATUS "  LOG_CONFIGURE=${_tb_log_configure} LOG_BUILD=${_tb_log_build}")
foreach(_tb_arg IN LISTS _tb_cmake_args)
  message(STATUS "  CMAKE_ARGS: ${_tb_arg}")
endforeach()

# Build only -- never run TransferBench's own install rules (see header). The
# binary is consumed directly from the build tree by the install(PROGRAMS)
# below, so there is no separate child install directory to leak into packages.
ExternalProject_Add(TransferBenchCLI
  SOURCE_DIR        "${TRANSFERBENCH_SOURCE_DIR}"
  BINARY_DIR        "${TRANSFERBENCH_BUILD_DIR}"
  CMAKE_ARGS        ${_tb_cmake_args}
  BUILD_COMMAND     ${_tb_build_command}
  BUILD_ALWAYS      OFF
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
  LOG_CONFIGURE     ${_tb_log_configure}
  LOG_BUILD         ${_tb_log_build}
  LOG_OUTPUT_ON_FAILURE ON
)

# Install the TransferBench CLI binary into the rvs package layout (same
# component as rvs itself, so it lands in the same DEB/RPM). Pulled straight
# from the child build tree -- TransferBench emits it at the build-dir root via
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY .).
install(PROGRAMS "${TRANSFERBENCH_BUILD_DIR}/TransferBench"
        DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        COMPONENT rvsmodule)
