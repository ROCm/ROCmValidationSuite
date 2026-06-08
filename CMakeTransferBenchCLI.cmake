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
# rocm-cmake — which is not on the module path in RVS's CentOS/manylinux CI
# container — and matches the relocatable RPATH layout RVS itself uses.
#
# Inputs:
#   TRANSFERBENCH_SOURCE_DIR  - path to the TransferBench submodule
#   ROCM_PATH                 - ROCm install prefix (forwarded to the sub-build)
#   ROCM_MAJOR_VERSION        - forwarded so TB's relocatable RPATH lines up
#   CPACK_PACKAGING_INSTALL_PREFIX - forwarded so TB lives under the same
#                              prefix as rvs (e.g. /opt/rocm/extras-7)
#
################################################################################

include(ExternalProject)

if(NOT TRANSFERBENCH_SOURCE_DIR)
  message(FATAL_ERROR "TRANSFERBENCH_SOURCE_DIR not set")
endif()

set(TRANSFERBENCH_BUILD_DIR   "${CMAKE_BINARY_DIR}/TransferBench-cli-build")
set(TRANSFERBENCH_INSTALL_DIR "${CMAKE_BINARY_DIR}/TransferBench-cli-install")

# Forward only the knobs the child build cares about. We deliberately do NOT
# forward CMAKE_CXX_COMPILER — TransferBench picks amdclang++ itself.
set(_tb_cmake_args
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=${TRANSFERBENCH_INSTALL_DIR}
  -DROCM_PATH=${ROCM_PATH}
  -DBUILD_RELOCATABLE_PACKAGE=ON
  -DBUILD_PACKAGES=OFF
)
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
)

# Install the TransferBench CLI binary into the rvs package layout (same
# component as rvs itself, so it lands in the same DEB/RPM).
install(PROGRAMS "${TRANSFERBENCH_INSTALL_DIR}/bin/TransferBench"
        DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        COMPONENT rvsmodule)
