################################################################################
##
## Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
##
## MIT LICENSE
## (full license header — same as the rest of the repo)
##
################################################################################
#
# Build the TransferBench CLI binary from the external/TransferBench submodule
# as a sub-build of RVS, and install it alongside the rvs executable.
#
# TransferBench has its own CMake project that pins amdclang++ as the compiler
# before its project() call, so it cannot be add_subdirectory'd into RVS — the
# child compiler would conflict with the parent. ExternalProject_Add runs the
# child build in an isolated CMake invocation, so the two builds do not
# interfere.
#
# Inputs:
#   TRANSFERBENCH_SOURCE_DIR  - path to the TransferBench submodule
#   ROCM_PATH                 - ROCm install prefix (forwarded to the sub-build)
# Outputs:
#   TransferBench executable installed under ${CMAKE_INSTALL_BINDIR}
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
ExternalProject_Add(TransferBenchCLI
  SOURCE_DIR        "${TRANSFERBENCH_SOURCE_DIR}"
  BINARY_DIR        "${TRANSFERBENCH_BUILD_DIR}"
  INSTALL_DIR       "${TRANSFERBENCH_INSTALL_DIR}"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${TRANSFERBENCH_INSTALL_DIR}
    -DROCM_PATH=${ROCM_PATH}
    -DBUILD_PACKAGES=OFF
  BUILD_ALWAYS      OFF
  INSTALL_COMMAND   ${CMAKE_COMMAND} --install <BINARY_DIR> --component devel
  TEST_COMMAND      ""
)

# Install the TransferBench CLI binary into the rvs package layout.
install(PROGRAMS "${TRANSFERBENCH_INSTALL_DIR}/bin/TransferBench"
        DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        COMPONENT rvsmodule)
