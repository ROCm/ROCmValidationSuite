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

set(TRANSFERBENCH_BUILD_DIR "${CMAKE_BINARY_DIR}/TransferBench-cli-build")

# Narrow GPU target set by default. Empty string = take TB's own default.
set(TRANSFERBENCH_GPU_TARGETS
    "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
    CACHE STRING "GPU targets to build TransferBench for")

set(_tb_cmake_args
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DROCM_PATH=${ROCM_PATH}
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

# Build only -- never run TransferBench's own install rules (see header). The
# binary is consumed directly from the build tree by the install(PROGRAMS)
# below, so there is no separate child install directory to leak into packages.
ExternalProject_Add(TransferBenchCLI
  SOURCE_DIR        "${TRANSFERBENCH_SOURCE_DIR}"
  BINARY_DIR        "${TRANSFERBENCH_BUILD_DIR}"
  CMAKE_ARGS        ${_tb_cmake_args}
  BUILD_ALWAYS      OFF
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
  LOG_CONFIGURE     ON
  LOG_BUILD         ON
  LOG_OUTPUT_ON_FAILURE ON
)

# Install the TransferBench CLI binary into the rvs package layout (same
# component as rvs itself, so it lands in the same DEB/RPM). Pulled straight
# from the child build tree -- TransferBench emits it at the build-dir root via
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY .).
install(PROGRAMS "${TRANSFERBENCH_BUILD_DIR}/TransferBench"
        DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        COMPONENT rvsmodule)
