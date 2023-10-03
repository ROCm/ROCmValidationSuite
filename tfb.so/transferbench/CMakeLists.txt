# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
if (DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCm install directory")
else()
    set(ROCM_PATH "/opt/rocm" CACHE STRING "ROCm install directory")
endif()
cmake_minimum_required(VERSION 3.5)

project(TransferBench VERSION 1.0 LANGUAGES CXX)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -L${ROCM_PATH}/lib")
include_directories(${ROCM_PATH}/include)
link_libraries(numa hsa-runtime64 pthread)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ..)
add_executable(TransferBench src/TransferBench.cpp)
target_include_directories(TransferBench PRIVATE src/include)

find_package(ROCM 0.8 REQUIRED PATHS ${ROCM_PATH})
include(ROCMInstallTargets)
include(ROCMCreatePackage)
set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)

set(PACKAGE_NAME TB)
set(LIBRARY_NAME TransferBench)

rocm_install(TARGETS TransferBench)

rocm_create_package(
    NAME ${LIBRARY_NAME}
    DESCRIPTION "TransferBench package"
    MAINTAINER "RCCL Team <gilbert.lee@amd.com>"
)
