################################################################################
##
## Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
##
## MIT LICENSE:
## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
################################################################################

cmake_minimum_required ( VERSION 3.5.0 )

project ("rocm-validation-suite" )

message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

## Support for in-source compilaton
if ( ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin" )
else()
    set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )
endif ()

## Set default module path if not already set
if ( NOT DEFINED CMAKE_MODULE_PATH )
    set ( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" )
endif ()

## Set default module path if not already set
if ( NOT DEFINED CPACK_GENERATOR )
    set ( CPACK_GENERATOR "DEB;RPM" )
endif ()

if ( NOT DEFINED CMAKE_PACKAGING_INSTALL_PREFIX )
    set (CMAKE_PACKAGING_INSTALL_PREFIX "/opt/rocm" )
endif ()

## Include common cmake modules
include ( utils )

## Setup the package version default (the actual one is taken from the latest Git tag).
get_version ( "0.0.0" )

# Package Generator  #######################################################
# set(CPACK_GENERATOR "DEB")
# set(CMAKE_PACKAGING_INSTALL_PREFIX "/opt/rocm")
set(CPACK_PACKAGE_NAME "rocm-validation-suite")
set(CPACK_PACKAGE_DESCRIPTION "ROCm Validation Suite")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The ROCm Validation Suite – The ROCm Validation Suite is a system administrator and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPUs running in high performance computing environments, enabled using the ROCm software stack on a compatible platform.")
set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_PACKAGE_VENDOR "AMD")
set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL applications rvsmodule)

# Set PATH variables
set(ROCR_INC_DIR "/opt/rocm/include/" CACHE PATH "Contains header files exported by ROC Runtime")
set(ROCR_LIB_DIR "/opt/rocm/lib/" CACHE PATH "Contains library files exported by ROC Runtime")
set(ROCT_INC_DIR "/opt/rocm/include/libhsakmt" CACHE PATH "Contains header files exported by ROC Trunk")
set(ROCT_LIB_DIR "/opt/rocm/lib/" CACHE PATH "Contains library files exported by ROC Trunk")
set(HIP_INC_DIR "/opt/rocm/hip/include/hip/" CACHE PATH "Contains header files exported by ROC Runtime")
set(ROCBLAS_INC_DIR "$ENV{RVS}/../build/rocBLAS/build/release/rocblas-install/include/" CACHE PATH "Contains header files exported by rocBLAS")
set(ROCBLAS_LIB_DIR "$ENV{RVS}/../build/rocBLAS/build/release/rocblas-install/lib/" CACHE PATH "Contains library files exported by rocBLAS")
set(ROCM_SMI_INC_DIR "$ENV{RVS}/rocm_smi_lib/include" CACHE PATH "Contains header files exported by rocm_smi")
set(ROCM_SMI_LIB_DIR "$ENV{RVS}/../build/rocm_smi_lib" CACHE PATH "Contains library files exported by rocm_smi")
#add_subdirectory(yaml-cpp)

add_subdirectory(rvs)
add_subdirectory(gpup.so)
add_subdirectory(pesm.so)
add_subdirectory(rcqt.so)
add_subdirectory(peqt.so)
add_subdirectory(smqt.so)
add_subdirectory(pebb.so)
add_subdirectory(gm.so)
add_subdirectory(gst.so)
add_subdirectory(pqt.so)
add_subdirectory(iet.so)

#add_dependencies(rvs yaml-cpp)

add_custom_target(doc ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} doxygen
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS Programmer's Reference Guide"
    VERBATIM)

add_custom_target(man1 ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} doxygen DoxyMan1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS man.1 files"
    VERBATIM)

add_custom_target(userguide ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} doxygen DoxyGuide
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS User Guide"
    VERBATIM)

include (CPack)
