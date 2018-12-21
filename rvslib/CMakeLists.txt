################################################################################
##
## Copyright (c) 2018 2018 ROCm Developer Tools
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
if ( ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  message(FATAL "In-source build is not allowed")
endif ()

set ( RVS "rvslib" )
set ( RVS_PACKAGE "rvs-roct" )
set ( RVS_COMPONENT "lib${RVS}" )
set ( RVS_TARGET "${RVS}" )

project ( ${RVS_TARGET} )

message(STATUS "MODULE: ${RVS}")

add_compile_options(-std=c++11)
add_compile_options(-pthread)
add_compile_options(-Wl,-no-as-needed)
add_compile_options(-Wall -Wextra)
add_compile_options(-fPIC)
add_compile_options(-DRVS_OS_TYPE_NUM=${RVS_OS_TYPE_NUM})
if (RVS_COVERAGE)
  add_compile_options(-o0 -fprofile-arcs -ftest-coverage)
  set(CMAKE_EXE_LINKER_FLAGS "--coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "--coverage")
endif()

## Set default module path if not already set
if ( NOT DEFINED CMAKE_MODULE_PATH )
    set ( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules/" )
endif ()

## Include common cmake modules
include ( utils )
#include (CTest)

## Setup the package version default (the actual one is taken from the latest Git tag).
get_version ( "0.0.0" )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( BUILD_VERSION_MINOR ${VERSION_MINOR} )
set ( BUILD_VERSION_PATCH ${VERSION_PATCH} )
set ( LIB_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

if ( DEFINED VERSION_BUILD AND NOT ${VERSION_BUILD} STREQUAL "" )
    set ( BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}" )
endif ()
set ( BUILD_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

## make version numbers visible to C code
add_compile_options(-DBUILD_VERSION_MAJOR=${VERSION_MAJOR})
add_compile_options(-DBUILD_VERSION_MINOR=${VERSION_MINOR})
add_compile_options(-DBUILD_VERSION_PATCH=${VERSION_PATCH})
add_compile_options(-DLIB_VERSION_STRING="${LIB_VERSION_STRING}")
add_compile_options(-DBUILD_VERSION_STRING="${BUILD_VERSION_STRING}")

## define include directories
include_directories(./ ../
  ${ROCM_SMI_INC_DIR} ${ROCR_INC_DIR} ${ROCBLAS_INC_DIR} ${HIP_INC_DIR}
)


## define common source files
set(SOURCES
  ../src/gpu_util.cpp
  ../src/rvs_util.cpp
  ../src/rsmi_util.cpp

  ../src/rvsactionbase.cpp
  ../src/rvsthreadbase.cpp

  ../src/rvsliblogger.cpp
  ../src/rvslognodebase.cpp
  ../src/rvslognoderec.cpp
  ../src/rvslognode.cpp
  ../src/rvslognodestring.cpp
  ../src/rvslognodeint.cpp

  ../src/rvs_blas.cpp
  ../src/rvshsa.cpp
  )

## define run-time specific source files
set(SOURCES_RT
  ../src/rvsloglp.cpp
  ../src/pci_caps.cpp
  )

## define unit testing specific source files (mocking)
set(SOURCES_UT
  ../src/rvsloglp_utest.cpp
  ../src/pci_caps.cpp
  ../src/rvs_unit_testing_defs.cpp
   )

## define rvslibrt (run-time) library
add_library(${RVS_TARGET}rt  ${SOURCES_RT})

## define rvslib library
add_library(${RVS_TARGET}  ${SOURCES})
add_dependencies(${RVS_TARGET} ${RVS_TARGET}rt)

if (RVS_BUILD_TESTS)
  ## define rvslibut (unit test) library
  add_library(${RVS_TARGET}ut  ${SOURCES_UT})
  target_compile_definitions(${RVS_TARGET}ut PRIVATE RVS_UNIT_TEST)
  add_dependencies(${RVS_TARGET}ut ${RVS_TARGET}rt)
endif()