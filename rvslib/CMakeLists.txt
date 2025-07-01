################################################################################
##
## Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set ( RVS "rvslib" )
set ( RVS_PACKAGE "rvs-roct" )
set ( RVS_COMPONENT "lib${RVS}" )
set ( RVS_TARGET "${RVS}" )

project ( ${RVS_TARGET} )

message(STATUS "MODULE: ${RVS}")

add_compile_options(-pthread)
add_compile_options(-Wall )
add_compile_options(-fPIC)
add_compile_options(-DRVS_OS_TYPE_NUM=${RVS_OS_TYPE_NUM})

find_package(OpenMP)

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
add_compile_options(-DRVS_VERSION_STRING="${RVS_VERSION}")

#ROCBLAS VERSION CHECK FLAGS TO CHECK REORG VERSION 2.44.0
add_compile_options(-DRVS_ROCBLAS_VERSION_FLAT=${RVS_ROCBLAS_VERSION_FLAT})

# Determine HSA_PATH
if(NOT DEFINED HIPCC_PATH)
  if(NOT DEFINED ENV{HIPCC_PATH})
    set(HIPCC_PATH "${ROCM_PATH}" CACHE PATH "Path to which hipcc runtime has been installed")
     else()
       set(HIPCC_PATH $ENV{HIPCC_PATH} CACHE PATH "Path to which hipcc runtime has been installed")
     endif()
endif()

# Add HIP_VERSION to CMAKE_<LANG>_FLAGS
set(HIP_HCC_BUILD_FLAGS "${HIP_HCC_BUILD_FLAGS} -DHIP_VERSION_MAJOR=${HIP_VERSION_MAJOR} -DHIP_VERSION_MINOR=${HIP_VERSION_MINOR} -DHIP_VERSION_PATCH=${HIP_VERSION_GITDATE}")

set(HIP_HCC_BUILD_FLAGS)
set(HIP_HCC_BUILD_FLAGS "${HIP_HCC_BUILD_FLAGS} -fPIC ${HCC_CXX_FLAGS} -I${HSA_INC_DIR} ${ASAN_CXX_FLAGS}")

#Set compiler and compiler flags
set(CMAKE_CXX_COMPILER "${HIPCC_PATH}/bin/hipcc")
set(CMAKE_C_COMPILER   "${HIPCC_PATH}/bin/hipcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_BUILD_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${HIP_HCC_BUILD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_LD_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_LD_FLAGS}")

if(BUILD_ADDRESS_SANITIZER)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
            OUTPUT_VARIABLE ASAN_LIB_FULL_PATH)
  get_filename_component(ASAN_LIB_PATH ${ASAN_LIB_FULL_PATH} DIRECTORY)
else()
  set(ASAN_LIB_PATH "$ENV{LD_LIBRARY_PATH}")
endif()

## define include directories
include_directories(./ ../ ../rvs
  ${AMD_SMI_INC_DIR} ${ROCR_INC_DIR} ${ROCBLAS_INC_DIR} ${HIP_INC_DIR} ${HIPBLASLT_INC_DIR} ${YAML_CPP_INCLUDE_DIR} ${HIPRAND_INC_DIR} ${ROCRAND_INC_DIR} ${HIPBLAS-COMMON_INCLUDE_DIR} ${MXDATAGENERATOR_INC_DIR})

link_directories(${ASAN_LIB_PATH} ${AMD_SMI_LIB_DIR})

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
  ../src/rvsminnode.cpp
  ../src/rvslognodelist.cpp
  ../src/rvs_blas.cpp
  ../src/rvshsa.cpp

## Run-time specific source files
  ../src/rvsloglp.cpp
  ../src/pci_caps.cpp

## rvshelper source files
  ../rvs/src/rvsmodule.cpp
  ../rvs/src/rvsif_base.cpp
  ../rvs/src/rvsif0.cpp
  ../rvs/src/rvsif1.cpp
  ../rvs/src/rvsaction.cpp
  ../rvs/src/rvscli.cpp
  ../rvs/src/rvsexec.cpp
  ../rvs/src/rvsexec_do_yaml.cpp
  ../rvs/src/rvsoptions.cpp
  ../rvs/src/rvs_interface.cpp
)

## define unit testing specific source files (mocking)
set(SOURCES_UT
  ../src/rvsloglp_utest.cpp
  ../src/pci_caps.cpp
  ../src/rvs_unit_testing_defs.cpp
   )

## define rvslib library
add_library(${RVS_TARGET} SHARED ${SOURCES})

target_compile_options(${RVS_TARGET} PRIVATE -mf16c)
target_compile_definitions(${RVS_TARGET} PRIVATE ROCM_USE_FLOAT16)
set_target_properties(${RVS_TARGET}
 PROPERTIES SUFFIX .so.${LIB_VERSION_STRING}
 LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
target_link_libraries(${RVS_TARGET} ${AMD_SMI_LIB})
## Install shared library librvslib.so
add_custom_command(TARGET ${RVS_TARGET} POST_BUILD
COMMAND ln -fs ./lib${RVS}.so.${LIB_VERSION_STRING} lib${RVS}.so.${VERSION_MAJOR} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
COMMAND ln -fs ./lib${RVS}.so.${VERSION_MAJOR} lib${RVS}.so WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

install(TARGETS ${RVS_TARGET} LIBRARY DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} COMPONENT rvsmodule)
install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${RVS}.so.${VERSION_MAJOR}"
	DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} COMPONENT rvsmodule)
install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${RVS}.so"
	DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} COMPONENT rvsmodule)

if (RVS_BUILD_TESTS)
  ## define rvslibut (unit test) library
  add_library(${RVS_TARGET}ut SHARED ${SOURCES_UT})
  target_compile_definitions(${RVS_TARGET}ut PRIVATE RVS_UNIT_TEST)
  add_dependencies(${RVS_TARGET}ut ${RVS_TARGET})
endif()
