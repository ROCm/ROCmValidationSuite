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
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set ( RVS "rvs" )
set ( RVS_PACKAGE "rvs-roct" )
set ( RVS_COMPONENT "lib${RVS}" )
set ( RVS_TARGET "${RVS}" )

project ( ${RVS_TARGET} )

message(STATUS "MODULE: ${RVS}")

add_compile_options(-std=c++11)
add_compile_options(-pthread)
add_compile_options(-Wall -Wextra)
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
include_directories(./ ../ ${YAML_INC_DIR})
## define lib directories
link_directories(${CMAKE_CURRENT_BINARY_DIR} ${RVS_LIB_DIR})
## additional libraries
set (PROJECT_LINK_LIBS libdl.so "${YAML_LIB_DIR}/libyaml-cpp.a" libpthread.so)

## define source files
set(SOURCES
  src/rvsmodule.cpp
  src/rvsif_base.cpp
  src/rvsif0.cpp
  src/rvsif1.cpp
  src/rvsaction.cpp
  src/rvscli.cpp
  src/rvsexec.cpp
  src/rvsexec_do_yaml.cpp
  src/rvsoptions.cpp
)

## define helper lib
add_library(rvshelper ${SOURCES})

## define target
add_executable(${RVS_TARGET} src/rvs.cpp)
target_link_libraries(${RVS_TARGET} rvshelper rvslib ${PROJECT_LINK_LIBS} )
add_dependencies(${RVS_TARGET} rvshelper)


install(TARGETS ${RVS_TARGET}
  RUNTIME
  DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs
  COMPONENT applications
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/conf
  DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs
  COMPONENT applications
  FILES_MATCHING PATTERN "*.conf"
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/.rvsmodules.config"
  DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs
  COMPONENT applications
)

# TEST SECTION
if (RVS_BUILD_TESTS)
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests.cmake)
endif()
