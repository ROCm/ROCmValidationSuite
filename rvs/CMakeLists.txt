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
add_compile_options(-std=c++11)
add_compile_options(-pthread)
add_compile_options(-Wall -Werror)

message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

## Support for in-source compilaton
if ( ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin" )
else()
    set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )
endif ()

set ( RVS "rvs" )
set ( RVS_PACKAGE "rvs-roct" )
set ( RVS_COMPONENT "lib${RVS}" )
set ( RVS_TARGET "${RVS}" )
set(CMAKE_BUILD_TYPE Debug)

project ( ${RVS_TARGET} )

## Set default module path if not already set
if ( NOT DEFINED CMAKE_MODULE_PATH )
    set ( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules/" )
endif ()

## Include common cmake modules
include ( utils )

## Setup the package version default (the actual one is taken from the latest Git tag).
get_version ( "0.0.0" )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( BUILD_VERSION_MINOR ${VERSION_MINOR} )
set ( BUILD_VERSION_PATCH ${VERSION_PATCH} )
set ( LIB_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

if ( DEFINED VERSION_BUILD AND NOT ${VERSION_BUILD} STREQUAL "" )
    message ( "VERSION BUILD DEFINED ${VERSION_BUILD}" )
    set ( BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}" )
endif ()
set ( BUILD_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

## make version numbers visible to C code
add_compile_options(-DBUILD_VERSION_MAJOR=${VERSION_MAJOR})
add_compile_options(-DBUILD_VERSION_MINOR=${VERSION_MINOR})
add_compile_options(-DBUILD_VERSION_PATCH=${VERSION_PATCH})
add_compile_options(-DLIB_VERSION_STRING="${LIB_VERSION_STRING}")
add_compile_options(-DBUILD_VERSION_STRING="${BUILD_VERSION_STRING}")

# Set project requirements
set(ROC_THUNK_NAME "hsakmt")
set(CORE_RUNTIME_NAME "hsa-runtime")
set(ROC_THUNK_LIBRARY "lib${ROC_THUNK_NAME}")
set(CORE_RUNTIME_TARGET "${CORE_RUNTIME_NAME}64")
set(CORE_RUNTIME_LIBRARY "lib${CORE_RUNTIME_TARGET}")

# Determine Roc Runtime header files are accessible
if(NOT EXISTS ${ROCR_INC_DIR}/hsa/hsa.h)
  message("ERROR: ROC Runtime headers can't be found under specified path. Please set ROCR_INC_DIR path. Current value is : " ${ROCR_INC_DIR})
  RETURN()
endif()

if(NOT EXISTS ${ROCR_LIB_DIR}/${CORE_RUNTIME_LIBRARY}.so)
  message("ERROR: ROC Runtime libraries can't be found under specified path. Please set ROCR_LIB_DIR path. Current value is : " ${ROCR_LIB_DIR})
  RETURN()
endif()

# Add directories to look for header files to compile
INCLUDE_DIRECTORIES(${ROCR_INC_DIR})

# Add directories to look for library files to link 
LINK_DIRECTORIES(${ROCR_LIB_DIR})

## additional libraries
set ( PROJECT_LINK_LIBS libdl.so "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../yaml-cpp/libyaml-cpp.a" libpthread.so)

## define include directories
include_directories(include ../include ${CMAKE_CURRENT_SOURCE_DIR}/../yaml-cpp/include )

## define source files
set(SOURCES
  src/rvs.cpp src/rvsmodule.cpp src/rvsif_base.cpp src/rvsif0.cpp src/rvsif1.cpp src/rvsaction.cpp src/rvscli.cpp src/rvsexec.cpp src/rvsexec_do_yaml.cpp 
  src/rvsoptions.cpp
  ../src/rvsliblogger.cpp ../src/rvslognodebase.cpp ../src/rvslognoderec.cpp ../src/rvslognode.cpp ../src/rvslognodestring.cpp ../src/rvslognodeint.cpp ../src/rvsloglp.cpp ../src/rvshsa.cpp
)

## define target
add_executable(${RVS_TARGET} ${SOURCES})

target_link_libraries(${RVS_TARGET} ${PROJECT_LINK_LIBS} ${CORE_RUNTIME_TARGET} ${ROC_THUNK_NAME})

add_custom_command(TARGET ${RVS_TARGET} POST_BUILD
COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/.rvsmodules.config
COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/.rvsmodules.config ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/.rvsmodules.config
COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/conf
COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/conf ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/conf
)

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

install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../doc/man/man1/rvs.1"
  DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs/man
  COMPONENT applications
)

install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../doc/userguide/html
  DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs/userguide
  COMPONENT applications
)

