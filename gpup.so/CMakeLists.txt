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
add_compile_options(-std=c++11)

set ( RVS "gpup" )
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

## Setup the package version.
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


## additional libraries
## set ( PROJECT_LINK_LIBS libdl.so )

## define include directories
include_directories(include ../include)

## define source files
set(SOURCES src/rvs_module.cpp src/action.cpp ../src/rvslib.cpp ../src/rvsloglp.cpp)

## define target
add_library( ${RVS_TARGET} SHARED ${SOURCES})
set_target_properties(${RVS_TARGET} PROPERTIES SUFFIX .so.${LIB_VERSION_STRING}) 
#target_link_libraries(${RVS_TARGET} ${PROJECT_LINK_LIBS} )

add_custom_command(TARGET ${RVS_TARGET} POST_BUILD 
COMMAND rm -f ${CMAKE_CURRENT_BINARY_DIR}/../bin/lib${RVS}.so.${VERSION_MAJOR}
COMMAND ln -s ${CMAKE_CURRENT_BINARY_DIR}/lib${RVS}.so.${LIB_VERSION_STRING} ${CMAKE_CURRENT_BINARY_DIR}/../bin/lib${RVS}.so.${VERSION_MAJOR})



