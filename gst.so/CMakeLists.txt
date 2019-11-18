################################################################################
##
## Copyright (c) 2018 ROCm Developer Tools
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

set ( RVS "gst" )
set ( RVS_PACKAGE "rvs-roct" )
set ( RVS_COMPONENT "lib${RVS}" )
set ( RVS_TARGET "${RVS}" )

project ( ${RVS_TARGET} )

message(STATUS "MODULE: ${RVS}")
add_compile_options(-std=c++11)
add_compile_options(-Wall -Wextra)
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

## Setup the package version.
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

set(ROCBLAS_LIB "rocblas")
set(HIP_HCC_LIB "hip_hcc")

# Determine Roc Runtime header files are accessible
if(NOT EXISTS ${HIP_INC_DIR}/include/hip/hip_runtime.h)
  message("ERROR: ROC Runtime headers can't be found under specified path. Please set HIP_INC_DIR path. Current value is : " ${HIP_INC_DIR})
  RETURN()
endif()

if(NOT EXISTS ${HIP_INC_DIR}/include/hip/hip_runtime_api.h)
  message("ERROR: ROC Runtime headers can't be found under specified path. Please set HIP_INC_DIR path. Current value is : " ${HIP_INC_DIR})
  RETURN()
endif()

# Determine Roc Runtime header files are accessible
if(DEFINED RVS_ROCMSMI)
  if(NOT RVS_ROCMSMI EQUAL 1)
    if(NOT EXISTS ${ROCBLAS_INC_DIR}/rocblas.h)
    message("ERROR: rocBLAS headers can't be found under specified path. Please set ROCBLAS_INC_DIR path. Current value is : " ${ROCBLAS_INC_DIR})
    RETURN()
    endif()

    if(NOT EXISTS "${ROCBLAS_LIB_DIR}/lib${ROCBLAS_LIB}.so")
      message("ERROR: rocBLAS library can't be found under specified path. Please set ROCBLAS_LIB_DIR path. Current value is : " ${ROCBLAS_LIB_DIR})
      RETURN()
    endif()
  endif()
endif()


if(NOT EXISTS "${ROCR_LIB_DIR}/lib${HIP_HCC_LIB}.so")
  message("ERROR: ROC Runtime libraries can't be found under specified path. Please set ROCR_LIB_DIR path. Current value is : " ${ROCR_LIB_DIR})
  RETURN()
endif()


## define include directories
include_directories(./ ../ ${ROCR_INC_DIR} ${ROCBLAS_INC_DIR} ${HIP_INC_DIR})
# Add directories to look for library files to link
link_directories(${RVS_LIB_DIR} ${ROCR_LIB_DIR} ${ROCBLAS_LIB_DIR})
## additional libraries
set (PROJECT_LINK_LIBS rvslibrt rvslib libpthread.so libpci.so libm.so)

## define source files
set(SOURCES src/rvs_module.cpp src/action.cpp src/gst_worker.cpp)

## define target
add_library( ${RVS_TARGET} SHARED ${SOURCES})
set_target_properties(${RVS_TARGET} PROPERTIES
        SUFFIX .so.${LIB_VERSION_STRING}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
target_link_libraries(${RVS_TARGET} ${PROJECT_LINK_LIBS} ${HIP_HCC_LIB} ${ROCBLAS_LIB})
add_dependencies(${RVS_TARGET} rvslibrt rvslib)

add_custom_command(TARGET ${RVS_TARGET} POST_BUILD
COMMAND ln -fs ./lib${RVS}.so.${LIB_VERSION_STRING} lib${RVS}.so.${VERSION_MAJOR} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
COMMAND ln -fs ./lib${RVS}.so.${VERSION_MAJOR} lib${RVS}.so WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

install(TARGETS ${RVS_TARGET} LIBRARY DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs COMPONENT rvsmodule)
install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${RVS}.so.${VERSION_MAJOR}" DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs COMPONENT rvsmodule)
install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${RVS}.so" DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs COMPONENT rvsmodule)

# TEST SECTION
if (RVS_BUILD_TESTS)
  add_custom_command(TARGET ${RVS_TARGET} POST_BUILD
  COMMAND ln -fs ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${RVS}.so.${VERSION_MAJOR} ${RVS_BINTEST_FOLDER}/lib${RVS}.so WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests.cmake)
endif()
