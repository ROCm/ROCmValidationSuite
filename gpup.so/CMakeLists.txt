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

set ( RVS "gpup" )
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


# Determine HSA_PATH
if(NOT DEFINED HIPCC_PATH)
  if(NOT DEFINED ENV{HIPCC_PATH})
    set(HIPCC_PATH "${ROCM_PATH}/hip" CACHE PATH "Path to which hipcc runtime has been installed")
     else()
       set(HIPCC_PATH $ENV{HIPCC_PATH} CACHE PATH "Path to which hipcc runtime has been installed")
     endif()
endif()

# Determine HSA_PATH
if(NOT DEFINED HSA_PATH)
     if(NOT DEFINED ENV{HSA_PATH})
          set(HSA_PATH "${ROCM_PATH}/hsa" CACHE PATH "Path to which HSA runtime has been installed")
     else()
          set(HSA_PATH $ENV{HSA_PATH} CACHE PATH "Path to which HSA runtime has been installed")
     endif()
endif()

# Add HIP_VERSION to CMAKE_<LANG>_FLAGS
set(HIP_HCC_BUILD_FLAGS "${HIP_HCC_BUILD_FLAGS} -DHIP_VERSION_MAJOR=${HIP_VERSION_MAJOR} -DHIP_VERSION_MINOR=${HIP_VERSION_MINOR} -DHIP_VERSION_PATCH=${HIP_VERSION_GITDATE}")

# Add remaining flags
set(HCC_CXX_FLAGS  "-Xlinker --enable-new-dtags -fno-gpu-rdc --offload-arch=gfx803 --offload-arch=gfx900 --offload-arch=gfx906 --offload-arch=gfx908 ")
set(HIP_HCC_BUILD_FLAGS)
set(HIP_HCC_BUILD_FLAGS "${HIP_HCC_BUILD_FLAGS} -fPIC ${HCC_CXX_FLAGS} -I${HSA_PATH}/include")


# Set compiler and compiler flags
set(CMAKE_CXX_COMPILER "${HIPCC_PATH}/bin/hipcc")
set(CMAKE_C_COMPILER   "${HIPCC_PATH}/bin/hipcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_BUILD_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${HIP_HCC_BUILD_FLAGS}")



## define include directories
include_directories(./ ../ include ../include)
# Add directories to look for library files to link
link_directories(${RVS_LIB_DIR})
## additional libraries
set (PROJECT_LINK_LIBS rvslibrt rvslib libpci.so libm.so)

## define source files
set(SOURCES src/rvs_module.cpp src/action.cpp)

## define target
add_library( ${RVS_TARGET} SHARED ${SOURCES})
set_target_properties(${RVS_TARGET} PROPERTIES
        SUFFIX .so.${LIB_VERSION_STRING}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

target_link_libraries(${RVS_TARGET} ${PROJECT_LINK_LIBS} )
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
