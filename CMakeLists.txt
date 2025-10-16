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
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-Wl,--enable-new-dtags,--rpath,$ORIGIN:$ORIGIN/.." CACHE STRING "RUNPATH for libraries")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--enable-new-dtags,--rpath,$ORIGIN/../lib" CACHE STRING "RUNPATH for executables")

project ("rocm-validation-suite"
          VERSION 1.3.0)

# Default libdir to "lib", this skips GNUInstallDirs from trying to take a guess if it's unset:
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")

#including GNUInstall Dir definitions.
include( GNUInstallDirs )

# Set C++ standard as version 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ( ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  message(FATAL "In-source build is not allowed")
endif ()
enable_testing()
# In future use the api provided by rocm-core library to get ROCm Install path
# For backward compatibility use ROCM_PATH provided by build arguments.
# Using the api option is turned OFF by default
option(FETCH_ROCMPATH_FROM_ROCMCORE
        "Use the api provided by rocm-core library to get ROCM_PATH" OFF)

# Prerequisite - Check if rocblas was already installed
find_package (rocblas)
if(rocblas_FOUND)
  message(STATUS "rocblas found")
  message(STATUS "rocblas Version: ${rocblas_VERSION}")
  set(RVS_ROCBLAS_VERSION ${rocblas_VERSION} )
  set(RVS_ROCBLAS_VERSION_MAJ ${rocblas_VERSION_MAJOR} )
  set(RVS_ROCBLAS_VERSION_MIN ${rocblas_VERSION_MINOR} )
  set(RVS_ROCBLAS_VERSION_PAT ${rocblas_VERSION_PATCH} )
  set(ROCBLAS_INC_DIR ${rocblas_INCLUDE_DIR} )
  get_filename_component(ROCBLAS_LIB_DIR "${ROCBLAS_INC_DIR}/../lib" ABSOLUTE)

  if( (RVS_ROCBLAS_VERSION_MAJ GREATER 999) OR (RVS_ROCBLAS_VERSION_MIN GREATER 999) OR (RVS_ROCBLAS_VERSION_PAT GREATER 999) )
    message(FATAL_ERROR "RVS rocblas: Too big ROCBLAS version number(s)" )
  else() #if Valid ROCBLAS Version flags
    math(EXPR RVS_ROCBLAS_VERSION_FLAT  "(${RVS_ROCBLAS_VERSION_MAJ}*1000000+${RVS_ROCBLAS_VERSION_MIN}*1000+${RVS_ROCBLAS_VERSION_PAT})" )
    message( STATUS "rocblas: Version string ${RVS_ROCBLAS_VERSION_FLAT} For Version Check" )
    if(RVS_ROCBLAS_VERSION_FLAT LESS 2044000)
	set(ROCBLAS_MODULE_NM_PREFIX "")
	message( STATUS "ROCBLAS PRE-REORG VERSION: ${RVS_ROCBLAS_VERSION_FLAT}" )
    else() #If Reorg Version
	set(ROCBLAS_MODULE_NM_PREFIX "rocblas/")
	message( STATUS "ROCBLAS REORG VERSION: ${RVS_ROCBLAS_VERSION_FLAT}" )
    endif() #ROCBLAS Version check reorg or not
  endif() #ROCBLAS Valid Version flag check
else() #If ROCBLAS not found
  message(FATAL_ERROR "rocblas not found !!! Install rocblas to proceed ...")
endif(rocblas_FOUND)

find_package (hipblaslt)
if(hipblaslt_FOUND)
  message(STATUS "hipblaslt found")
  set(HIPBLASLT_INC_DIR ${hipblaslt_INCLUDE_DIR} )
  get_filename_component(HIPBLASLT_LIB_DIR "${HIPBLASLT_INC_DIR}/../lib" ABSOLUTE)
else() #If HIPBLASLT not found
  message(FATAL_ERROR "hipblaslt not found !!! Install hipblaslt to proceed ...")
endif(hipblaslt_FOUND)

find_package (hsa-runtime64)
if(hsa-runtime64_FOUND)
  message(STATUS "hsa-runtime64 found")
  get_target_property(HSA_INC_DIR hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
  get_filename_component(HSA_PATH "${HSA_INC_DIR}/.." ABSOLUTE)
  get_filename_component(HSA_LIB_DIR "${HSA_PATH}/lib" ABSOLUTE)
  set(ROCR_INC_DIR ${HSA_INC_DIR})
  set(ROCR_LIB_DIR ${HSA_LIB_DIR})
else() #If HSA not found
  message(FATAL_ERROR "hsa not found !!! Install hsa to proceed ...")
endif(hsa-runtime64_FOUND)

find_package (hip)
if(hip_FOUND)
  set(HIP_INC_DIR ${hip_INCLUDE_DIR})
  set(HIP_LIB_DIR ${hip_LIB_INSTALL_DIR})
else() #If HIP not found
  message(FATAL_ERROR "hip not found !!! Install hip to proceed ...")
endif(hip_FOUND)

find_package (amd_smi)
if(amd_smi_FOUND)
  message(STATUS "amd_smi found")
  set(AMD_SMI_INC_DIR ${amd_smi_INCLUDE_DIR} )
  set(AMD_SMI_LIB_DIR ${amd_smi_LIB_DIR} )
else() #If AMD_SMI not found
  message(FATAL_ERROR "amd-smi not found !!! Install amd-smi to proceed ...")
endif(amd_smi_FOUND)

find_package (hiprand)
if(hiprand_FOUND)
  message(STATUS "hiprand found")
  set(HIPRAND_INC_DIR ${hiprand_INCLUDE_DIR} )
  get_filename_component(HIPRAND_LIB_DIR "${hiprand_INCLUDE_DIR}/../lib" ABSOLUTE)
else() #If HIPRAND not found
  message(FATAL_ERROR "hiprand not found !!! Install hiprand to proceed ...")
endif(hiprand_FOUND)

find_package (rocrand)
if(rocrand_FOUND)
  message(STATUS "rocrand found")
  set(ROCRAND_INC_DIR ${rocrand_INCLUDE_DIR} )
  get_filename_component(ROCRAND_LIB_DIR "${rocrand_INCLUDE_DIR}/../lib" ABSOLUTE)
else() #If ROCRAND not found
  message(FATAL_ERROR "rocrand not found !!! Install rocrand to proceed ...")
endif(rocrand_FOUND)

find_package(hipblas-common)
if(hipblas-common_FOUND)
  message(STATUS "hipblas-common found")
else() #If hipblas-common not found
  message(FATAL_ERROR "hipblas-common not found !!! Install hipblas-common to proceed ...")
endif(hipblas-common_FOUND)

# Making ROCM_PATH, CMAKE_INSTALL_PREFIX, CPACK_PACKAGING_INSTALL_PREFIX as CACHE
# variables since we will pass them as cmake params appropriately, and 
# all find_packages relevant to this build will be in ROCM path hence appending it to CMAKE_PREFIX_PATH 
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCM install path")
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "CMAKE installation directory")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Prefix used in built packages")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

add_definitions(-DROCM_PATH="${ROCM_PATH}")
if(FETCH_ROCMPATH_FROM_ROCMCORE)
  add_compile_options(-DFETCH_ROCMPATH_FROM_ROCMCORE=${FETCH_ROCMPATH_FROM_ROCMCORE})
  # rocm-core library will provide the ROCM_PATH during runtime
  # Link the library as required
  set(ROCM_CORE "rocm-core")
  # The absolute library path will be ${ROCM_PATH}/${RVS_LIB_PATH}
  # Usage is ROCM_PATH will be fetched from rocm-core and append ${RVS_LIB_PATH}
  add_definitions(-DRVS_LIB_PATH="${CMAKE_INSTALL_LIBDIR}/rvs")
else()
  add_definitions(-DRVS_LIB_PATH="${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/rvs")
  # Set ROCM_CORE to empty, so no linking to rocm-core library/package
  set(ROCM_CORE "")
endif()
#
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their
# definition and dump it in the cache along with proper documentation,
# otherwise set CMAKE_BUILD_TYPE to Debug prior to calling PROJECT()
#
if(DEFINED CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
     build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
     Release RelWithDebInfo MinSizeRel.")
else()
   set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build,
     options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
     RelWithDebInfo MinSizeRel.")
endif()

#include ( CTest )

if(NOT CMAKE_CONFIGURATION_TYPES)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/rvs_os_helper.sh
  RESULT_VARIABLE result
  OUTPUT_VARIABLE RVS_OS_TYPE
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (result)
   message(FATAL_ERROR "Could not obtain OS type: ${result}")
endif()
message(STATUS "RVS_OS_TYPE: ${RVS_OS_TYPE}")
if (${RVS_OS_TYPE} STREQUAL "ubuntu")
  set(RVS_OS_TYPE_NUM 1)
elseif ((${RVS_OS_TYPE} STREQUAL "centos") OR (${RVS_OS_TYPE} STREQUAL "rhel")
	OR (${RVS_OS_TYPE} STREQUAL "sles") OR (${RVS_OS_TYPE} STREQUAL "mariner"))
  set(RVS_OS_TYPE_NUM 2)
else()
  set(RVS_OS_TYPE_NUM 0)
endif()
message(STATUS "RVS_OS_TYPE_NUM: ${RVS_OS_TYPE_NUM}")

## Set default module path if not already set
if ( NOT DEFINED CMAKE_MODULE_PATH )
    set ( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" )
endif()

## Include common cmake modules
include ( utils )

## Update RVS version
set ( RVS_VERSION "${PROJECT_VERSION}" )
get_version ( ${RVS_VERSION} )

# Package Generator  #######################################################
set(CPACK_PACKAGE_NAME "rocm-validation-suite")
set(CPACK_PACKAGE_DESCRIPTION "ROCm Validation Suite")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Tool for validating AMD GPU functionality, performance, reliability in ROCm.")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices Inc.")
set(CPACK_PACKAGE_CONTACT "ROCM Validation Suite Support <rocm-validation-suite.support@amd.com>")

# Package dependencies
if(FETCH_ROCMPATH_FROM_ROCMCORE)
  set(ROCM_DEPENDENT_PACKAGES "hip-runtime-amd, comgr, hsa-rocr, rocblas, amd-smi-lib, hiprand, hipblaslt, ${ROCM_CORE}")
else()
  set(ROCM_DEPENDENT_PACKAGES "hip-runtime-amd, comgr, hsa-rocr, rocblas, amd-smi-lib, hiprand, hipblaslt")
endif()
if (${RVS_OS_TYPE} STREQUAL "ubuntu")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "${ROCM_DEPENDENT_PACKAGES}, libpci3, libyaml-cpp-dev")
elseif (${RVS_OS_TYPE} STREQUAL "sles")
set(CPACK_RPM_PACKAGE_REQUIRES "${ROCM_DEPENDENT_PACKAGES}, libpci3, libyaml-cpp0_6")
else ()
# other supported rpm distros - RHEL, CentOS
set(CPACK_RPM_PACKAGE_REQUIRES "${ROCM_DEPENDENT_PACKAGES}, pciutils-libs, yaml-cpp")
endif()

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL applications rvsmodule)

# Set default ROCm version for package naming
set(ROCM_VERSION_FOR_PACKAGE "99999")

# Update ROCm version for package naming
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
endif()

# debian related changes
if (DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
   set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
   set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()

# RMP related changes
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

# distro changes
if(CPACK_RPM_PACKAGE_RELEASE)
  set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
endif()

# Complete RVS package version naming
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

# Set and Install license file
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/doc/${CPACK_PACKAGE_NAME})

set(RVS_COVERAGE FALSE CACHE BOOL "TRUE if code coverage is to be provided")
set(RVS_BUILD_TESTS TRUE CACHE BOOL "TRUE if tests are to be built")

set(RVS_DO_TRACE "1" CACHE STRING "Expand RVSTRACE_ macro")
set(RVS_ROCBLAS "0" CACHE STRING "1 = use local rocBLAS")
#set(RVS_ROCMSMI "0" CACHE STRING "1 = use local amd_smi_lib")
set(RVS_AMDSMI "0" CACHE STRING "1 = use local amd_smi_lib")

set(RVS_LIB_DIR "${CMAKE_BINARY_DIR}/rvslib" CACHE PATH "Contains RVS library")

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "RVS_COVERAGE = ${RVS_COVERAGE}")
message(STATUS "RVS_BUILD_TESTS = ${RVS_BUILD_TESTS}")

message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")


if (DEFINED RVS_DO_TRACE)
  if (${RVS_DO_TRACE} STREQUAL "1")
    add_definitions(-DRVS_DO_TRACE)
    message(STATUS "RVS_DO_TRACE defined")
  else()
      message(STATUS "RVS_DO_TRACE not defined")
  endif()
else()
    message(STATUS "RVS_DO_TRACE not defined")
endif()


## Set default module path if not already set
if (NOT DEFINED CPACK_GENERATOR )
  if (RVS_OS_TYPE_NUM EQUAL 1)
    set (CPACK_GENERATOR "DEB")
  elseif (RVS_OS_TYPE_NUM EQUAL 2)
    set (CPACK_GENERATOR "RPM")
  else()
    set (CPACK_GENERATOR "DEB;RPM")
  endif()
endif()
message (STATUS "CPACK_GENERATOR ${CPACK_GENERATOR}" )


################################################################################
# check yaml-cpp available at configure time
# check yaml-cpp available at configure time
# First try to find static yaml-cpp library
find_library(YAML_CPP_STATIC_LIB 
  NAMES libyaml-cpp.a yaml-cpp.a
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib ${CMAKE_PREFIX_PATH}/lib64 ${CMAKE_PREFIX_PATH}/lib
  DOC "Path to yaml-cpp static library"
)

find_path(YAML_CPP_INCLUDE_DIR 
  NAMES yaml-cpp/yaml.h
  PATHS /usr/include /usr/local/include ${CMAKE_PREFIX_PATH}/include
  DOC "Path to yaml-cpp include directory"
)

# Set variables for yaml-cpp linking
set(YAML_CPP_STATIC_FOUND FALSE)
set(YAML_CPP_SHARED_FOUND FALSE)

if (YAML_CPP_STATIC_LIB AND YAML_CPP_INCLUDE_DIR)
  set(YAML_CPP_STATIC_FOUND TRUE)
  message(STATUS "yaml-cpp static library found: ${YAML_CPP_STATIC_LIB}")
  set(YAML_CPP_LIBRARIES ${YAML_CPP_STATIC_LIB})
  get_filename_component(YAML_CPP_LIBRARY_DIR ${YAML_CPP_STATIC_LIB} DIRECTORY)
  
  # Create imported target for static linking
  add_library(yaml-cpp-static STATIC IMPORTED)
  set_target_properties(yaml-cpp-static PROPERTIES
    IMPORTED_LOCATION ${YAML_CPP_STATIC_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${YAML_CPP_INCLUDE_DIR}
  )
  
  # For static linking, we might need pthread
  find_package(Threads REQUIRED)
  set_target_properties(yaml-cpp-static PROPERTIES
    INTERFACE_LINK_LIBRARIES Threads::Threads
  )
  
  # Use static library as the target
  set(YAML_CPP_TARGET yaml-cpp-static)
  
else()
  # Fall back to find_package for shared library
  find_package(yaml-cpp QUIET)
  if (yaml-cpp_FOUND)
    set(YAML_CPP_SHARED_FOUND TRUE)
    message(STATUS "yaml-cpp shared library found: ${YAML_CPP_LIBRARIES}")
    
    if (TARGET yaml-cpp::yaml-cpp)
      set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
      get_target_property(YAML_CPP_INCLUDE_DIR yaml-cpp::yaml-cpp INTERFACE_INCLUDE_DIRECTORIES)
      if (NOT YAML_CPP_INCLUDE_DIR)
        set(YAML_CPP_INCLUDE_DIR ${YAML_CPP_INCLUDE_DIRS})
      endif()
    else()
      # Legacy variables from find_package
      set(YAML_CPP_TARGET ${YAML_CPP_LIBRARIES})
    endif()
  else()
    message(FATAL_ERROR "yaml-cpp not found !!! Install yaml-cpp (preferably static version) to proceed ...")
  endif()
endif()

# Final status message
if (YAML_CPP_STATIC_FOUND)
  message(STATUS "Using yaml-cpp STATIC library: ${YAML_CPP_LIBRARIES}")
elseif (YAML_CPP_SHARED_FOUND)
  message(STATUS "Using yaml-cpp SHARED library: ${YAML_CPP_LIBRARIES}")
endif()

################################################################################
## GOOGLE TEST
if(RVS_BUILD_TESTS)
  # Download and unpack googletest at configure time
  configure_file(CMakeGtestDownload.cmake googletest-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR}/googletest-src -B${CMAKE_BINARY_DIR}/googletest-build
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-src )
  if(result)
    message(FATAL_ERROR "Config step for googletest-src failed: ${result}")
  endif()

  add_custom_target(rvs_gtest_target
    DEPENDS ${CMAKE_BINARY_DIR}/googletest-build/lib/libgtest_main.a
  )

  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/googletest-build/lib/libgtest_main.a
    COMMAND make -C ${CMAKE_BINARY_DIR}/googletest-build
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-src
    COMMENT "Generating googletest targets"
    VERBATIM)

  ## Set default unit test framework include path
  if (NOT DEFINED UT_INC)
      set (UT_INC "${CMAKE_BINARY_DIR}/googletest-src/googletest/include")
      message ("UT_INC ${UT_INC}")
  endif ()

  ## Set default unit test framework include path
  if (NOT DEFINED UT_LIB)
      set (UT_LIB "${CMAKE_BINARY_DIR}/googletest-build/lib")
      message ("UT_LIB ${UT_LIB}")
  endif()

endif()
################################################################################

################################################################################
# mxDataGenerator Download
configure_file(CMakeMXDataGeneratorDownload.cmake mxDataGenerator-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mxDataGenerator-download )
if(result)
  message(FATAL_ERROR "CMake step for mxDataGenerator failed: ${result}")
endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mxDataGenerator-download )
if(result)
  message(FATAL_ERROR "Build step for mxDataGenerator failed: ${result}")
endif()
  execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR}/mxDataGenerator-src -B${CMAKE_BINARY_DIR}/mxDataGenerator-build
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mxDataGenerator-src )
if(result)
  message(FATAL_ERROR "Config step for mxDataGenerator-src failed: ${result}")
endif()

if (NOT DEFINED MXDATAGENERATOR_INC_DIR)
  set (MXDATAGENERATOR_INC_DIR "${CMAKE_BINARY_DIR}/mxDataGenerator-src/lib/include")
  message ("MXDATAGENERATOR_INC_DIR ${MXDATAGENERATOR_INC_DIR}")
endif ()
################################################################################

## rocBLAS

if(RVS_ROCBLAS EQUAL 1)

  # Download and unpack rocBLAS at configure time
  configure_file(CMakeRBLASDownload.cmake rvs_rblas-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_rblas-download )
  if(result)
    message(FATAL_ERROR "CMake step for rocBLAS failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_rblas-download )
  if(result)
    message(FATAL_ERROR "Build step for rocBLAS failed: ${result}")
  endif()

  add_custom_target(rvs_rblas_target
    DEPENDS ${CMAKE_BINARY_DIR}/rvs_rblas-src/build/release/rocblas-install/lib/librocblas.so
  )
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/rvs_rblas-src/build/release/rocblas-install/lib/librocblas.so
    COMMAND ${CMAKE_BINARY_DIR}/rvs_rblas-src/install.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_rblas-src
    COMMENT "Generating rocBLAS libraries"
    VERBATIM)

endif()  # if(RVS_ROCBLAS EQUAL 1)

################################################################################
## amd_smi_lib

if (RVS_AMDSMI EQUAL 1)

# This is in order to avoid unneccessary rebuild of amd_smi_lib
# (there is glitch in cmake when the included project also generates files in
#  the buld process)
if(NOT EXISTS ${CMAKE_BINARY_DIR}/amd_smi-build/libamd_smi.so)

execute_process(COMMAND rm -rf "${CMAKE_BINARY_DIR}/amd_smi-src"
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/amd_smi-download )
# if(result)
#   message(FATAL_ERROR "Error removing amd_smi-src build folder: ${result}")
# else()
#   message( "amd_smi-src folders removed: ${result}")
# endif()

execute_process(COMMAND rm -rf "${CMAKE_BINARY_DIR}/amd_smi-download"
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/amd_smi-download )
# if(result)
#   message(FATAL_ERROR "Error removing amd_smi-download build folder: ${result}")
# else()
#   message( "amd_smi-download folders removed: ${result}")
# endif()

execute_process(COMMAND rm -rf "${CMAKE_BINARY_DIR}/amd_smi-build"
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/amd_smi-build )
# if(result)
#   message(FATAL_ERROR "Error removing amd_smi-build build folder: ${result}")
# else()
#   message( "amd_smi-build folders removed: ${result}")
# endif()


# Download and unpack amd_smi_lib at configure time
configure_file(CMakeRSMIDownload.cmake amd_smi-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/amd_smi-download )
if(result)
  message(FATAL_ERROR "CMake step for amd_smi-download failed: ${result}")
endif()

# execute download
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/amd_smi-download )
if(result)
  message(FATAL_ERROR "Build step for amd_smi-download failed: ${result}")
endif()


# generate MAKE script
execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR}/amd_smi-src -B${CMAKE_BINARY_DIR}/amd_smi-build
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/amd_smi-src )
if(result)
  message(FATAL_ERROR "CMake step for amd_smi-src failed: ${result}")
endif()

# post build step to create symbolic link
# (we want .so i /bin folder to simplify packaging when local smi lib is used)
# add_custom_command(TARGET rvs_smi_target POST_BUILD
#   COMMAND rm -f ${CMAKE_BINARY_DIR}/bin/libamd_smi64*
#   COMMAND ln -s ${CMAKE_BINARY_DIR}/amd_smi-build/libamd_smi64.so ${CMAKE_BINARY_DIR}/bin/libamd_smi64.so
# )

endif()  # if(NOT EXISTS ${CMAKE_BINARY_DIR}/amd_smi-build/libamd_smi64.so)

# Add target for amd_smi_lib
add_custom_target(rvs_smi_target
  DEPENDS ${CMAKE_BINARY_DIR}/amd_smi-build/libamd_smi.so
)

# Define build command for amd_smi_lib
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/amd_smi-build/libamd_smi.so
  COMMAND make -C ${CMAKE_BINARY_DIR}/amd_smi-build
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/amd_smi-src
  COMMENT "Generating amd_smi_lib targets"
  VERBATIM
)

endif()  # if (RVS_AMDSMI EQUAL 1)

if (RVS_ROCBLAS EQUAL 1)
  set(ROCBLAS_INC_DIR "${CMAKE_BINARY_DIR}/rvs_rblas-src/build/release/rocblas-install")
  set(ROCBLAS_LIB_DIR "${CMAKE_BINARY_DIR}/rvs_rblas-src/build/release/rocblas-install/lib/")
endif()

if (RVS_ROCMSMI EQUAL 1)
  set(AMD_SMI_INC_DIR "${CMAKE_BINARY_DIR}/amd_smi-src/include")
  set(AMD_SMI_LIB_DIR "${CMAKE_BINARY_DIR}/amd_smi-build")
endif()
set(AMD_SMI_LIB "amd_smi" CACHE STRING "amd_smi library name")

# Determine HSA_PATH
if(NOT DEFINED HSA_PATH)
     if(NOT DEFINED ENV{HSA_PATH})
          set(HSA_PATH "${ROCM_PATH}" CACHE PATH "Path to which HSA runtime has been installed")
     else()
          set(HSA_PATH $ENV{HSA_PATH} CACHE PATH "Path to which HSA runtime has been installed")
     endif()
endif()

set(RVS_BINTEST_FOLDER ${CMAKE_BINARY_DIR}/bintest)
# Check if Address Sanitizer enabled.
if(BUILD_ADDRESS_SANITIZER)
  set(ASAN_CXX_FLAGS "-fsanitize=address  -shared-libasan -O0")
  set(ASAN_LD_FLAGS "-fuse-ld=lld")
endif()

set(HCC_CXX_FLAGS "-fno-gpu-rdc --offload-arch=gfx90a --offload-arch=gfx1030 --offload-arch=gfx803 --offload-arch=gfx900 --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx1101 --offload-arch=gfx942 --offload-arch=gfx1200 --offload-arch=gfx1201 --offload-arch=gfx1100 --offload-arch=gfx950")

add_subdirectory(rvslib)
add_subdirectory(rvs)
add_subdirectory(gm.so)
add_subdirectory(gpup.so)
add_subdirectory(gst.so)
add_subdirectory(iet.so)
add_subdirectory(pebb.so)
add_subdirectory(peqt.so)
add_subdirectory(pesm.so)
add_subdirectory(pbqt.so)
add_subdirectory(rcqt.so)
add_subdirectory(smqt.so)
add_subdirectory(mem.so)
add_subdirectory(babel.so)
add_subdirectory(perf.so)
add_subdirectory(tst.so)

if (RVS_BUILD_TESTS)
  add_subdirectory(testif.so)
endif()

add_dependencies(pesm rvslib)
if(RVS_ROCMSMI EQUAL 1)
  add_dependencies(gm rvs_smi_target)
  add_dependencies(iet rvs_smi_target)
endif()

if(RVS_ROCBLAS EQUAL 1)
  add_dependencies(rvslib rvs_rblas_target)
  add_dependencies(gst rvs_rblas_target)
  add_dependencies(iet rvs_rblas_target)
endif()

add_custom_target(rvs_bin_folder ALL
COMMAND ${CMAKE_COMMAND} -E make_directory "bin"
COMMAND rm -f ${CMAKE_BINARY_DIR}/bin/.rvsmodules.config
COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/rvs/.rvsmodules.config ${CMAKE_BINARY_DIR}/bin/.rvsmodules.config
COMMAND rm -f ${CMAKE_BINARY_DIR}/bin/conf
COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/rvs/conf ${CMAKE_BINARY_DIR}/bin
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
COMMENT "Create the bin directory"
VERBATIM)

if (RVS_BUILD_TESTS)
  add_custom_target(rvs_bintest_folder ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RVS_BINTEST_FOLDER}
  COMMAND rm -f ${RVS_BINTEST_FOLDER}/.rvsmodules.config
  COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/rvs/.rvsmodules.config ${RVS_BINTEST_FOLDER}/.rvsmodules.config
  COMMAND rm -f ${RVS_BINTEST_FOLDER}/conf
  COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/rvs/conf ${RVS_BINTEST_FOLDER}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Create the bintest directory"
  VERBATIM)
  add_dependencies(rvs_bintest_folder rvs_gtest_target)
endif()

add_custom_target(rvs_doc ALL
COMMAND ${CMAKE_COMMAND} -E make_directory "doc"
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
COMMENT "Create the doc directory"
VERBATIM)

add_custom_target(copy-files ALL
  COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_SOURCE_DIR}/testscripts/*.sh
      ${CMAKE_BINARY_DIR}/bin/)

find_package(Doxygen)
if (DOXYGEN_FOUND)
message(STATUS "Doxygen: ${DOXYGEN_EXECUTABLE}")
add_custom_target(rvs_progguide ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} ${DOXYGEN_EXECUTABLE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS Programmer's Reference Guide"
    VERBATIM)
add_dependencies(rvs_progguide rvs_doc)

add_custom_target(rvs_man1 ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} ${DOXYGEN_EXECUTABLE} DoxyMan1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS man.1 files"
    VERBATIM)
add_dependencies(rvs_man1 rvs_doc)

add_custom_target(rvs_userguide ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} ${DOXYGEN_EXECUTABLE} DoxyGuide
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS User Guide"
    VERBATIM)
add_dependencies(rvs_userguide rvs_doc)

install(DIRECTORY "${CMAKE_BINARY_DIR}/doc/man/man1/"
	DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/man/man1
  COMPONENT applications 
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/userguide/html
	DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/${CPACK_PACKAGE_NAME}/userguide
  COMPONENT applications
)

else()
  message("Doxygen is not found. Will not generate documentation.")
endif(DOXYGEN_FOUND)

# make changes to deviceid.sh.in and rename it to deviceid.sh
configure_file(${CMAKE_SOURCE_DIR}/rvs/conf/deviceid.sh.in 
  ${CMAKE_SOURCE_DIR}/rvs/conf/deviceid.sh
  @ONLY)
install(FILES ${CMAKE_SOURCE_DIR}/rvs/conf/deviceid.sh
	DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/${CPACK_PACKAGE_NAME}/conf/
        COMPONENT rvsmodule
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/testscripts/
	DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/${CPACK_PACKAGE_NAME}/testscripts
  COMPONENT rvsmodule
)

if(RVS_ROCMSMI EQUAL 1)
  install(FILES "${AMD_SMI_LIB_DIR}/libamd_smi64.so"
          DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
          COMPONENT rvsmodule)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rvsso.conf"
	  DESTINATION /${CMAKE_INSTALL_SYSCONFDIR}/ld.so.conf.d
          COMPONENT rvsmodule)
endif()  # if(RVS_ROCMSMI EQUAL 1)


# Create and install rvs-config.cmake & rvs-config-version.cmake file
include(CMakePackageConfigHelpers)

set(LIB_INSTALL_DIR  "${CMAKE_INSTALL_LIBDIR}")
set(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_INCLUDEDIR}")
set(BIN_INSTALL_DIR  "${CMAKE_INSTALL_BINDIR}")

configure_package_config_file(rvs-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rvs-config.cmake
    INSTALL_DESTINATION ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/rvs
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/rvs-config-version.cmake
    VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rvs-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/rvs-config-version.cmake
    DESTINATION
    ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/rvs
    COMPONENT dev)

# TEST SECTION

#add_test(NAME test1 COMMAND rvs)

set(CPACK_RPM_PACKAGE_AUTOREQ 0)

include (CPack)
