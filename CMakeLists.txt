################################################################################
##
## Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
project ("rocm-validation-suite")

# Default libdir to "lib", this skips GNUInstallDirs from trying to take a guess if it's unset:
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")

#including GNUInstall Dir definitions.
include( GNUInstallDirs )

if ( ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  message(FATAL "In-source build is not allowed")
endif ()
enable_testing()

# Prerequisite - Check if rocblas was already installed
find_package (rocblas)
if(rocblas_FOUND)
  message(STATUS "rocblas found")
  message(STATUS "rocblas Version: ${rocblas_VERSION}")
  set(RVS_ROCBLAS_VERSION ${rocblas_VERSION} )
  set(RVS_ROCBLAS_VERSION_MAJ ${rocblas_VERSION_MAJOR} )
  set(RVS_ROCBLAS_VERSION_MIN ${rocblas_VERSION_MINOR} )
  set(RVS_ROCBLAS_VERSION_PAT ${rocblas_VERSION_PATCH} )

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

# Making ROCM_PATH, CMAKE_INSTALL_PREFIX, CMAKE_PACKAGING_INSTALL_PREFIX as CACHE
# variables since we will pass them as cmake params appropriately, and 
# all find_packages relevant to this build will be in ROCM path hence appending it to CMAKE_PREFIX_PATH 
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCM install path")
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "CMAKE installation directory")
set(CMAKE_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Prefix used in built packages")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
set(ROCR_INC_DIR "${ROCM_PATH}/include" CACHE PATH "Contains header files exported by ROC Runtime" FORCE)
set(ROCR_LIB_DIR "${ROCM_PATH}/lib" CACHE PATH "Contains library files exported by ROC Runtime" FORCE)
set(HIP_INC_DIR "${ROCM_PATH}" CACHE PATH "Contains header files exported by ROC Runtime")
set(ROCT_INC_DIR "${ROCM_PATH}/include" CACHE PATH "Contains header files exported by ROC Trunk" FORCE)


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
  OR (${RVS_OS_TYPE} STREQUAL "sles"))
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

## Setup the package version default .
set ( RVS_VERSION "1.0.0" )
get_version ( ${RVS_VERSION} )

# Package Generator  #######################################################
# set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "rocm-validation-suite")
set(CPACK_PACKAGE_DESCRIPTION "ROCm Validation Suite")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ROCm Validation Suite validates the AMD platform using ROCm Runtime - The ROCm Validation Suite is a system administrator and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPUs running in high performance computing environments, enabled using the ROCm software stack on a compatible platform.")
set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices Inc.")
set(CPACK_PACKAGE_CONTACT "ROCM Validation Suite Support <rocm-validation-suite.support@amd.com>")
#set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocblas")
set(CPACK_RPM_PACKAGE_REQUIRES "rocblas")

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL applications rvsmodule)

# Package naming change
set(ROCM_VERSION_FOR_PACKAGE "99999")
# update version accordingly
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

set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

# Set and Install license file
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/doc/${CPACK_PACKAGE_NAME})

set(RVS_COVERAGE FALSE CACHE BOOL "TRUE if code coverage is to be provided")
set(RVS_BUILD_TESTS TRUE CACHE BOOL "TRUE if tests are to be built")

set(RVS_DO_TRACE "1" CACHE STRING "Expand RVSTRACE_ macro")
set(RVS_ROCBLAS "0" CACHE STRING "1 = use local rocBLAS")
set(RVS_ROCMSMI "0" CACHE STRING "1 = use local rocm_smi_lib")

set(RVS_LIB_DIR "${CMAKE_BINARY_DIR}/rvslib" CACHE PATH "Contains RVS library")
set(YAML_INC_DIR "${CMAKE_BINARY_DIR}/yaml-src/include" CACHE PATH "Contains header files exported by yaml-cpp")
set(YAML_LIB_DIR "${CMAKE_BINARY_DIR}/yaml-build" CACHE PATH "Contains library files exported by yaml-cpp")

if (${RVS_OS_TYPE} STREQUAL "centos")
  set(ROCT_LIB_DIR "${ROCM_PATH}/lib64" CACHE PATH "Contains library files exported by ROC Trunk")
elseif (${RVS_OS_TYPE} STREQUAL "rhel")
  set(ROCT_LIB_DIR "${ROCM_PATH}/lib64" CACHE PATH "Contains library files exported by ROC Trunk")
elseif (${RVS_OS_TYPE} STREQUAL "sles")
  set(ROCT_LIB_DIR "${ROCM_PATH}/lib64" CACHE PATH "Contains library files exported by ROC Trunk")
else()
  set(ROCT_LIB_DIR "${ROCM_PATH}/lib" CACHE PATH "Contains library files exported by ROC Trunk")
endif()

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
# Download and unpack yaml-cpp at configure time
configure_file(CMakeYamlDownload.cmake yaml-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/yaml-download )
if(result)
  message(FATAL_ERROR "CMake step for yaml-download failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/yaml-download )
if(result)
  message(FATAL_ERROR "Build step for yaml-download failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR}/yaml-src -B${CMAKE_BINARY_DIR}/yaml-build
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/yaml-src )
if(result)
  message(FATAL_ERROR "Config step for yaml-src failed: ${result}")
endif()

add_custom_target(rvs_yaml_target
  DEPENDS ${CMAKE_BINARY_DIR}/yaml-build/libyaml-cpp.a
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/yaml-build/libyaml-cpp.a
  COMMAND make -C ${CMAKE_BINARY_DIR}/yaml-build
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/yaml-src
  COMMENT "Generating yaml-cpp targets"
  VERBATIM)

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
## rocm_smi_lib

if (RVS_ROCMSMI EQUAL 1)

# This is in order to avoid unneccessary rebuild of rocm_smi_lib
# (there is glitch in cmake when the included project also generates files in
#  the buld process)
if(NOT EXISTS ${CMAKE_BINARY_DIR}/rvs_smi-build/librocm_smi64.so)

execute_process(COMMAND rm -rf "${CMAKE_BINARY_DIR}/rvs_smi-src"
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_smi-download )
# if(result)
#   message(FATAL_ERROR "Error removing rocm_smi-src build folder: ${result}")
# else()
#   message( "rocm_smi-src folders removed: ${result}")
# endif()

execute_process(COMMAND rm -rf "${CMAKE_BINARY_DIR}/rvs_smi-download"
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_smi-download )
# if(result)
#   message(FATAL_ERROR "Error removing rocm_smi-download build folder: ${result}")
# else()
#   message( "rocm_smi-download folders removed: ${result}")
# endif()

execute_process(COMMAND rm -rf "${CMAKE_BINARY_DIR}/rvs_smi-build"
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_smi-build )
# if(result)
#   message(FATAL_ERROR "Error removing rocm_smi-build build folder: ${result}")
# else()
#   message( "rocm_smi-build folders removed: ${result}")
# endif()


# Download and unpack rocm_smi_lib at configure time
configure_file(CMakeRSMIDownload.cmake rvs_smi-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_smi-download )
if(result)
  message(FATAL_ERROR "CMake step for rvs_smi-download failed: ${result}")
endif()

# execute download
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_smi-download )
if(result)
  message(FATAL_ERROR "Build step for rvs_smi-download failed: ${result}")
endif()


# generate MAKE script
execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR}/rvs_smi-src -B${CMAKE_BINARY_DIR}/rvs_smi-build
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_smi-src )
if(result)
  message(FATAL_ERROR "CMake step for rvs_smi-src failed: ${result}")
endif()

# post build step to create symbolic link
# (we want .so i /bin folder to simplify packaging when local smi lib is used)
# add_custom_command(TARGET rvs_rsmi_target POST_BUILD
#   COMMAND rm -f ${CMAKE_BINARY_DIR}/bin/librocm_smi64*
#   COMMAND ln -s ${CMAKE_BINARY_DIR}/rvs_smi-build/librocm_smi64.so ${CMAKE_BINARY_DIR}/bin/librocm_smi64.so
# )

endif()  # if(NOT EXISTS ${CMAKE_BINARY_DIR}/rvs_smi-build/librocm_smi64.so)

# Add target for rocm_smi_lib
add_custom_target(rvs_rsmi_target
  DEPENDS ${CMAKE_BINARY_DIR}/rvs_smi-build/librocm_smi64.so
)

# Define build command for rocm_smi_lib
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/rvs_smi-build/librocm_smi64.so
  COMMAND make -C ${CMAKE_BINARY_DIR}/rvs_smi-build
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/rvs_smi-src
  COMMENT "Generating rocm_smi_lib targets"
  VERBATIM
)

endif()  # if (RVS_ROCMSMI EQUAL 1)

if (RVS_ROCBLAS EQUAL 1)
  set(ROCBLAS_INC_DIR "${CMAKE_BINARY_DIR}/rvs_rblas-src/build/release/rocblas-install")
  set(ROCBLAS_LIB_DIR "${CMAKE_BINARY_DIR}/rvs_rblas-src/build/release/rocblas-install/lib/")
else()
  set(ROCBLAS_INC_DIR "${ROCM_PATH}/include")
  set(ROCBLAS_LIB_DIR "${ROCM_PATH}/lib")
endif()

if (RVS_ROCMSMI EQUAL 1)
  set(ROCM_SMI_INC_DIR "${CMAKE_BINARY_DIR}/rvs_smi-src/include")
  set(ROCM_SMI_LIB_DIR "${CMAKE_BINARY_DIR}/rvs_smi-build")
else()
  #NOTE:Include folders need to be updated to latest include dir as per REORG Folder structure.
  #For backward compatibility we are currently using version details to determine the include path
  #Current Work Around to check ROCMSMI Version is using ROCBLAS REORG VERSION
  if( RVS_ROCBLAS_VERSION_FLAT LESS 2044000 )
    set(ROCM_SMI_INC_DIR "${ROCM_PATH}/rocm_smi/include")
    set(ROCM_SMI_LIB_DIR "${ROCM_PATH}/rocm_smi/lib")
  else()
    message( STATUS "ROCBLAS REORG Enabled Version: ${RVS_ROCBLAS_VERSION_FLAT}" )
    set(ROCM_SMI_INC_DIR "${ROCM_PATH}/include")
    set(ROCM_SMI_LIB_DIR "${ROCM_PATH}/lib")
  endif()
endif()
set(ROCM_SMI_LIB "rocm_smi64" CACHE STRING "rocm_smi library name")

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

set(HCC_CXX_FLAGS  "-fno-gpu-rdc  --offload-arch=gfx90a --offload-arch=gfx1030 --offload-arch=gfx803 --offload-arch=gfx900 --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx1101 --offload-arch=gfx940")

add_subdirectory(rvs)
add_subdirectory(rvslib)
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
#add_subdirectory(edp.so)
add_subdirectory(perf.so)

if (RVS_BUILD_TESTS)
  add_subdirectory(testif.so)
endif()

add_dependencies(rvshelper rvs_bin_folder rvs_doc rvs_yaml_target)


add_dependencies(pesm rvslib rvslibrt)
if(RVS_ROCMSMI EQUAL 1)
  add_dependencies(gm rvs_rsmi_target)
  add_dependencies(iet rvs_rsmi_target)
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
  add_dependencies(rvshelper rvs_bintest_folder rvs_gtest_target)
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
	DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/man/man1
  COMPONENT applications 
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/userguide/html
	DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/${CPACK_PACKAGE_NAME}/userguide
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
	DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/${CPACK_PACKAGE_NAME}/conf/
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
  COMPONENT rvsmodule
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/testscripts/
	DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/${CPACK_PACKAGE_NAME}/testscripts
  USE_SOURCE_PERMISSIONS
  COMPONENT rvsmodule
)

if(RVS_ROCMSMI EQUAL 1)
  install(FILES "${ROCM_SMI_LIB_DIR}/librocm_smi64.so"
          DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
          COMPONENT rvsmodule)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rvsso.conf"
	  DESTINATION /${CMAKE_INSTALL_SYSCONFDIR}/ld.so.conf.d
          COMPONENT rvsmodule)
endif()  # if(RVS_ROCMSMI EQUAL 1)

# TEST SECTION

#add_test(NAME test1 COMMAND rvs)

set(CPACK_RPM_PACKAGE_AUTOREQ 0)

include (CPack)
