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
if ( ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  message(FATAL "In-source build is not allowed")
endif ()

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

project ("rocm-validation-suite" )
enable_testing()
#include ( CTest )


if (NOT CMAKE_CONFIGURATION_TYPES)
#  message( "CMAKE_CONFIGURATION_TYPES not set")
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
if(result)
   message(FATAL_ERROR "Could not obtain OS type: ${result}")
endif()
message(STATUS "RVS_OS_TYPE: ${RVS_OS_TYPE}")
if (${RVS_OS_TYPE} STREQUAL "ubuntu")
  set(RVS_OS_TYPE_NUM 1)
elseif (${RVS_OS_TYPE} STREQUAL "centos")
  set(RVS_OS_TYPE_NUM 2)
elseif (${RVS_OS_TYPE} STREQUAL "rhel")
  set(RVS_OS_TYPE_NUM 2)
elseif (${RVS_OS_TYPE} STREQUAL "sles")
  set(RVS_OS_TYPE_NUM 2)
else()
  set(RVS_OS_TYPE_NUM 0)
endif()
message(STATUS "RVS_OS_TYPE_NUM: ${RVS_OS_TYPE_NUM}")

## Set default module path if not already set
if ( NOT DEFINED CMAKE_MODULE_PATH )
    set ( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" )
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

set(RVS_COVERAGE FALSE CACHE BOOL "TRUE if code coverage is to be provided")
set(RVS_BUILD_TESTS TRUE CACHE BOOL "TRUE if tests are to be built")

set(RVS_DO_TRACE "1" CACHE STRING "Expand RVSTRACE_ macro")
set(RVS_ROCBLAS "0" CACHE STRING "1 = use local rocBLAS")
set(RVS_ROCMSMI "0" CACHE STRING "1 = use local rocm_smi_lib")

set(RVS_LIB_DIR "${CMAKE_BINARY_DIR}/rvslib" CACHE PATH "Contains RVS library")
set(YAML_INC_DIR "${CMAKE_BINARY_DIR}/yaml-src/include" CACHE PATH "Contains header files exported by yaml-cpp")
set(YAML_LIB_DIR "${CMAKE_BINARY_DIR}/yaml-build" CACHE PATH "Contains library files exported by yaml-cpp")
set(ROCR_INC_DIR "/opt/rocm/include" CACHE PATH "Contains header files exported by ROC Runtime")
set(ROCR_LIB_DIR "/opt/rocm/lib" CACHE PATH "Contains library files exported by ROC Runtime")
set(HIP_INC_DIR "/opt/rocm/hip" CACHE PATH "Contains header files exported by ROC Runtime")
set(ROCT_INC_DIR "/opt/rocm/include" CACHE PATH "Contains header files exported by ROC Trunk")
if (${RVS_OS_TYPE} STREQUAL "centos")
  set(ROCT_LIB_DIR "/opt/rocm/lib64" CACHE PATH "Contains library files exported by ROC Trunk")
elseif (${RVS_OS_TYPE} STREQUAL "rhel")
  set(ROCT_LIB_DIR "/opt/rocm/lib64" CACHE PATH "Contains library files exported by ROC Trunk")
elseif (${RVS_OS_TYPE} STREQUAL "sles")
  set(ROCT_LIB_DIR "/opt/rocm/lib64" CACHE PATH "Contains library files exported by ROC Trunk")
else()
  set(ROCT_LIB_DIR "/opt/rocm/lib" CACHE PATH "Contains library files exported by ROC Trunk")
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

if ( NOT DEFINED CMAKE_PACKAGING_INSTALL_PREFIX )
    set (CMAKE_PACKAGING_INSTALL_PREFIX "/opt/rocm" )
endif ()

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
  set(ROCBLAS_INC_DIR "/opt/rocm/include")
  set(ROCBLAS_LIB_DIR "/opt/rocm/lib")
endif()

if (RVS_ROCMSMI EQUAL 1)
  set(ROCM_SMI_INC_DIR "${CMAKE_BINARY_DIR}/rvs_smi-src/include")
  set(ROCM_SMI_LIB_DIR "${CMAKE_BINARY_DIR}/rvs_smi-build")
else()
  set(ROCM_SMI_INC_DIR "/opt/rocm/rocm_smi/include")
  set(ROCM_SMI_LIB_DIR "/opt/rocm/rocm_smi/lib")
endif()
set(ROCM_SMI_LIB "rocm_smi64" CACHE STRING "rocm_smi library name")

set(RVS_BINTEST_FOLDER ${CMAKE_BINARY_DIR}/bintest)

add_subdirectory(rvs)
add_subdirectory(rvslib)
add_subdirectory(gm.so)
add_subdirectory(gpup.so)
add_subdirectory(gst.so)
add_subdirectory(iet.so)
add_subdirectory(pebb.so)
add_subdirectory(peqt.so)
add_subdirectory(pesm.so)
add_subdirectory(pqt.so)
add_subdirectory(rcqt.so)
add_subdirectory(smqt.so)
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

install(FILES "${CMAKE_BINARY_DIR}/doc/man/man1/rvs.1"
  DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs/man
  COMPONENT applications
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/userguide/html
  DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs/userguide
  COMPONENT applications
)

else()
  message("Doxygen is not found. Will not generate documentation.")
endif(DOXYGEN_FOUND)

if(RVS_ROCMSMI EQUAL 1)
  install(FILES "${ROCM_SMI_LIB_DIR}/librocm_smi64.so"
          DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs/lib
          COMPONENT rvsmodule)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rvsso.conf"
          DESTINATION /etc/ld.so.conf.d
          COMPONENT rvsmodule)
endif()  # if(RVS_ROCMSMI EQUAL 1)

# TEST SECTION

#add_test(NAME test1 COMMAND rvs)


include (CPack)
