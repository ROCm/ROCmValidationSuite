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
project ("rocm-validation-suite" )
enable_testing()
include ( CTest )

## Support for in-source compilaton
if ( ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin" )
else()
    set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )
endif ()

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

# Set PATH variables
set(rvs_build_type "Debug")
set(RVS_DO_TRACE "1" CACHE STRING "Expand RVSTRACE_ macro")
set(RVS_ROCBLAS "0" CACHE STRING "1 = use local rocBLAS")
set(RVS_ROCMSMI "1" CACHE STRING "1 = use local rocm_smi_lib")

set(YAML_INC_DIR "${CMAKE_BINARY_DIR}/yaml-src/include/" CACHE PATH "Contains header files exported by yaml-cpp")
set(YAML_LIB_DIR "${CMAKE_BINARY_DIR}/yaml-build/" CACHE PATH "Contains library files exported by yaml-cpp")
set(ROCR_INC_DIR "/opt/rocm/include/" CACHE PATH "Contains header files exported by ROC Runtime")
set(ROCR_LIB_DIR "/opt/rocm/lib/" CACHE PATH "Contains library files exported by ROC Runtime")
set(ROCT_INC_DIR "/opt/rocm/include/libhsakmt" CACHE PATH "Contains header files exported by ROC Trunk")
set(ROCT_LIB_DIR "/opt/rocm/lib/" CACHE PATH "Contains library files exported by ROC Trunk")
set(HIP_INC_DIR "/opt/rocm/hip/include/hip/" CACHE PATH "Contains header files exported by ROC Runtime")

message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE ${rvs_build_type} CACHE STRING "Build type" FORCE)
    # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_definitions(-DDEBUG)
    message(STATUS "DEBUG defined")
endif()

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

if (DEFINED RVS_DO_TRACE)
  if (${RVS_DO_TRACE} STREQUAL "1")
    add_definitions(-DRVS_DO_TRACE)
    message(STATUS "RVS_DO_TRACE defined")
  else()
      message(STATUS "RVS_DO_TRACE not defined")
  endif()
else()
    message(STATUS "RVS_DO_TRACE not defined")
endif( )


## Set default unit test framework include path
if ( NOT DEFINED UT_INC )
    set ( UT_INC "${CMAKE_BINARY_DIR}/googletest-src/googletest/include" )
    message ( "UT_INC ${UT_INC}" )
endif ()

## Set default unit test framework include path
if ( NOT DEFINED UT_LIB )
    set ( UT_LIB "${CMAKE_BINARY_DIR}/googletest-build/lib" )
    message ( "UT_LIB ${UT_LIB}" )
endif ()

## Set default module path if not already set
if ( NOT DEFINED CPACK_GENERATOR )
    set ( CPACK_GENERATOR "DEB;RPM" )
endif ()

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

# post build step to create symbolic link
# (we want .so i /bin folder to simplify packaging when local smi lib is used)
# add_custom_command(TARGET rvs_rsmi_target POST_BUILD
#   COMMAND rm -f ${CMAKE_BINARY_DIR}/bin/librocm_smi64*
#   COMMAND ln -s ${CMAKE_BINARY_DIR}/rvs_smi-build/librocm_smi64.so ${CMAKE_BINARY_DIR}/bin/librocm_smi64.so
# )

endif()  # if (RVS_ROCMSMI EQUAL 1)

if (RVS_ROCBLAS EQUAL 1)
  set(ROCBLAS_INC_DIR "${CMAKE_BINARY_DIR}/rvs_rblas-src/build/release/rocblas-install/include/")
  set(ROCBLAS_LIB_DIR "${CMAKE_BINARY_DIR}/rvs_rblas-src/build/release/rocblas-install/lib/")
else()
  set(ROCBLAS_INC_DIR "/opt/rocm/include/")
  set(ROCBLAS_LIB_DIR "/opt/rocm/lib/")
endif()

if (RVS_ROCMSMI EQUAL 1)
  set(ROCM_SMI_INC_DIR "${CMAKE_BINARY_DIR}/rvs_smi-src/include")
  set(ROCM_SMI_LIB_DIR "${CMAKE_BINARY_DIR}/rvs_smi-build")
else()
  set(ROCM_SMI_INC_DIR "/opt/rocm/include/")
  set(ROCM_SMI_LIB_DIR "/opt/rocm/lib/")
endif()
set(ROCM_SMI_LIB "rocm_smi64" CACHE STRING "rocm_smi library name")

add_subdirectory(rvs)
add_subdirectory(gpup.so)
add_subdirectory(pesm.so)
add_subdirectory(rcqt.so)
add_subdirectory(peqt.so)
add_subdirectory(smqt.so)
add_subdirectory(pebb.so)
add_subdirectory(gm.so)
add_subdirectory(gst.so)
add_subdirectory(pqt.so)
add_subdirectory(iet.so)

add_dependencies(rvstest rvs_gtest_target)
add_dependencies(rvs rvs_doc)
add_dependencies(rvs rvs_yaml_target)
add_dependencies(rvsfail rvs_yaml_target)
add_dependencies(rvstest rvs_yaml_target)

if(RVS_ROCMSMI EQUAL 1)
  add_dependencies(gm rvs_rsmi_target)
  add_dependencies(iet rvs_rsmi_target)
endif()

if(RVS_ROCBLAS EQUAL 1)
  add_dependencies(gst rvs_rblas_target)
  add_dependencies(iet rvs_rblas_target)
endif()

add_custom_target(rvs_doc ALL
COMMAND ${CMAKE_COMMAND} -E make_directory "doc"
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
COMMENT "Create the doc directory"
VERBATIM)

add_custom_target(progguide ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} doxygen
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS Programmer's Reference Guide"
    VERBATIM)
add_dependencies(progguide rvs_doc)

add_custom_target(man1 ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} doxygen DoxyMan1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS man.1 files"
    VERBATIM)
add_dependencies(man1 rvs_doc)

add_custom_target(userguide ALL
    COMMAND RVSVER=${VERSION_STRING} RVSBUILD=${CMAKE_BINARY_DIR} doxygen DoxyGuide
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating RVS User Guide"
    VERBATIM)
add_dependencies(userguide rvs_doc)

if(RVS_ROCMSMI EQUAL 1)
  install(FILES "${ROCM_SMI_LIB_DIR}/librocm_smi64.so"
                "${ROCM_SMI_LIB_DIR}/librocm_smi64.so.1"
                "${ROCM_SMI_LIB_DIR}/librocm_smi64.so.1.0.0"
          DESTINATION ${CMAKE_PACKAGING_INSTALL_PREFIX}/rvs/lib
          COMPONENT rvsmodule)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rvsso.conf"
          DESTINATION /etc/ld.so.conf.d
          COMPONENT rvsmodule)
endif()  # if(RVS_ROCMSMI EQUAL 1)

# TEST SECTION

#add_test(NAME test1 COMMAND rvs)


include (CPack)
