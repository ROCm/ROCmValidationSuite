# NOTE: I know we can use find_library and find librvs... but that is a hacky
# workaround way instead of using more conventional find_package.

cmake_minimum_required(VERSION 3.20)

set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/"
    CACHE INTERNAL "Default module path.")

# ROCM_DIR should be passed in via command line
set(ROCM_DIR "/opt/rocm" CACHE PATH "ROCm directory.")

# needed to find rocm packages
list(APPEND CMAKE_PREFIX_PATH ${ROCM_DIR})

option(CMAKE_VERBOSE_MAKEFILE "Enable verbose output" ON)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands for linters and autocompleters" ON)

project(main LANGUAGES CXX)

option(BUILD_TESTS "Build test suite" OFF)
option(ADDRESS_SANITIZER "Enable address sanitizer" OFF)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED
    ON
    CACHE BOOL "")
set(CMAKE_CXX_EXTENSIONS
    OFF
    CACHE BOOL "")

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4.0)
    message("Compiler version is " ${CMAKE_CXX_COMPILER_VERSION})
    message(FATAL_ERROR "Require at least gcc-5.4.0")
endif()

# Include CMAKE_INSTALL_* variables
# this must go after project()
include(GNUInstallDirs)
# search for packages
include(FindPkgConfig)

add_executable(main main.cpp)

# Find RVS
#find_package(rvs CONFIG REQUIRED)

# Print expected variables for RVS package

# ISSUE: Only Version present
# maybe EXPECTED: Libraries and Include dirs to be filled
message(STATUS "rvs package found:")
message(STATUS "  Version: ${rvs_VERSION}")
message(STATUS "  Libraries: ${rvs_LIBRARIES}")
message(STATUS "  Include Dirs: ${rvs_INCLUDE_DIRS}")

# Print all variables starting with rvs or rocm

# ISSUE: Using these vars makes no sense here!!
# -- rocm_validation_suite_INCLUDE_DIR=/opt/rocm/include
# -- rocm_validation_suite_LIBRARY=rvslib
# -- rocm_validation_suite_LIB_DIR=/opt/rocm/lib
# EXPECTED: rvs::rvs to be a library TARGET
# ISSUE: Fails during linking! rvs target exists but it's useless because it's
# empty. This makes CMake step pass but compile step fail.
#target_link_libraries(main PUBLIC rvs)

# ISSUE: doesn't work, not expected but worth a try
# NOTE: can use find_library but that's awful
#target_link_libraries(main PUBLIC rvslib)

# ok use my workaround file
include(Findrvs)
# ISSUE: A whole bunch of dependencies are missing... need to find them.....
# EXPECTED: ... maybe these should these be found by RVS?
target_link_libraries(main PUBLIC rvs::rvs)

# this is my full workaround...
find_package(yaml-cpp REQUIRED) # NOTE!! THIS MUST BE 0.6.3 or older! Same issue as https://github.com/luxonis/depthai-ros/issues/5
find_package(hip REQUIRED)
find_package(hiprand REQUIRED)
find_package(hsa-runtime64 REQUIRED)
find_package(hsakmt REQUIRED)
find_package(rocblas REQUIRED)
find_package(hipblaslt REQUIRED)
find_package(amd_smi REQUIRED)
find_package(rvs REQUIRED)
find_library(rocm-core
    NAMES rocm-core
    REQUIRED)
set(COMBINED_LIBS roc::rocblas roc::hipblaslt hsakmt::hsakmt hsa-runtime64::hsa-runtime64 hip::amdhip64 hip::hiprand yaml-cpp amd_smi ${rocm-core})
target_link_libraries(main PUBLIC pthread dl rvs::rvs yaml-cpp pci m ${COMBINED_LIBS})

get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)
foreach(_variableName ${_variableNames})
    if(_variableName MATCHES "^(rocm|rvs)")
        message(STATUS "${_variableName}=${${_variableName}}")
    endif()
endforeach()

