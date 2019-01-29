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

set(CTEST_SOURCE_DIRECTORY "ROCmValidationSuite")
set(CTEST_BINARY_DIRECTORY "build")

set(CTEST_SITE "prj47-rack-07")
set(CTEST_BUILD_NAME "${RVS_HOST} ${RVS_TAG} ${RVS_BRANCH} ${CTEST_BUILD_CONFIGURATION}")
if (RVS_BUILD_TESTS)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME} TB")
endif()
if (WITH_TESTING)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME} TR")
endif()
if (RVS_COVERAGE)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME} C")
endif()
if(RVS_ROCBLAS EQUAL 1)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME} rocBLAS")
endif()
if(RVS_ROCMSMI EQUAL 1)
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME} rocm_smi")
endif()


set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_OPTIONS "")

set(WITH_MEMCHECK FALSE)

#######################################################################

#ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

find_program(CTEST_GIT_COMMAND NAMES git)
find_program(CTEST_COVERAGE_COMMAND NAMES gcov)
find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)

set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE ${CTEST_SOURCE_DIRECTORY}/tests/valgrind.supp)

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
    message(STATUS "Source directory does not exist.")
#    return()
  set(CTEST_CHECKOUT_COMMAND "${CTEST_GIT_COMMAND} clone  -b ${RVS_BRANCH} https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git ${CTEST_SOURCE_DIRECTORY}")
endif()

# if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
#     message(FATAL_ERROR "Build directory does not exist.")
#     return()
# #  set(CTEST_CHECKOUT_COMMAND "${CTEST_GIT_COMMAND} clone git://git.libssh.org/projects/libssh/libssh.git ${CTEST_SOURCE_DIRECTORY}")
# endif()

set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} -DRVS_BUILD_TESTS:BOOL=${RVS_BUILD_TESTS}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} -DRVS_COVERAGE:BOOL=${RVS_COVERAGE}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} -DRVS_ROCBLAS=${RVS_ROCBLAS}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} -DRVS_ROCMSMI=${RVS_ROCMSMI}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} ${CTEST_BUILD_OPTIONS}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"-G${CTEST_CMAKE_GENERATOR}\"")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"../${CTEST_SOURCE_DIRECTORY}\"")

ctest_start(${RVS_CTEST_BUILD_TYPE})
ctest_update()
ctest_configure()
ctest_build()
if (WITH_TESTING)
  if (CTEST_INCLUDE)
    message("Tests to run: ${CTEST_INCLUDE}")
    ctest_test(INCLUDE ${CTEST_INCLUDE})
  else()
    ctest_test()
  endif()
endif(WITH_TESTING)

set(CTEST_COVERAGE_EXTRA_FLAGS " -s $ENV{RVS_WB}/build")
# set(CTEST_COVERAGE_EXTRA_FLAGS " -s $ENV{RVS_WB}/build/yaml-src/include/yaml-cpp/")
# set(CTEST_COVERAGE_EXTRA_FLAGS "${CTEST_COVERAGE_EXTRA_FLAGS} -s $ENV{RVS_WB}/build/googletest-src")

if (RVS_COVERAGE AND CTEST_COVERAGE_COMMAND)
  ctest_coverage()
endif ()
# if (WITH_MEMCHECK AND CTEST_MEMORYCHECK_COMMAND)
#   ctest_memcheck()
# endif (WITH_MEMCHECK AND CTEST_MEMORYCHECK_COMMAND)
ctest_submit()
