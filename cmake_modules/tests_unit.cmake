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


## define additional unit testing include directories
include_directories(${UT_INC})
## define additional unit testing lib directories
link_directories(${UT_LIB} ${RVS_LIB_DIR})

file(GLOB TESTSOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/test*.cpp )
#message ( "TESTSOURCES: ${TESTSOURCES}" )


# add unit tests
FOREACH(SINGLE_TEST ${TESTSOURCES})
#  MESSAGE("${SINGLE_TEST}")
  string(REPLACE "test/test_" "unit.${RVS}." TMP_TEST_NAME ${SINGLE_TEST})
  string(REPLACE ".cpp" "" TEST_NAME ${TMP_TEST_NAME})
  MESSAGE("unit test: ${TEST_NAME}")

  add_executable(${TEST_NAME}
    ${SINGLE_TEST} ${UT_SOURCES}
  )
  target_link_libraries(${TEST_NAME}
    ${UT_LINK_LIBS}  rvslibut rvslib gtest_main gtest pthread
  )
  target_compile_definitions(${TEST_NAME} PUBLIC RVS_UNIT_TEST)
  if(DEFINED tcd.${TEST_NAME})
    message(STATUS "tcd.${TEST_NAME} defined")
    message(STATUS "value: ${tcd.${TEST_NAME}}")
    target_compile_definitions(${TEST_NAME} PUBLIC ${tcd.${TEST_NAME}})
  endif()
  set_target_properties(${TEST_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY   ${RVS_BINTEST_FOLDER}
  )

  add_test(NAME ${TEST_NAME}
    WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
    COMMAND ${TEST_NAME}
  )
ENDFOREACH()
