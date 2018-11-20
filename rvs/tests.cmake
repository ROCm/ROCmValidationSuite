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


add_test(NAME unit.rvs.1
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvstest -d 3
)

add_test(NAME unit.rvs.cli1
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -g
)

add_test(NAME unit.rvs.cli2
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -t
)

add_test(NAME conf.gm1
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -d 3 -c conf/gm1.conf
)

# add tests
FOREACH(SINGLE_TEST ${TESTSOURCES})
  MESSAGE("${SINGLE_TEST}")
  string(REPLACE "test/" "unit.rvs." TMP_TEST_NAME ${SINGLE_TEST})
  string(REPLACE ".cpp" "" TEST_NAME ${TMP_TEST_NAME})
  MESSAGE("${TEST_NAME}")

  add_executable(${TEST_NAME} ${SINGLE_TEST} ${SOURCES})
  target_link_libraries(${TEST_NAME} ${PROJECT_LINK_LIBS} gtest_main gtest)

  MESSAGE("${TEST_NAME}")
  MESSAGE("${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

  add_test(NAME ${TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${TEST_NAME}
  )
ENDFOREACH()
