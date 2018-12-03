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


LINK_DIRECTORIES(${UT_LIB})
file(GLOB TESTSOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/test*.cpp )
#message ( "TESTSOURCES: ${TESTSOURCES}" )
set (UT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src/rvsloglp_utest.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/rvsliblogger.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/rvslognode.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/rvslognodebase.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/rvslognodeint.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/rvslognodestring.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/rvslognoderec.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/test/unitactionbase.cpp
    )
message("UT_SOURCES: ${UT_SOURCES}")

# add unit tests
FOREACH(SINGLE_TEST ${TESTSOURCES})
#  MESSAGE("${SINGLE_TEST}")
  string(REPLACE "test/test" "unit.${RVS}." TMP_TEST_NAME ${SINGLE_TEST})
  string(REPLACE ".cpp" "" TEST_NAME ${TMP_TEST_NAME})
  MESSAGE("unit test: ${TEST_NAME}")

  add_executable(${TEST_NAME}
    ${SINGLE_TEST} ${TEST_SOURCES} ${SOURCES} ${UT_SOURCES}
  )
  target_link_libraries(${TEST_NAME}
    ${PROJECT_LINK_LIBS} ${PROJECT_TEST_LINK_LIBS} gtest_main gtest
  )
  set_target_properties(${TEST_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY   ${RVS_BINTEST_FOLDER}
  )

  add_test(NAME ${TEST_NAME}
    WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
    COMMAND ${TEST_NAME}
  )
ENDFOREACH()

# add .conf file tests
add_test(NAME conf.pesm.0
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -d 3 -c conf/pesm.conf
)

add_test(NAME conf.pesm.1
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -d 3 -c conf/pesm1.conf
)

add_test(NAME conf.pesm.2
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -d 3 -c conf/pesm2.conf
)

add_test(NAME conf.pesm.3
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvsfail -d 3 -c conf/pesm3.conf
)

add_test(NAME conf.pesm.4
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvsfail -d 3 -c conf/pesm4.conf
)

add_test(NAME conf.pesm.5
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -d 3 -c conf/pesm5.conf
)

add_test(NAME conf.pesm.6
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -d 3 -c conf/pesm6.conf
)

add_test(NAME conf.pesm.7
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -d 3 -c conf/pesm7.conf
)

add_test(NAME conf.pesm.8
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -d 3 -c conf/pesm8.conf
)
