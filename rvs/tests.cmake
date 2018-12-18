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

# add_executable(rvstest test/test1.cpp)
# target_link_libraries(rvstest rvshelper ${PROJECT_LINK_LIBS} gtest_main gtest libpci.so)
# set_target_properties(rvstest PROPERTIES
#   RUNTIME_OUTPUT_DIRECTORY   ${RVS_BINTEST_FOLDER}
# )
# add_dependencies(rvstest rvshelper)

## define target for "test-to-fail"
add_executable(${RVS_TARGET}fail src/rvs.cpp)
target_link_libraries(${RVS_TARGET}fail librvshelper.a rvslib ${PROJECT_LINK_LIBS} )
target_compile_definitions(${RVS_TARGET}fail PRIVATE RVS_INVERT_RETURN_STATUS)
set_target_properties(${RVS_TARGET}fail PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY   ${RVS_BINTEST_FOLDER}
)
add_dependencies(${RVS_TARGET}fail rvshelper)


# add_test(NAME unit.rvs.1
#   WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
#   COMMAND rvstest -d 3
# )

add_test(NAME unit.rvs.cli1
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvs -g
)

add_test(NAME unit.rvs.cli2
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvs -t
)


## define include directories
include_directories(${UT_INC})
## define lib directories
link_directories(${UT_LIB})
## additional libraries for unit tests
set (PROJECT_TEST_LINK_LIBS ${PROJECT_LINK_LIBS} libpci.so)

## define unit testing targets
file(GLOB TESTSOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/test*.cpp )
#message ( "TESTSOURCES ${TESTSOURCES}" )

# add tests
FOREACH(SINGLE_TEST ${TESTSOURCES})
#  MESSAGE("${SINGLE_TEST}")
  string(REPLACE "test/" "unit.rvs." TMP_TEST_NAME ${SINGLE_TEST})
  string(REPLACE ".cpp" "" TEST_NAME ${TMP_TEST_NAME})
  MESSAGE("unit test: ${TEST_NAME}")

  add_executable(${TEST_NAME} ${SINGLE_TEST}
    )
  target_link_libraries(${TEST_NAME}
    ${PROJECT_LINK_LIBS}
    ${PROJECT_TEST_LINK_LIBS}
    rvshelper rvslib rvslibut gtest_main gtest pthread
  )
  target_compile_definitions(${TEST_NAME} PRIVATE RVS_UNIT_TEST)
  add_compile_options(-Wall -Wextra -save-temps)
  set_target_properties(${TEST_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY   ${RVS_BINTEST_FOLDER}
  )
  add_test(NAME ${TEST_NAME}
    WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
    COMMAND ${TEST_NAME}
  )
ENDFOREACH()
