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

## additional libraries
set ( PROJECT_TEST_LINK_LIBS ${PROJECT_LINK_LIBS} libpci.so)

## define test sources
set(TEST_SOURCES
   ../src/gpu_util.cpp ../src/pci_caps.cpp ../src/rvs_unit_testing_defs.cpp
   ../src/rvslognode.cpp ../src/rvslognodeint.cpp ../src/rvslognodestring.cpp ../src/rvslognoderec.cpp ../src/rvslognodebase.cpp
)

add_executable(rvstest ${SOURCES})
target_link_libraries(rvstest ${PROJECT_LINK_LIBS} gtest_main gtest)
set_target_properties(rvstest PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY   ${RVS_BINTEST_FOLDER}
)

add_test(NAME unit.rvs.1
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvstest -d 3
)

add_test(NAME unit.rvs.cli1
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvs -g
)

add_test(NAME unit.rvs.cli2
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvs -t
)


# add tests
FOREACH(SINGLE_TEST ${TESTSOURCES})
  MESSAGE("${SINGLE_TEST}")
  string(REPLACE "test/" "unit.rvs." TMP_TEST_NAME ${SINGLE_TEST})
  string(REPLACE ".cpp" "" TEST_NAME ${TMP_TEST_NAME})
  MESSAGE("${TEST_NAME}")

  add_executable(${TEST_NAME} ${SINGLE_TEST} ${TEST_SOURCES} ${SOURCES})
  target_link_libraries(${TEST_NAME} ${PROJECT_LINK_LIBS} ${PROJECT_TEST_LINK_LIBS} gtest_main gtest)
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
