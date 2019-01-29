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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/novernum.config ${RVS_BINTEST_FOLDER}/novernum.config)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/wrongvernum.config ${RVS_BINTEST_FOLDER}/wrongvernum.config)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/empty.config ${RVS_BINTEST_FOLDER}/empty.config)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nomodule.config ${RVS_BINTEST_FOLDER}/nomodule.config)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/testif.config ${RVS_BINTEST_FOLDER}/testif.config)


# add_test(NAME unit.rvs.1
#   WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
#   COMMAND rvstest -d 3
# )

add_test(NAME unit.rvs.cli1
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -g
)

add_test(NAME unit.rvs.cli2
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -t
)

add_test(NAME unit.rvs.cli3
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs --version
)

add_test(NAME unit.rvs.cli4
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -h
)

add_test(NAME unit.rvs.cli5
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -q
)

add_test(NAME unit.rvs.cli6
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -v
)

add_test(NAME unit.rvs.cli7
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -i all -c conf/gpup_6.conf
)

add_test(NAME unit.rvs.cli8
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND rvs -i 6255,3254 -c conf/gpup_6.conf
)

add_test(NAME unit.ttf.rvs.cli.1
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -d adf
)

add_test(NAME unit.ttf.rvs.cli.2
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -d 7
)

add_test(NAME unit.ttf.rvs.cli.3
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -d -7
)

add_test(NAME unit.ttf.rvs.cli.4
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -c xxx
)

add_test(NAME unit.ttf.rvs.cli.5
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -l //
)

add_test(NAME unit.ttf.rvs.cli.6
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -d
)

add_test(NAME unit.ttf.rvs.cli.7
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -c conf/broken_no_module.conf 
)

add_test(NAME unit.ttf.rvs.cli.8
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -c conf/broken_wrong_module.conf 
)

add_test(NAME unit.ttf.rvs.cli.9
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND ${CMAKE_SOURCE_DIR}/regression/run_no_config.sh
)

add_test(NAME unit.ttf.rvs.cli.10
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -m /
)

add_test(NAME unit.ttf.rvs.cli.11
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail -d -d
)

add_test(NAME unit.ttf.rvs.cli.12
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND rvsfail d
)

add_test(NAME unit.ttf.rvs.config.empty
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND ${CMAKE_SOURCE_DIR}/regression/run_config.sh empty.config
)

add_test(NAME unit.ttf.rvs.config.wrongvernum
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND ${CMAKE_SOURCE_DIR}/regression/run_config.sh wrongvernum.config
)

add_test(NAME unit.ttf.rvs.config.novernum
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND ${CMAKE_SOURCE_DIR}/regression/run_config.sh novernum.config
)

add_test(NAME unit.ttf.rvs.config.nomodule
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND ${CMAKE_SOURCE_DIR}/regression/run_config.sh nomodule.config
)

add_test(NAME unit.ttf.rvs.config.noconfig
  WORKING_DIRECTORY ${RVS_BINTEST_FOLDER}
  COMMAND ${CMAKE_SOURCE_DIR}/regression/run_no_config.sh
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
  string(REPLACE "test/test_" "unit.rvs." TMP_TEST_NAME ${SINGLE_TEST})
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
